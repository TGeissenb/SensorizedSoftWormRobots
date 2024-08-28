import time

import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def closest_point(arr, point, threshold=1000.0):
    dist = (arr[:, 0] - point[0])**2 + (arr[:, 1] - point[1])**2
    idx = dist.argmin()
    if dist[idx] > threshold:
        idx = None
    return idx


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


def compute_curvature(img, n, plotting=True):

    # Skeletonize
    skeleton = skeletonize(img, method="lee")
    skeleton_mask = skeleton * img

    # Compute CG row-wise
    mask = img > 127
    """
    idcs = np.array([range(0, mask.shape[1]) for _ in range(mask.shape[0])])
    masked_idcs = mask * idcs

    masked_idcs = masked_idcs.astype(np.float32)
    masked_idcs[masked_idcs == 0] = np.nan
    masked_idcs_mean = np.nanmean(masked_idcs, axis=1)
    """

    # Compute CG row-wise via scipy
    cgs_r = []
    for i in range(mask.shape[0]):
        lbl = ndimage.label(mask[i])[0]
        cgs_r.append(ndimage.center_of_mass(mask[i], lbl, range(1, n+1)))
    cgs_r = np.array(cgs_r).squeeze()

    # Extract data points
    xx = []
    yy = []
    for i in range(n):
        xx.append(cgs_r[:, i].tolist())
        yy.append(list(range(mask.shape[0])))
    
    # Compute CG column-wise
    """
    idcs = np.array([range(0, mask.shape[0]) for _ in range(mask.shape[1])])
    masked_idcs = mask.T * idcs

    masked_idcs = masked_idcs.astype(np.float32)
    masked_idcs[masked_idcs == 0] = np.nan
    masked_idcs_mean = np.nanmean(masked_idcs, axis=1)
    """

    # Compute CG column-wise via scipy
    cgs_c = []
    mask_t = mask.T
    for i in range(mask.shape[1]):
        lbl = ndimage.label(mask_t[i])[0]
        cgs_c.append(ndimage.center_of_mass(mask_t[i], lbl, range(1, n + 1)))
    cgs_c = np.array(cgs_c).squeeze()

    # Extract data points
    for i in range(n):
        xx.append(list(range(mask_t.shape[0])))
        yy.append(cgs_c[:, i].tolist())

    xx = flatten_concatenation(xx)
    yy = flatten_concatenation(yy)

    # Reformatting
    xx = np.array(xx).reshape(1, -1)
    yy = np.array(yy).reshape(1, -1)
    yy = yy[~np.isnan(xx)]
    xx = xx[~np.isnan(xx)]
    xx = xx[~np.isnan(yy)]
    yy = yy[~np.isnan(yy)]
    points = np.vstack((xx, yy)).T
    points = np.unique(points, axis=0)

    # Sort points
    points_ordered = [points[0, :]]
    points = np.delete(points, 0, axis=0)
    idx = closest_point(points, points_ordered[-1])
    while idx is not None and points.shape[0] > 1:
        points_ordered.append(points[idx, :])
        points = np.delete(points, idx, axis=0)
        idx = closest_point(points, points_ordered[-1])

    # Search in other direction
    idx = closest_point(points, points_ordered[0])
    while idx is not None and points.shape[0] > 1:
        points_ordered.insert(0, points[idx, :])
        points = np.delete(points, idx, axis=0)
        idx = closest_point(points, points_ordered[0])

    points_ordered = np.array(points_ordered)

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points_ordered, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    # Interpolation for different methods:
    s = 1
    n = 10000
    alpha = np.linspace(0, s, n)

    interpolator = interp1d(distance, points_ordered, kind="cubic", axis=0)
    interpolated_points = interpolator(alpha)

    # Compute curvature locally
    h = s/n
    dp = np.array([(interpolated_points[i + 1, :] - interpolated_points[i, :]) / h for i in range(n-1)])
    ddp = np.array([(dp[i + 1, :] - dp[i, :]) / h for i in range(n-2)])

    dp = np.array([(interpolated_points[i + 1, :] + interpolated_points[i, :]) / 2. for i in range(n-2)])

    curv = np.array(abs(dp[:, 0] * ddp[:, 1] - dp[:, 1] * ddp[:, 0]) / np.sqrt((dp[:, 0] ** 2 + dp[:, 1] ** 2) ** 3))

    # Plot graph
    fig = None
    if plotting:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(skeleton_mask)
        #plt.scatter(*points_ordered.T, 0.1, 'r')
        ax.scatter(*points.T, 0.1, 'k')
        threshold = np.percentile(np.abs(curv), 80)
        scat = ax.scatter(*interpolated_points[1:-1].T, c=curv, s=1, cmap='cool', vmin=0, vmax=threshold)
        fig.colorbar(scat, label='Curvature')

    return curv, fig


if __name__ == "__main__":
    img = cv2.imread(r"C:\Users\tobia\Uni\Skripten\Master-Thesis\06 FEM\Abaqus Whole Worm\Empirical Data\D8_18.6psi.jpg", cv2.IMREAD_COLOR)
    plt.imshow(img)

    theta = np.linspace(0, 46 * np.pi / 180, 100)
    plt.plot(1525 + 2900 * (np.cos(theta) - 1), 1275 + 2900 * np.sin(theta), linewidth=7)

    plt.show()
    """
    def mouseRGB(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
            colors = img[x, y]
            print("BRG Format: ", colors)

    cv2.namedWindow('mouseRGB')
    cv2.setMouseCallback('mouseRGB', mouseRGB)
    imS = cv2.resize(img, (960, 540))
    cv2.imshow('mouseRGB', imS)

    while True:
        if cv2.waitKey(1) == 27:
            break
    
    diff = 50
    lower_color_bounds = np.array((175 - diff, 175 - diff, 175 - diff), dtype=np.uint8, ndmin=1)
    upper_color_bounds = np.array((175 + diff, 175 + diff, 175 + diff), dtype=np.uint8, ndmin=1)

    mask = cv2.inRange(img, lower_color_bounds, upper_color_bounds)
    cv2.imshow("Mask", mask)
    curvature, fig = compute_curvature(mask, 10, plotting=True)
    """

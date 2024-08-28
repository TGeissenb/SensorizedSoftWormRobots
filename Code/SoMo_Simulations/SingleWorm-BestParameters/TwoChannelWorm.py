import random

import pybullet as p
import pybullet_data
import json
from datetime import datetime
import numpy as np

import os
import sys

path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "..", "..")
)  # this is a bit hacky... just in case the user doesnt have somo installed...
sys.path.insert(0, path)

from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.sm_actuator_definition import SMActuatorDefinition
from somo.sm_link_definition import SMLinkDefinition
from somo.sm_joint_definition import SMJointDefinition
from somo.sm_continuum_manipulator import SMContinuumManipulator

from somo.utils import load_constrained_urdf

import sorotraj

######## SIMULATION SETUP ########
def simulate(params):
    VIDEO_LOGGING = params["VIDEO_LOGGING"]
    GUI = params["GUI"]
    num_obstacles_x = params["num_obstacles_x"]
    num_obstacles_y = params["num_obstacles_y"]
    spacing_x = params["spacing_x"]
    spacing_y = params["spacing_y"]
    base_length = params["base_length"]
    actuator_length = params["actuator_length"]
    base = params["base"]
    mass = params["mass"]
    center_of_mass = params["center_of_mass"]
    stiffness = params["stiffness"]
    gravity = params["gravity"]
    time_step = params["time_step"]
    n_steps = params["n_steps"]
    ground_friction = params["ground_friction"]

    ### prepare everything for the physics client / rendering
    ## Pretty rendering
    opt_str = "--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0"  # this opens the gui with a white background and no ground grid
    cam_width, cam_height = 1920, 1640
    if cam_width is not None and cam_height is not None:
        opt_str += " --width=%d --height=%d" % (cam_width, cam_height)

    physicsClient = p.connect(
        GUI, options=opt_str
    )  # starts the physics client with the options specified above. replace p.GUI with p.DIRECT to avoid gui

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Search for URDFs in pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set the camera position. This goes right after you instantiate the GUI:
    cam_distance, cam_yaw, cam_pitch, cam_xyz_target = 50.0, 70.0, -45.0, [10.0, -40.0, 2.4]
    p.resetDebugVisualizerCamera(
        cameraDistance=cam_distance,
        cameraYaw=cam_yaw,
        cameraPitch=cam_pitch,
        cameraTargetPosition=cam_xyz_target,
    )

    ## Set physics parameters and simulation properties
    p.setGravity(0, 0, gravity)
    p.setPhysicsEngineParameter(enableConeFriction=1)
    p.setRealTimeSimulation(0)  # this is necessary to enable torque control. only if this is set to 0 and the simulation is done with explicit steps will the torque control work correctly

    ## Specify time steps
    p.setTimeStep(time_step)

    ### load all the objects into the environment
    # load the ground plane
    planeId = p.loadURDF("plane.urdf", globalScaling=10, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
    p.changeDynamics(planeId, -1, lateralFriction=1)  # set ground plane friction
    p.changeDynamics(planeId, -1, restitution=0.2)
    p.changeVisualShape(planeId, -1, rgbaColor=[0.8, 0.8, 0.8, 0.3])


    ### Create and load the manipulator / worm
    # load the manipulator definition
    worm_manipulator_def = SMManipulatorDefinition.from_file("definitions/worm_short.yaml")
    if base:
        worm_manipulator_def.actuator_definitions[0].actuator_length = base_length
        worm_manipulator_def.actuator_definitions[0].n_segments = int(100 * base_length)
        worm_manipulator_def.actuator_definitions[0].link_definition.mass = mass * base_length/(base_length+actuator_length) * (1 - center_of_mass)
        worm_manipulator_def.actuator_definitions[1].actuator_length = actuator_length
        worm_manipulator_def.actuator_definitions[1].n_segments = int(100 * actuator_length)
        worm_manipulator_def.actuator_definitions[0].link_definition.mass = mass * actuator_length/(base_length+actuator_length) * center_of_mass
    else:
        worm_manipulator_def.actuator_definitions[0].actuator_length = actuator_length/2
        worm_manipulator_def.actuator_definitions[0].n_segments = int(100 * actuator_length/2)
        worm_manipulator_def.actuator_definitions[0].link_definition.mass = mass * (1 - center_of_mass)
        worm_manipulator_def.actuator_definitions[1].actuator_length = actuator_length/2
        worm_manipulator_def.actuator_definitions[1].n_segments = int(100 * actuator_length/2)
        worm_manipulator_def.actuator_definitions[1].link_definition.mass = mass * center_of_mass

    for actuator in worm_manipulator_def.actuator_definitions:
        for joint in actuator.joint_definitions:
            joint.spring_stiffness = stiffness

    # create the worm manipulator...
    worm = SMContinuumManipulator(worm_manipulator_def)

    obstacle_id = p.createCollisionShape(p.GEOM_CYLINDER,
                                         fileName=r"/bullet3/data/cube.obj")
    for i in range(num_obstacles_y):
        for j in range(num_obstacles_x):
            p.createMultiBody(baseMass=1000,
                              baseCollisionShapeIndex=obstacle_id,
                              basePosition=[spacing_x * (j - num_obstacles_x/2 + 0.5), -spacing_y * i + spacing_y/2*j, 1])

    # ... and load it
    startPos = [0, 0, 1]
    startOr = p.getQuaternionFromEuler([1, 0, 0])
    worm.load_to_pybullet(
        baseStartPos=startPos,
        baseStartOrn=startOr,
        baseConstraint="free",  # other options are free and constrained, but those are not recommended rn
        physicsClient=physicsClient,
        flags=p.URDF_USE_SELF_COLLISION,
    )

    # below is an example of how lateral friction and restitution can be changed for the whole manipulator.
    contact_properties = {
        "lateralFriction": ground_friction,
        # 'restitution': 0.0, # uncomment to change restitution
    }
    worm.set_contact_property(contact_properties)


    ######## PRESCRIBE A TRAJECTORY ########
    # here, the trajectory is hard-coded (booh!) and prepared using the sorotraj format
    traj = sorotraj.TrajBuilder(graph=False)

    traj.load_traj_def("bidirectional")
    trajectory = traj.get_trajectory()
    interp = sorotraj.Interpolator(trajectory)
    actuation_fn = interp.get_interp_function(
        num_reps=50, speed_factor=1, invert_direction=False, as_list=False
    )


    ######## EXECUTE SIMULATION ########
    # if desired, start video logging - this goes before the run loop
    if VIDEO_LOGGING:
        vid_filename = "vid.mp4"
        logIDvideo = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, vid_filename)

    worm_pos_time = []
    worm_or_time = []

    # this for loop is the actual simulation
    for i in range(n_steps):

        wormPos, wormOr = p.getBasePositionAndOrientation(worm.bodyUniqueId)
        wormOr = p.getEulerFromQuaternion(wormOr)
        worm_pos_time.append(wormPos)
        worm_or_time.append(wormOr)

        rotation = np.array([[np.cos(wormOr[1]), np.sin(wormOr[1])],
                             [-np.sin(wormOr[1]), np.cos(wormOr[1])]])

        torques = actuation_fn(i * time_step) * 1  # retrieve control torques from the trajectory.

        # applying the control torques
        if base:
            worm.apply_actuation_torques(
                actuator_nrs=[0, 0, 1, 1],
                axis_nrs=[0, 1, 0, 1],
                actuation_torques=[0, 0] + np.matmul(torques, rotation).tolist(),
            )
        else:
            worm.apply_actuation_torques(
                actuator_nrs=[0, 0, 1, 1],
                axis_nrs=[0, 1, 0, 1],
                actuation_torques=np.matmul(torques, rotation).tolist() + np.matmul(torques, rotation).tolist(),
            )
        """
        p.resetDebugVisualizerCamera(
            cameraDistance=10,
            cameraYaw=180,
            cameraPitch=-20,
            cameraTargetPosition=wormPos,
        )
        """
        #p.applyExternalForce(worm.bodyUniqueId, -1, [0, -20, 0], [0, 0, 0], p.WORLD_FRAME)
        print("Timestep: {0:d},\ttime: {1:.2f}s".format(i, i*time_step))
        p.stepSimulation()


    ######## CLEANUP AFTER SIMULATION ########
    # this goes after the run loop
    if VIDEO_LOGGING:
        p.stopStateLogging(logIDvideo)
    # ... aaand disconnect pybullet
    p.disconnect()

    # Save parameters and trajectories
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    parameters = {"Parameters": params,
                  "Blob Trajectory": worm_pos_time,
                  "Orientation": worm_or_time}

    with open(os.path.join("Logs_Unidirectional", date_time + '_Blob_Log.json'), 'w') as json_file:
        json.dump(parameters, json_file)


if __name__ == '__main__':

    # select whether you want to record a video or not
    VIDEO_LOGGING = False
    GUI = p.GUI

    for i in range(1):

        # select number of obstacles
        num_obstacles_x = 4
        num_obstacles_y = 15
        spacing_x = 3.5
        spacing_y = 6

        # select physical parameters
        gravity = -9.81
        ground_friction_coefficient = 0.7

        # select worm geometry
        base_length = 0.1
        actuator_length = 0.15
        base = True
        mass = 3
        center_of_mass = 0.05  # 0: (0.25) tethered; 1: (0.75) tip; 0.5: balanced
        stiffness = 500

        ## Specify time steps
        time_step = 0.001
        n_steps = 120000

        # set parameters
        params = {"VIDEO_LOGGING": VIDEO_LOGGING,
                  "GUI": GUI,
                  "num_obstacles_x": num_obstacles_x,
                  "num_obstacles_y": num_obstacles_y,
                  "spacing_x": spacing_x,
                  "spacing_y": spacing_y,
                  "gravity": gravity,
                  "ground_friction": ground_friction_coefficient,
                  "base_length": base_length,
                  "actuator_length": actuator_length,
                  "base": base,
                  "mass": mass,
                  "center_of_mass": center_of_mass,
                  "stiffness": stiffness,
                  "time_step": time_step,
                  "n_steps": n_steps}

        simulate(params)

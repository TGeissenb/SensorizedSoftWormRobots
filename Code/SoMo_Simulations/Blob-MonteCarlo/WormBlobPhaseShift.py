import pybullet as p
import pybullet_data
import random
import numpy as np
import json

import os
import sys
from datetime import datetime


path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "..", "..")
)  # this is a bit hacky... just in case the user doesnt have somo installed...
sys.path.insert(0, path)

random.seed(10)

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
    height = params["obs_height"]
    spacing_x = params["spacing_x"]
    spacing_y = params["spacing_y"]
    num_wiggler = params["num_wiggler"]
    num_puller = params["num_puller"]
    start_pos_range = params["start_pos_range"]
    start_orient_range = params["start_orient_range"]
    base_length = params["base_length"]
    actuator_length = params["actuator_length"]
    base = params["base"]
    mass = params["mass"]
    center_of_mass = params["center_of_mass"]
    stiffness = params["stiffness"]
    torque = params["torque"]
    phase_shift = params["phase_shift"]
    gravity = params["gravity"]
    ground_friction_coefficient = params["ground_friction"]
    time_step = params["time_step"]
    n_steps = params["n_steps"]

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
    cam_distance, cam_yaw, cam_pitch, cam_xyz_target = 50.0, 10.0, -40.0, [0.0, 0.0, 2.4]
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

    ### load all the objects into the environment
    # load the ground plane
    planeId = p.loadURDF("plane.urdf", flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL, globalScaling=10)
    p.changeDynamics(planeId, -1, lateralFriction=ground_friction_coefficient)  # set ground plane friction
    p.changeDynamics(planeId, -1, restitution=0)
    p.changeVisualShape(planeId, -1, rgbaColor=[0.8, 0.8, 0.8, 0.3])


    ### Create and load the manipulator / worm
    # load the manipulator definition
    worm_manipulator_def = SMManipulatorDefinition.from_file("definitions/worm.yaml")
    worm_manipulator_def_wiggler = SMManipulatorDefinition.from_file("definitions/worm_wiggler.yaml")

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

    if base:
        worm_manipulator_def_wiggler.actuator_definitions[0].actuator_length = base_length
        worm_manipulator_def_wiggler.actuator_definitions[0].n_segments = int(100 * base_length)
        worm_manipulator_def_wiggler.actuator_definitions[0].link_definition.mass = mass * base_length/(base_length+actuator_length) * (1 - center_of_mass)
        worm_manipulator_def_wiggler.actuator_definitions[1].actuator_length = actuator_length
        worm_manipulator_def_wiggler.actuator_definitions[1].n_segments = int(100 * actuator_length)
        worm_manipulator_def_wiggler.actuator_definitions[0].link_definition.mass = mass * actuator_length/(base_length+actuator_length) * center_of_mass
    else:
        worm_manipulator_def_wiggler.actuator_definitions[0].actuator_length = actuator_length/2
        worm_manipulator_def_wiggler.actuator_definitions[0].n_segments = int(100 * actuator_length/2)
        worm_manipulator_def_wiggler.actuator_definitions[0].link_definition.mass = mass * (1 - center_of_mass)
        worm_manipulator_def_wiggler.actuator_definitions[1].actuator_length = actuator_length/2
        worm_manipulator_def_wiggler.actuator_definitions[1].n_segments = int(25 * actuator_length/2)
        worm_manipulator_def_wiggler.actuator_definitions[1].link_definition.mass = mass * center_of_mass

    for actuator in worm_manipulator_def_wiggler.actuator_definitions:
        for joint in actuator.joint_definitions:
            joint.spring_stiffness = stiffness

    # create the worm manipulator...
    wiggler_worms = []
    for i in range(num_wiggler):
        wiggler_worms.append(SMContinuumManipulator(worm_manipulator_def_wiggler))

    puller_worms = []
    for i in range(num_puller):
        puller_worms.append(SMContinuumManipulator(worm_manipulator_def))

    obstacle_id = p.createCollisionShape(p.GEOM_CYLINDER,
                                         fileName=r"/bullet3/data/cube.obj",
                                         height=height)
    for i in range(num_obstacles_x):
        for j in range(num_obstacles_y):
            p.createMultiBody(baseMass=1000000,
                              baseCollisionShapeIndex=obstacle_id,
                              basePosition=[spacing_x*i - (num_obstacles_x/2 - 2) * spacing_y, -spacing_y*(j-1.5), height/2])

    # ... and load it
    for i, worm in enumerate(wiggler_worms):
        startPos = [random.uniform(-start_pos_range/2, start_pos_range/2) - 5, random.uniform(-start_pos_range/2, start_pos_range/2), 0.7 * i + 1]
        startOr = p.getQuaternionFromEuler([random.uniform(-start_orient_range/2, start_orient_range/2), np.pi/2, 0])
        worm.load_to_pybullet(
            baseStartPos=startPos,
            baseStartOrn=startOr,
            baseConstraint="free",  # other options are free and constrained, but those are not recommended rn
            physicsClient=physicsClient,
            flags=p.URDF_USE_SELF_COLLISION,
        )

    # ... and load it
    for i, worm in enumerate(puller_worms):
        startPos = [random.uniform(-start_pos_range/2, start_pos_range/2) + 5, random.uniform(-start_pos_range/2, start_pos_range/2) + 4, 0.7 * (i + 3 + len(wiggler_worms))]
        startOr = p.getQuaternionFromEuler([np.pi/2, np.pi/2, random.uniform(-start_orient_range/2, start_orient_range/2)])
        worm.load_to_pybullet(
            baseStartPos=startPos,
            baseStartOrn=startOr,
            baseConstraint="free",  # other options are free and constrained, but those are not recommended rn
            physicsClient=physicsClient,
            flags=p.URDF_USE_SELF_COLLISION,
        )


    # set time step
    p.setTimeStep(time_step)

    # below is an example of how lateral friction and restitution can be changed for the whole manipulator.
    contact_properties = {
        "lateralFriction": ground_friction_coefficient,
        # 'restitution': 0.0, # uncomment to change restitution
    }
    for worm in wiggler_worms:
        worm.set_contact_property(contact_properties)
    for worm in puller_worms:
        worm.set_contact_property(contact_properties)

    ######## EXECUTE SIMULATION ########
    # if desired, start video logging - this goes before the run loop
    if VIDEO_LOGGING:
        vid_filename = "vid.mp4"
        logIDvideo = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, vid_filename)

    worm_pos_time = []
    worm_or_time = []

    # this for loop is the actual simulation
    for i in range(n_steps):

        wiggler_torque = -torque*stiffness * np.array([0, 1, 0, 1])

        puller_torque_1 = torque*stiffness * np.array(
            [0, 1, np.sin((i * time_step - phase_shift[0]) / 10 * 2 * np.pi), 0])
        puller_torque_2 = torque*stiffness * np.array(
            [0, 1, np.sin((i * time_step - phase_shift[1]) / 10 * 2 * np.pi), 0])
        puller_torque_3 = torque * stiffness * np.array(
            [0, 1, np.sin((i * time_step - phase_shift[2]) / 10 * 2 * np.pi), 0])

        # initialize position logging
        worm_pos = []
        worm_or = []

        # applying the control torques
        for worm in wiggler_worms:

            wormPos, wormOr = p.getBasePositionAndOrientation(worm.bodyUniqueId)
            wormOr = p.getEulerFromQuaternion(wormOr)
            worm_pos.append(wormPos)
            worm_or.append(wormOr)

            if i * time_step > 1:
                worm.apply_actuation_torques(
                    actuator_nrs=[0, 0, 1, 1],
                    axis_nrs=[0, 1, 0, 1],
                    actuation_torques=wiggler_torque.tolist(),
                )

        for k, worm in enumerate(puller_worms):

            wormPos, wormOr = p.getBasePositionAndOrientation(worm.bodyUniqueId)
            (x, y, z), (a, b, c, d), _, _, _, _ = p.getLinkState(worm.bodyUniqueId, p.getNumJoints(worm.bodyUniqueId) - 1)
            wormOr = p.getEulerFromQuaternion((a, b, c, d))
            worm_pos.append(wormPos)
            worm_or.append(wormOr)

            if k % 3 == 0:
                if i * time_step > phase_shift[0]:
                    worm.apply_actuation_torques(
                        actuator_nrs=[0, 0, 1, 1],
                        axis_nrs=[0, 1, 0, 1],
                        actuation_torques=puller_torque_1.tolist(),
                    )
            elif k % 3 == 1:
                if i * time_step > phase_shift[1]:
                    worm.apply_actuation_torques(
                        actuator_nrs=[0, 0, 1, 1],
                        axis_nrs=[0, 1, 0, 1],
                        actuation_torques=puller_torque_2.tolist(),
                    )
            else:
                if i * time_step > phase_shift[2]:
                    worm.apply_actuation_torques(
                        actuator_nrs=[0, 0, 1, 1],
                        axis_nrs=[0, 1, 0, 1],
                        actuation_torques=puller_torque_3.tolist(),
                    )

        worm_pos_time.append(worm_pos)
        worm_or_time.append(worm_or)

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
                  "Blob Trajectory": worm_pos_time}

    with open(os.path.join("Logs_PhaseShift", date_time + '_Blob_Log.json'), 'w') as json_file:
        json.dump(parameters, json_file)


if __name__ == '__main__':

    for _ in range(1):
        # select whether you want to record a video or not
        VIDEO_LOGGING = False
        GUI = p.DIRECT

        # select blob parameters
        num_wiggler = 2
        num_puller = 3

        base_length = random.uniform(0.05, 0.1)
        actuator_length = random.uniform(0.10, 0.3)
        base = random.choice([True, False])
        if not base:
            base_length = 0.0
        mass = random.uniform(0.1, 3)
        center_of_mass = random.uniform(0.01, 0.99)  # 0: (0.25) tethered; 1: (0.75) tip; 0.5: balanced
        stiffness = random.uniform(100, 1000)
        torque = random.uniform(0.5, 2)

        start_pos_range = 1
        start_orient_range = np.pi / 180 * 10

        # Set actuation parameters
        phase_shift = [2, random.uniform(2, 7), random.uniform(2, 7)]

        # set environment parameters
        num_obstacles_x = 6
        num_obstacles_y = 15
        obs_height = random.uniform(0.5, 5)
        spacing_x = random.uniform(5, 12)
        spacing_y = spacing_x

        # select physical parameters
        gravity = -9.81
        ground_friction_coefficient = random.uniform(0.1, 2)

        ## Specify time steps
        time_step = 0.001
        n_steps = 60000

        # set parameters
        params = {"VIDEO_LOGGING": VIDEO_LOGGING,
                  "GUI": GUI,
                  "num_puller": num_puller,
                  "num_wiggler": num_wiggler,
                  "start_pos_range": start_pos_range,
                  "start_orient_range": start_orient_range,
                  "phase_shift": phase_shift,
                  "num_obstacles_x": num_obstacles_x,
                  "num_obstacles_y": num_obstacles_y,
                  "obs_height": obs_height,
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
                  "torque": torque,
                  "time_step": time_step,
                  "n_steps": n_steps}

        with open(os.path.join(r"C:\Users\tobia\PycharmProjects\SoMo_Simulation\Simulations\BlobSimulationOrientationControl\Logs_PhaseShift",
                               "20240717_120029_Blob_Log.json"), 'r') as json_file:
            parameters = json.load(json_file)

        params = parameters["Parameters"]
        params["GUI"] = p.GUI
        params["n_steps"] = 180000

        simulate(params)

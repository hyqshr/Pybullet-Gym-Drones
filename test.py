import math
import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import random
import torch

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
cube = p.loadURDF("cube.urdf", basePosition=[3,0,0.5])

# p.setGravity(0, 0, -10, physicsClientId=client)
# samurai = p.loadURDF("samurai.urdf") 
drone = p.loadURDF("./drone_envs/resources/drone.urdf", basePosition=[0,0,0.2])
position, orientation = p.getBasePositionAndOrientation(drone)
p.setRealTimeSimulation(0)
p.setTimeStep(1/30)

# p.resetDebugVisualizerCamera(cameraDistance = 5, cameraYaw=0, cameraPitch=-40,cameraTargetPosition=[5,5,3])

def setup_obstacles(obstacle_num = 100):
    obstacle_list = []
    random.seed(23)
    print("setting up ", obstacle_num, " obstacles...")
    for _ in range(obstacle_num):
        obstacle_pos = [random.randint(-12,12), random.randint(-12,12), random.randint(0,9)]
        print(obstacle_pos)
        cube = p.loadURDF("cube.urdf", basePosition=obstacle_pos)
        obstacle_list.append(cube)
    print("setting up done! ", obstacle_list)
    return obstacle_list
        
# obstacle_list = setup_obstacles()
init_location = [0,0,0]
up_vector = np.array([0, 1, 0])

for step in range(300):
    # if step == 100:
    p.applyExternalForce(
        drone,
        4,
        forceObj=[0.1,0.1, 0],
        posObj=[0, 0, 0],
        flags=p.LINK_FRAME,
    )
    
    p.stepSimulation()
        
    # Base information
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=80, 
        aspect=1,
        nearVal=0.01, 
        farVal=100)
    pos, ori = p.getBasePositionAndOrientation(drone)
    
    linear_vel, _ = p.getBaseVelocity(drone)


    # Compute the view matrix using the camera position and orientation
    view_matrix = p.computeViewMatrix((pos[0] + 0.05, pos[1],pos[2]+0.05), 
                                      (pos[0] + linear_vel[0], pos[1] + linear_vel[1] , pos[2] + linear_vel[2]), [0, 0, 1])

    # rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    # up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
    # view_matrix = p.computeViewMatrix((pos[0], pos[1],pos[2]+0.05), camera_vec, [0,1,0])

    # Display image
    frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[3]
    init_location = pos
    



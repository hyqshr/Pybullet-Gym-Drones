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
        
obstacle_list = setup_obstacles()

for step in range(300):
    pos, ori = p.getBasePositionAndOrientation(drone)
    # if step == 100:
    p.applyExternalForce(
        drone,
        4,
        forceObj=[1, 0, 0.28],
        posObj=[0, 0, 0],
        flags=p.LINK_FRAME,
    )

    print(step)
    p.stepSimulation()
    for i in obstacle_list:
        contacts = p.getContactPoints(drone,cube)
        if contacts:
            print("!!!!!!!!!!!!!")
        print(contacts)
        
    # Base information
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=80, 
        aspect=1,
        nearVal=0.01, 
        farVal=100)
    pos, ori = p.getBasePositionAndOrientation(drone)
    # Rotate camera direction
    rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    camera_vec = np.matmul(rot_mat, [1, 0, 0])
    up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
    view_matrix = p.computeViewMatrix((pos[0], pos[1],pos[2]+0.05), pos + camera_vec, up_vec)

    # Display image
    frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
    frame = np.reshape(frame, (100, 100, 4))
    plt.imshow(np.zeros((100, 100, 4))).set_data(frame)
    plt.draw()
    # plt.pause(.0001)
    # time.sleep(0.001)
    
class mySequential(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            print(module, input)
            if type(inputs) != int:
                inputs = module(*inputs)
                print(1)
            else:
                inputs = module(inputs)
                print(2)
                
        return inputs
    
nn = mySequential(
    torch.nn.Linear(9, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 3)
)

nn()


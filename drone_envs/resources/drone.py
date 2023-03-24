import pybullet as p
import os
import math
import numpy as np

class Drone:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), './drone.urdf')
        self.drone = p.loadURDF(
            fileName=f_name,
            basePosition=[0, 0, 0.1],
            physicsClientId=client
        )

    def get_ids(self):
        return self.drone, self.client

    def apply_action(self, action):
        # print(action)
        # action is 3 dimension
        thrust_x, thrust_y, thrust_z = action

        # Clip thrust and torque
        thrust_x = np.clip(thrust_x, -0.1, 0.1)
        thrust_y = np.clip(thrust_y, -0.1, 0.1)
        thrust_z = np.clip(thrust_z, -0.1, 0.1)

        p.applyExternalForce(
            self.drone,
            4,
            forceObj=[thrust_x, thrust_y, thrust_z],
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
        )


    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.drone, self.client)
        ang = p.getEulerFromQuaternion(ang)

        # Get the velocity of the car
        linear_velocity, angular_velocity = p.getBaseVelocity(self.drone, self.client)
        # print("ob: ", ang, "vel: ", p.getBaseVelocity(self.drone, self.client))
        
        # Concatenate position, orientation, velocity
        observation = (pos + linear_velocity)
        # print(observation)
        return observation










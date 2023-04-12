import gym
import numpy as np
import math
import pybullet as p
from ..resources.drone import Drone
from ..resources.plane import Plane
from ..resources.goal import Goal
import time
from ..config import drone_env_v0 as config
from ..config import observation_space_v0 as observation_space
import matplotlib.pyplot as plt


class DroneNavigationV0(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        drone action space:
            - The desired thrust along the drone's z-axis
            - The desired torque around the drone's x-axis
            - The desired torque around the drone's y-axis
            - The desired torque around the drone's z-axis
        """
        self.action_space = gym.spaces.box.Box(
            low=np.array([-0.1, -0.1, -0.1], dtype=np.float32),
            high=np.array([0.1, 0.1, 0.1], dtype=np.float32)
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.array([observation_space["drone_lower_bound_x"], 
                          observation_space["drone_lower_bound_y"], 
                          observation_space["drone_lower_bound_z"], 
                        #   observation_space["drone_angle_lower_bound_x"], 
                        #   observation_space["drone_angle_lower_bound_y"], 
                        #   observation_space["drone_angle_lower_bound_z"],
                          observation_space["drone_velocity_lower_bound_x"],
                          observation_space["drone_velocity_lower_bound_y"],
                          observation_space["drone_velocity_lower_bound_z"],
                        #   observation_space["drone_angle_velocity_lower_bound_x"],
                        #   observation_space["drone_angle_velocity_lower_bound_y"],
                        #   observation_space["drone_angle_velocity_lower_bound_z"],                          
                          observation_space["goal_lower_bound_x"],
                          observation_space["goal_lower_bound_y"],
                          observation_space["goal_lower_bound_z"],
                          ], dtype=np.float32),
            high=np.array([observation_space["drone_upper_bound_x"], 
                          observation_space["drone_upper_bound_y"], 
                          observation_space["drone_upper_bound_z"], 
                        #   observation_space["drone_angle_upper_bound_x"], 
                        #   observation_space["drone_angle_upper_bound_y"], 
                        #   observation_space["drone_angle_upper_bound_z"],
                          observation_space["drone_velocity_upper_bound_x"],
                          observation_space["drone_velocity_upper_bound_y"],
                          observation_space["drone_velocity_upper_bound_z"],
                        #   observation_space["drone_angle_velocity_upper_bound_x"],
                        #   observation_space["drone_angle_velocity_upper_bound_y"],
                        #   observation_space["drone_angle_velocity_upper_bound_z"],
                          observation_space["goal_upper_bound_x"],
                          observation_space["goal_upper_bound_y"],
                          observation_space["goal_upper_bound_z"],
                          ], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(config["display"])
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.drone = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

    def step(self, action):
        # Feed action to the drone and get observation of drone's state
        self.drone.apply_action(action)
        p.stepSimulation()
        drone_ob = self.drone.get_observation()
        # Compute reward as L2 change in distance to goal
        reward = self.calculate_reward(drone_ob)
        # print(reward)
        dist_to_goal = self.calculate_distance_from_goal(drone_ob)
        self.prev_dist_to_goal = dist_to_goal

        ob = np.array(drone_ob + self.goal, dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        # print("reset env")
        p.resetSimulation(self.client)
        self.done = False
        
        # Reload the plane and drone
        Plane(self.client)
        self.drone = Drone(self.client)
        self.reset_goal_position()
        # Get observation to return
        drone_ob = self.drone.get_observation()

        # calculate initial distance from drone to goal
        self.prev_dist_to_goal = self.calculate_distance_from_goal(drone_ob)
        return np.array(drone_ob + self.goal, dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        drone_id, client_id = self.drone.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = p.getBasePositionAndOrientation(drone_id, client_id)

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix((pos[0], pos[1],pos[2]+0.05), pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        
        # plt.savefig('graph' +  str(time.time()) +  '.png')
        # print(frame)
        # plt.pause(.0001)

    def close(self):
        p.disconnect(self.client)

    def calculate_distance_from_goal(self, observation):
        """Calculate distance based on distance between drone and goal"""
        drone_pos = [observation[0], observation[1], observation[2]]
        p.resetDebugVisualizerCamera(cameraDistance = 1, cameraYaw=0, cameraPitch=0,cameraTargetPosition=drone_pos)
        
        return math.sqrt(
            (drone_pos[0] - self.goal[0]) ** 2 +
            (drone_pos[1] - self.goal[1]) ** 2 +
            (drone_pos[2] - self.goal[2]) ** 2
        )

    def reset_goal_position(self):
        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
             self.np_random.uniform(-5, -9))
        y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
             self.np_random.uniform(-5, -9))
        z = self.np_random.uniform(5, 9)
        self.goal = (x, y, z)

        # Visual element of the goal
        Goal(self.client, self.goal)
        return self.goal

    def calculate_reward(self, observation):
        # print(observation)
        distance = self.calculate_distance_from_goal(observation)
        distance_improvement = self.prev_dist_to_goal - distance
        reward = distance_improvement

        # Done by running off boundaries
        if (observation[0] >= 12 or observation[0] <= -12 or
                observation[1] >= 12 or observation[1] <= -12 or
                observation[2] <= 0 or observation[2] >= 12):
            # print("out of bound!")
            reward -= 0
            self.done = True

        # Done by reaching goal
        if distance < 2:
            self.done = True
            print("reach the goal!  timestamp-" + str(time.time()))
            reward += 50
        
        return reward
    
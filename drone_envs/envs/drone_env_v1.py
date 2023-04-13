import gym
import numpy as np
import math
import pybullet as p
from ..resources.drone import Drone
from ..resources.plane import Plane
from ..resources.goal import Goal
import time
from collections import deque
from ..config import drone_env_v1 as config
from ..config import observation_space_v1 as observation_space
import random
import matplotlib.pyplot as plt
import pybullet_data


class DroneNavigationV1(gym.Env):
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
            low=np.array([config['thrust_x_lower_bound'], 
                          config['thrust_y_lower_bound'], 
                          config['thrust_z_lower_bound']], dtype=np.float32),
            high=np.array(
                [config['thrust_x_upper_bound'], 
                 config['thrust_y_upper_bound'], 
                 config['thrust_z_upper_bound']], dtype=np.float32)
        )
        self.observation_space = gym.spaces.box.Box(
            low=np.array([observation_space["drone_lower_bound_x"], 
                          observation_space["drone_lower_bound_y"], 
                          observation_space["drone_lower_bound_z"], 
                          observation_space["drone_velocity_lower_bound_x"],
                          observation_space["drone_velocity_lower_bound_y"],
                          observation_space["drone_velocity_lower_bound_z"],
                          observation_space["goal_lower_bound_x"],
                          observation_space["goal_lower_bound_y"],
                          observation_space["goal_lower_bound_z"],
                          ], dtype=np.float32),
            high=np.array([observation_space["drone_upper_bound_x"], 
                          observation_space["drone_upper_bound_y"], 
                          observation_space["drone_upper_bound_z"], 
                          observation_space["drone_velocity_upper_bound_x"],
                          observation_space["drone_velocity_upper_bound_y"],
                          observation_space["drone_velocity_upper_bound_z"],
                          observation_space["goal_upper_bound_x"],
                          observation_space["goal_upper_bound_y"],
                          observation_space["goal_upper_bound_z"],
                          ], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(config["display"])        
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)
        self.obstacles_pos_list = []
        self.drone = None
        self.obstacle_list = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        p.resetSimulation(self.client)
        self.setup_obstacles()
        

    def step(self, action):
        
        # Feed action to the drone and get observation of drone's state
        self.drone.apply_action(action)
        p.stepSimulation()
        drone_ob = self.drone.get_observation()
        # Compute reward as L2 change in distance to goal
        reward = self.calculate_reward(drone_ob)
        dist_to_goal = self.calculate_distance_from_goal(drone_ob)
        self.prev_dist_to_goal = dist_to_goal

        image = self.get_drone_camera_image()
        metadata = np.array(drone_ob + self.goal, dtype=np.float32)
        ob = np.concatenate((image.flatten(), metadata))
        
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        print("reset env")
        
        # Reload the plane, drone, goal position, obstacle
        self.done = False
        Plane(self.client)
        if self.drone is not None:
            p.removeBody(self.drone.drone, self.client)
        self.drone = Drone(self.client)
        self.reset_goal_position()
        # Get observation to return
        drone_ob = self.drone.get_observation()
        # call render and reset the image queue
        self.render()
        image = self.get_drone_camera_image()
        # calculate initial distance from drone to goal
        self.prev_dist_to_goal = self.calculate_distance_from_goal(drone_ob)
        metadata = np.array(drone_ob + self.goal, dtype=np.float32)
             
        return np.concatenate((image.flatten(), metadata))

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((10, 10, 1)))

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
        frame = p.getCameraImage(10, 10, view_matrix, proj_matrix)[3]
        frame = np.reshape(frame, (10, 10, 1))
        self.rendered_img.set_data(frame)
        # plt.draw()
        
        # append the frame to img deque
        self.frame = frame

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
        if self.goal is not None:
            p.removeBody(self.goal, self.client)
        # Set the goal to a random target
        x = (self.np_random.uniform(8, 20) if self.np_random.randint(2) else
             self.np_random.uniform(-20, -8))
        y = (self.np_random.uniform(8, 20) if self.np_random.randint(2) else
             self.np_random.uniform(-20, -8))
        z = self.np_random.uniform(5, 12)
        self.goal = (x, y, z)

        # Visual element of the goal
        Goal(self.client, self.goal)
        return self.goal

    def calculate_reward(self, observation):
        distance = self.calculate_distance_from_goal(observation)
        distance_improvement = self.prev_dist_to_goal - distance
        reward = distance_improvement

        # Done by running off boundaries
        if (observation[0] >= 12 or observation[0] <= -12 or
                observation[1] >= 12 or observation[1] <= -12 or
                observation[2] <= 0.05 or observation[2] >= 12):
            reward -= 0
            self.done = True

        # Done by reaching goal
        if distance < 2:
            self.done = True
            print("reach the goal!  timestamp-" + str(time.time()))
            reward += 50
        
        # check if collision happen
        if self.check_collisions(self.drone.drone):
            reward -= 30
        return reward
    
    def setup_obstacles(self, obstacle_num = 50):
        """
        set up obstacles in the environment
        """
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        obstacle_list = []
        random.seed(23)
        for _ in range(obstacle_num):
            obstacle_pos = [random.randint(-12,12), random.randint(-12,12), random.randint(2,9)]
            # print(obstacle_pos)
            cube = p.loadURDF("cube.urdf", basePosition=obstacle_pos)
            obstacle_list.append(cube), self.obstacles_pos_list.append(obstacle_pos)
        self.obstacle_list = obstacle_list
        return self.obstacle_list

    def check_collisions(self, object):
        """
        return True if there is collision between object and any obstacle
        """
        # for obstacle in self.obstacles_pos_list:
        #     contacts = p.getContactPoints(object, obstacle)
        #     return contacts if contacts else None
        # return None
        return True if p.getContactPoints(object) else False
    
    def get_drone_camera_image(self):
        """
        return the camera image of the drone
        """
        
        return self.frame
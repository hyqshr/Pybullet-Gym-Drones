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
from agent.PPOagent import PPO

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
        p.resetSimulation(self.client)
        self.setup_obstacles()
        self.goal_id = None
        self.camera_pixel = config["camera_pixel"]
        self.reach_target = False
        
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

        return ob, reward, self.done, self.reach_target

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        print("reset env")
        # Reload the plane, drone, goal position, obstacle
        self.done = False
        self.reach_target = False
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

    def render(self, mode = 'human'):
        
        # Base information
        drone_id, client_id = self.drone.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=100, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = p.getBasePositionAndOrientation(drone_id, client_id)
        linear_vel, _ = p.getBaseVelocity(drone_id)
        view_matrix = p.computeViewMatrix(
                                      (pos[0] + 0.1, pos[1],pos[2]+0.05), 
                                      (pos[0] + linear_vel[0], pos[1] + linear_vel[1], 
                                       pos[2] + linear_vel[2]), 
                                      [0, 0, 1]
                                      )
        yaw = math.atan2(linear_vel[0], linear_vel[2])
        pitch = math.atan2(linear_vel[1], linear_vel[2])
        p.resetDebugVisualizerCamera(cameraDistance = 0.5, cameraYaw=yaw, cameraPitch=pitch,cameraTargetPosition=pos)
        
        # get camera image
        frame = p.getCameraImage(self.camera_pixel, 
                                 self.camera_pixel, 
                                 view_matrix, proj_matrix)[3]
        frame = np.reshape(frame, (self.camera_pixel, self.camera_pixel, 1))
        
        # set the frame
        self.frame = frame

    def close(self):
        p.disconnect(self.client)

    def calculate_distance_from_goal(self, observation):
        """Calculate distance based on distance between drone and goal"""
        drone_pos = [observation[0], observation[1], observation[2]]
        
        return math.sqrt(
            (drone_pos[0] - self.goal[0]) ** 2 +
            (drone_pos[1] - self.goal[1]) ** 2 +
            (drone_pos[2] - self.goal[2]) ** 2
        )

    def reset_goal_position(self):
        if self.goal_id is not None:
            p.removeBody(self.goal_id, self.client)
        # Set the goal to a random target
        x = (self.np_random.uniform(8, 20) if self.np_random.randint(2) else
             self.np_random.uniform(-20, -8))
        y = (self.np_random.uniform(8, 20) if self.np_random.randint(2) else
             self.np_random.uniform(-20, -8))
        z = self.np_random.uniform(5, 12)
        self.goal = (x, y, z)

        # Visual element of the goal
        self.goal_id = Goal(self.client, self.goal).id
        return self.goal

    def calculate_reward(self, observation):
        distance = self.calculate_distance_from_goal(observation)
        distance_improvement = self.prev_dist_to_goal - distance
        reward = distance_improvement
        # print(reward)
        # Done by running off boundaries
        if (observation[0] >= 28 or observation[0] <= -28 or
                observation[1] >= 28 or observation[1] <= -28 or
                observation[2] <= 0.01 or observation[2] >= 20):
            reward -= 0.5
            self.done = True

        # Done by reaching goal
        if distance < 2:
            self.done = True
            self.reach_target = True
            print("reach the goal!  timestamp-" + str(time.time()))
            reward += 5
        
        # check if collision happen
        if self.check_collisions(self.drone.drone):
            reward -= 0.5
        return reward
    
    def setup_obstacles(self, obstacle_num = 30):
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
        return True if p.getContactPoints(object) else False
    
    def get_drone_camera_image(self):
        """
        return the camera image of the drone
        """
        return self.frame
import gym
import torch
from agent.TRPOagent import TRPOAgent
import drone_envs
import time
import pybullet as p
import argparse
from agent.TRPOagent import model
from drone_envs.config import drone_env_v1


import os
import time
from datetime import datetime
from drone_envs.config import drone_env_v1
import numpy as np
from drone_envs.config import drone_env_v1
import gym

from agent.PPOagent import PPO

# create an instance of the ArgumentParser class
parser = argparse.ArgumentParser(description='Description of your script.')

# add arguments to the parser
parser.add_argument(
    '--version', 
    type=int, 
    help='Version of the environment. 0 is the default environment without obstacle. 1 is the environment with obstacle.', 
    default=1, 
    choices=[0, 1]
    )
parser.add_argument(
    '--model', 
    type=str, 
    help='The model to run/train. Choice are TRPO & PPO', 
    default='PPO', 
    choices=['TRPO', 'PPO']
    )
parser.add_argument(
    '--mode', 
    type=str, 
    help='You can choose the train or run the model. Default to be run', 
    default='run', 
    choices=['run', 'train']
)

model_path = "agent/model"
action_dim = 3

trained_model_PPO_v1 = os.path.join(model_path, "PPO_drone_v1.pth")
trained_model_PPO_v0 = os.path.join(model_path, "PPO_drone_v0.pth")
trained_model_TRPO_v1 = os.path.join(model_path, "TRPO_drone_v1.pth")
trained_model_TRPO_v0 = os.path.join(model_path, "TRPO_drone_v0.pth")

def get_config_by_args(args):
    if args.model == 'TRPO' and args.version == 0:
        env_name = "DroneNavigation-v0"
        state_dim = 9
        agent = TRPOAgent(policy=model)
        model_file = trained_model_TRPO_v0
        
    if args.model == 'TRPO' and args.version == 1:
        env_name = "DroneNavigation-v1"
        state_dim = drone_env_v1["camera_pixel"]**2 + drone_env_v1["drone_metadata_space"]
        agent = TRPOAgent(policy=model)
        model_file = trained_model_TRPO_v1
        
    if args.model == 'PPO' and args.version == 0:
        env_name = "DroneNavigation-v0"
        state_dim = 9
        agent = PPO(state_dim, action_dim)
        model_file = trained_model_PPO_v0
        
    if args.model == 'PPO' and args.version == 1:
        env_name = "DroneNavigation-v1"
        state_dim = drone_env_v1["camera_pixel"]**2 + drone_env_v1["drone_metadata_space"]
        agent = PPO(state_dim, action_dim)
        model_file = trained_model_PPO_v1
        
    return env_name, state_dim, agent, model_file

def main():
    # parse the arguments
    args = parser.parse_args()
        
    env_name, state_dim, agent, model_file = get_config_by_args(args)
    print(f'Start {args.mode}ing mode+ with {args.model} model for drone env version: {args.version}.')
        
    agent.load_model(model_file)
    env = gym.make(env_name)
    ob = env.reset()
    round_count = 0
    success_count = 0
    
    while True:
        if args.model == 'PPO':
            action = agent.select_action(ob)
        else:
            action = agent(ob)
        ob, _r, done, info = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            round_count += 1
            if info == True:
                success_count += 1
            print(f'Round {round_count}. Success rate: {success_count * 100/round_count}%')
    

if __name__ == '__main__':
    main()


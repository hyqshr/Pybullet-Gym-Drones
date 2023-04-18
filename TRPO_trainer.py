import gym
import torch
from agent.TRPOagent import TRPOAgent
import time
import drone_envs
import pybullet as p
import sys
from datetime import datetime
from drone_envs.config import drone_env_v1
from drone_envs.config import drone_env_v0

def trainer(env_version = 0):
    # agent.load_model("agent-v0.pth")
    start_time = datetime.now().replace(microsecond=0)

    if env_version == 1:
        env = 'DroneNavigation-v1'
        nn = torch.nn.Sequential(
            torch.nn.Linear(drone_env_v1['camera_pixel'] ** 2 + drone_env_v1['drone_metadata_space'], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )
    else:
        env = 'DroneNavigation-v0'
        nn = torch.nn.Sequential(
            torch.nn.Linear(drone_env_v0['drone_observation_space'], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )
    print(f"Started training env: {env} at: {start_time}")
    
    agent = TRPOAgent(policy=nn)
    agent.train(env,
                seed=0,
                batch_size=4000,
                iterations=500,
                max_episode_length=800,
                verbose=True)
    
    # print total training time
    end_time = datetime.now().replace(microsecond=0)
    
    print("--------------------------------------------------------------------------------------------")
    print("Started training at : ", start_time)
    print("Finished training at: ", end_time)
    print("Total training time  : ", end_time - start_time)
    
    print("--------------------------------------------------------------------------------------------")
    print("saving the model...")
    agent.save_model(f"agent/model/TRPO_drone_v{env_version}.pth")

if __name__ == '__main__':
    trainer()

import gym
import torch
from agent import TRPOAgent
import time
import drone_envs
import pybullet as p
import sys

def main():
    nn = torch.nn.Sequential(
        torch.nn.Linear(9, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 3)
    )
    agent = TRPOAgent(policy=nn)
    agent.load_model("agent-v0.pth")

    env = 'DroneNavigation-v0'
    
    
    agent.train(env,
                seed=0,
                batch_size=5000,
                iterations=100,
                max_episode_length=200,
                verbose=True)
    print("saving the model")
    agent.save_model("agent-v0.pth")

    env = gym.make(env)
    ob = env.reset()
    
    # while True:
    #     action = agent(ob)
    #     ob, _r, done, _ = env.step(action)
    #     env.render()
    #     if done:
    #         ob = env.reset()
    #         time.sleep(1)


if __name__ == '__main__':
    main()

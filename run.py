import gym
import torch
from agent import TRPOAgent
import drone_envs
import time
import pybullet as p

def main():
    nn = torch.nn.Sequential(
        torch.nn.Linear(9, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 3)
    )
    
    agent = TRPOAgent(policy=nn)

    agent.load_model("agent.pth")
    env = gym.make('DroneNavigation-v0')
    ob = env.reset()
    
    while True:
        action = agent(ob)
        ob, _r, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(0.001)


if __name__ == '__main__':
    main()



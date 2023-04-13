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

    agent.load_model("agent-v0.pth")
    env = gym.make('DroneNavigation-v0')
    ob = env.reset()
    total_test = 100
    count = 0
    while count < 100:
        action = agent(ob)
        ob, _r, done, _ = env.step(action)
        # env.render()
        if done:
            ob = env.reset()
            # time.sleep(0.001)
            count += 1
    print(count)

if __name__ == '__main__':
    main()




import gym
import torch
from agent import TRPOAgent
import time
import drone_envs
import pybullet as p
import sys

class Model(torch.nn.Module):
    def __init__(self):
        """
        This model is design for drone_env_v1 option.
        It could process both image and metadata from the environment. 
        """
        super().__init__()

    def forward(self, image, metadata):
        """input
        image shape
        """
        return 

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

def main():
    nn = torch.nn.Sequential(
        torch.nn.Linear(9, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 3)
    )
    agent = TRPOAgent(policy=nn)
    agent.load_model("agent.pth")

    env = 'DroneNavigation-v0'
    # env = "DroneNavigation-v1"
    
    
    agent.train(env,
                seed=0,
                batch_size=5000,
                iterations=0,
                max_episode_length=200,
                verbose=True)
    print("saving the model")
    agent.save_model("agent.pth")

    env = gym.make(env)
    ob = env.reset()
    
    while True:
        action = agent(ob)
        ob, _r, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1)


if __name__ == '__main__':
    main()

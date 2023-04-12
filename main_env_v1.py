import gym
import torch
from agent_env_v1 import TRPOAgent
import time
import drone_envs
import pybullet as p
import sys
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        """
        This model is design for drone_env_v1 option.
        It could process both image and metadata from the environment. 
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 5, stride = 3)
        self.FC1 = nn.Linear(32 * 32 * 8, 1)
        self.FC2 = nn.Linear(5, 1)
        self.FC3 = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )
        
    def forward(self, image, metadata):
        """input
        image shape
        """
        image = image.permute(0,3,1,2)
        x = self.conv1(image)
        
        x = F.relu(x)
        x = torch.flatten(x, 1)

        x = self.FC1(x)
        
        x = torch.flatten(x)
        x = self.FC2(x)
        
        metadata = torch.squeeze(metadata,0)
        
        x = torch.cat((x,metadata) , 0)
        x = self.FC3(x)
        return x
    
    def string(self):
        return [module for module in self.modules()]
    
def main():
    agent = TRPOAgent(policy=Model())
    env = "DroneNavigation-v1"
    agent.train(env,
                seed=0,
                batch_size=5000,
                iterations=1,
                max_episode_length=200,
                verbose=True)
    print("saving the model")
    agent.save_model("agent_v1.pth")

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

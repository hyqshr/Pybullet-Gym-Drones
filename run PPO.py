import gym
import torch
from agent import TRPOAgent
import drone_envs
import time
import pybullet as p
from PPOagent import PPO

state_dim = 109
action_dim = 3
env_name = "DroneNavigation-v1"

has_continuous_action_space = True  # continuous action space; else discrete

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)          # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 100               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0 

def main():
    agent = PPO(state_dim, 
                    action_dim, 
                    lr_actor, 
                    lr_critic, 
                    gamma, 
                    K_epochs, 
                    eps_clip, 
                    has_continuous_action_space, 
                    action_std)
    

    agent.load("PPO_drone_v1.pth")
    env = gym.make('DroneNavigation-v1')
    ob = env.reset()
    total_test = 100
    count = 0
    while count < 100:
        action = agent.select_action(ob)
        ob, _r, done, _ = env.step(action)
        # env.render()
        if done:
            ob = env.reset()
            # time.sleep(0.001)
            count += 1
    print(count)

if __name__ == '__main__':
    main()




# Pybullet-Gym-Drone - 强化学习无人机

Drone auto-navigation stimulation developed in [Pybullet](https://pybullet.org/) + [OpenAI Gym](https://github.com/openai/gym) Environment. Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO) is implemented.


![drone_moving](./drone_moving.gif)

# Result
| Sucess Rate | DroneNavigationV0 |DroneNavigationV1   |
|-----------------|-----------------|-----------------|
| TRPO | 50.0% | 39.2% |
| PPO | 91.2% |11.2% |

![result](./image/result.png)


# Run

- You can run & see the real time successful rate by:

```
python run.py --model PPO --version 0  
```

- To train the model, you can run:
```
python PPO_trainer.py
```

or 

```
python TRPO_trainer.py
```

- You can modify the GUI in `drone_envs/config` `"display": p.GUI,`



Thanks [Medium series on creating OpenAI Gym Environments with PyBullet](https://medium.com/@gerardmaggiolino/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24). 

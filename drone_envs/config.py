import pybullet as p

drone_env_v0 = {
    "display": p.DIRECT,
    "thrust_x_lower_bound": -0.1,
    "thrust_y_lower_bound": -0.1,
    "thrust_z_lower_bound": -0.1,
    "thrust_x_upper_bound": 0.1,
    "thrust_y_upper_bound": 0.1,
    "thrust_z_upper_bound": 0.1,
}

observation_space_v0 = {
    "drone_lower_bound_x": -12, 
    "drone_upper_bound_x": 12,     
    "drone_lower_bound_y": -12, 
    "drone_upper_bound_y": 12, 
    "drone_lower_bound_z": 0, 
    "drone_upper_bound_z": 12, 
    "drone_velocity_lower_bound_x": -3, 
    "drone_velocity_upper_bound_x": 3,     
    "drone_velocity_lower_bound_y": -3, 
    "drone_velocity_upper_bound_y": 3, 
    "drone_velocity_lower_bound_z": -3, 
    "drone_velocity_upper_bound_z": 3,     
    "goal_lower_bound_x": -9,
    "goal_upper_bound_x": 9,
    "goal_lower_bound_y": -9,
    "goal_upper_bound_y": 9,
    "goal_lower_bound_z": 0,
    "goal_upper_bound_z": 9,
}

drone_env_v1 = {
    "display": p.DIRECT,
    "thrust_x_lower_bound": -0.2,
    "thrust_y_lower_bound": -0.2,
    "thrust_z_lower_bound": -0.2,
    "thrust_x_upper_bound": 0.2,
    "thrust_y_upper_bound": 0.2,
    "thrust_z_upper_bound": 0.2,
}
observation_space_v1 = {
        "drone_lower_bound_x": -25, 
        "drone_upper_bound_x": 25,     
        "drone_lower_bound_y": -25, 
        "drone_upper_bound_y": 25, 
        "drone_lower_bound_z": 0, 
        "drone_upper_bound_z": 15, 
        "drone_velocity_lower_bound_x": -3, 
        "drone_velocity_upper_bound_x": 3,     
        "drone_velocity_lower_bound_y": -3, 
        "drone_velocity_upper_bound_y": 3, 
        "drone_velocity_lower_bound_z": -3, 
        "drone_velocity_upper_bound_z": 3,     
        "goal_lower_bound_x": -21,
        "goal_upper_bound_x": 21,
        "goal_lower_bound_y": -21,
        "goal_upper_bound_y": 21,
        "goal_lower_bound_z": 0,
        "goal_upper_bound_z": 12,
    }
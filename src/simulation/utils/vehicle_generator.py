from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

# Import the custom python control backend
from controller.controller import NonlinearController
from trajectories.circle import Circle

# Auxiliary scipy and numpy modules
import time
import numpy as np
from scipy.spatial.transform import Rotation

def spawn_vehicles(num_vehicles_x, num_vehicles_y, spacing_between_vehicles):

    # Create the vehicles and the corresponding trajectories
    vehicle_id = 0
    vehicles = []

    # Random number generator without a seed
    rgn = np.random.default_rng(seed=int(time.time()))

    # Where the grid of robots will be spawned
    random_xy_offset = rgn.uniform(low=-25.0, high=10.0, size=(2,))

    # Compute the square center for each drone
    square_center = spacing_between_vehicles / 2.0

    for i in range(num_vehicles_x):
        for j in range(num_vehicles_y):

            # Create the desired trajectory for the vehicle i
            init_angle = float(rgn.uniform(low=0, high=2*np.pi))
            trajectory = Circle(
                radius=3.0,
                center_x=square_center + (i * spacing_between_vehicles) +  random_xy_offset[0],
                center_y=square_center + (j * spacing_between_vehicles) +  random_xy_offset[1],
                z_axis=1.0,
                init_angle=init_angle
            )

            # Create the vehicle i
            # Try to spawn the selected robot in the world to the specified namespace
            config_multirotor1 = MultirotorConfig()
            config_multirotor1.backends = [NonlinearController(
                Kp=[8.5, 8.5, 8.5],
                Kd=[8.5, 8.5, 8.5],
                Ki=[0.1, 0.1, 0.1],
                Kr=[2.0, 2.0, 2.0],
                Kw=[0.8, 0.8, 0.8], 
                trajectory=trajectory
            )]

            # Define the spawn position inside the square of each vehicle
            x = rgn.uniform(low=2.0, high=spacing_between_vehicles) + (i * spacing_between_vehicles) + random_xy_offset[0]
            y = rgn.uniform(low=2.0, high=spacing_between_vehicles) + (j * spacing_between_vehicles) + random_xy_offset[1]

            # Define the orientation of the vehicle
            yaw = rgn.uniform(low=0.0, high=360.0)

            vehicles += [Multirotor(
                "/World/quadrotor" + str(vehicle_id),
                ROBOTS['Iris'],
                vehicle_id,
                [x, y, 0.07],
                Rotation.from_euler("XYZ", [0.0, 0.0, yaw], degrees=True).as_quat(),
                config=config_multirotor1,
            )]

            vehicle_id += 1
            
    return vehicles
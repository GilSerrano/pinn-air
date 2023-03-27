from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

# Import the custom python control backend
from controller.controller import NonlinearController
from trajectories.circle import Circle

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation

def spawn_vehicles(num_vehicles_x, num_vehicles_y, spacing_between_vehicles):

    # Create the vehicles and the corresponding trajectories
    vehicle_id = 0
    vehicles = []

    # Random number generator without a seed
    rgn = np.random.default_rng(seed=None)

    # Where the grid of robots will be spawned
    random_xy_offset = rgn.uniform(low=-100.0, high=0.0, size=(2,))

    for i in range(num_vehicles_x):
        for j in range(num_vehicles_y):

            # Create the desired trajectory for the vehicle i
            trajectory = Circle(3.0, (i * spacing_between_vehicles) +  random_xy_offset[0], (j * spacing_between_vehicles) +  random_xy_offset[1], z_axis=1.0)

            # Create the vehicle i
            # Try to spawn the selected robot in the world to the specified namespace
            config_multirotor1 = MultirotorConfig()
            config_multirotor1.backends = [NonlinearController(
                Ki=[0.5, 0.5, 0.5],
                Kr=[2.0, 2.0, 2.0],
                trajectory=trajectory
            )]

            # Define the spawn position inside the square of each vehicle
            x = rgn.uniform(low=(i * spacing_between_vehicles), high=(i * spacing_between_vehicles + spacing_between_vehicles)) + random_xy_offset[0]
            y = rgn.uniform(low=(j * spacing_between_vehicles), high=(j * spacing_between_vehicles + spacing_between_vehicles)) + random_xy_offset[1]

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
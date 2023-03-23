from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

# Import the custom python control backend
from controller.controller import NonlinearController
from trajectories.circle import Circle

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

def spawn_vehicles(num_vehicles_x, num_vehicles_y, spacing_between_vehicles):

    # Create the vehicles and the corresponding trajectories
    vehicle_id = 0
    vehicles = []

    for i in range(num_vehicles_x):
        for j in range(num_vehicles_y):

            # Create the desired trajectory for the vehicle i
            trajectory = Circle(3.0, i * spacing_between_vehicles, j * spacing_between_vehicles, z_axis=1.0)

            # Create the vehicle i
            # Try to spawn the selected robot in the world to the specified namespace
            config_multirotor1 = MultirotorConfig()
            config_multirotor1.backends = [NonlinearController(
                Ki=[0.5, 0.5, 0.5],
                Kr=[2.0, 2.0, 2.0],
                trajectory=trajectory
            )]

            vehicles += [Multirotor(
                "/World/quadrotor" + str(vehicle_id),
                ROBOTS['Iris'],
                vehicle_id,
                [i * spacing_between_vehicles, j * spacing_between_vehicles, 0.07],
                Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
                config=config_multirotor1,
            )]

            vehicle_id += 1
            
    return vehicles
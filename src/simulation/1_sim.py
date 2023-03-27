#!/usr/bin/env python
"""
| File: 1_sim.py
| Author: Marcelo Jacinto, Joao Pinto, Gil Serrano, Jose Gomes
| License: BSD-3-Clause. Copyright (c) 2023, Pegasus Research. All rights reserved.
| Description: This files serves as generator for simulation flight data.
"""

# Imports to start Isaac Sim from this script
import carb
from omni.isaac.kit import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import numpy as np

# Import the Pegasus API for simulating drones
from pegasus.simulator.pegasus_app import PegasusApp

# Import the custom python control backend
from utils.vehicle_generator import spawn_vehicles

class Simulator(PegasusApp):
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Initialize the PegasusApp
        super().__init__(simulation_app, world="Default Environment")

        self.num_vehicles_x = 5
        self.num_vehicles_y = 5
        self.spacing_between_vehicles = 10.0 # meters

        # Random number generator without a seed
        rgn = np.random.default_rng(seed=None)

        # Where the grid of robots will be spawned
        self.elapsed_time = 0.0
        self.sim_time = float(rgn.uniform(low=12.0, high=20.0))
        carb.log_warn("Simulation time: " + str(self.sim_time) + " seconds")

        # Create the vehicles and the corresponding trajectories
        spawn_vehicles(self.num_vehicles_x, self.num_vehicles_y, self.spacing_between_vehicles)

        # Add a callback to stop the simulaiton after a given amount of time
        self.world.add_physics_callback('check_sim_stop', self.check_sim_stop)

        # Start the simulation
        self.start()

    def check_sim_stop(self, dt):
        """
        Method that checks if the simulation should stop
        """

        self.elapsed_time += dt

        # Stop the simulation after self.time
        if self.elapsed_time > self.sim_time:
            self.stop_sim = True

def main():

    # Instantiate the template app
    pg_app = Simulator()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()

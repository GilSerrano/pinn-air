import numpy as np

class Circle:

    def __init__(self, radius=1.0, center_x=0.0, center_y=0.0, z_axis=1.0, init_angle=0.0):
        """Constructor of the exponential trajectory

        Args:
            radius (float, optional): The radius of the circle. Defaults to 1.0.
            center_x (float, optional): The x coordinate of the center of the circle. Defaults to 0.0.
            center_y (float, optional): The y coordinate of the center of the circle. Defaults to 0.0.
            z_axis (float, optional): The z coordinate of the center of the circle. Defaults to 1.0.
        """

        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y
        self.z_axis = z_axis

        self.init_angle = init_angle

    def pd(self, t):
        """The desired position of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
        Returns:
            np.ndarray: A 3x1 array with the x, y ,z desired [m]
        """


        t = t + self.init_angle

        pd_t = np.zeros(3)

        pd_t[0] = self.radius * np.cos(t) + self.center_x
        pd_t[1] = self.radius * np.sin(t) + self.center_y
        pd_t[2] = self.z_axis

        return pd_t

    def d_pd(self, t):
        """The desired velocity of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A 3x1 array with the d_x, d_y ,d_z desired [m/s]
        """

        t = t + self.init_angle

        d_pd_t = np.zeros(3)

        d_pd_t[0] = -self.radius * np.sin(t)
        d_pd_t[1] = self.radius * np.cos(t)
        d_pd_t[2] = 0.0

        return d_pd_t

    def dd_pd(self, t):
        """The desired acceleration of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A 3x1 array with the dd_x, dd_y ,dd_z desired [m/s^2]
        """

        t = t + self.init_angle

        dd_pd_t = np.zeros(3)

        dd_pd_t[0] = -self.radius * np.cos(t)
        dd_pd_t[1] = -self.radius * np.sin(t)
        dd_pd_t[2] = 0.0

        return dd_pd_t

    def ddd_pd(self, t):
        """The desired jerk of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A 3x1 array with the ddd_x, ddd_y ,ddd_z desired [m/s^3]
        """

        t = t + self.init_angle
        
        ddd_pd_t = np.zeros(3)

        ddd_pd_t[0] = self.radius * np.sin(t)
        ddd_pd_t[1] = -self.radius * np.cos(t)
        ddd_pd_t[2] = 0.0

        return ddd_pd_t

    def yaw_d(self, t):
        """The desired yaw of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A float with the desired yaw in rad
        """

        t = t + self.init_angle

        return 0.0

    def d_yaw_d(self, t):
        """The desired yaw_rate of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A float with the desired yaw_rate in rad/s
        """

        t = t + self.init_angle
        
        return 0.0
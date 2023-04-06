import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, title, xlabel, ylabel):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot3d(x, y, z, title, xlabel, ylabel, zlabel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.scatter(x[0], y[0], z[0], c='red', marker='o', s=200)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()


def main():

    # Define the filename to plot
    filename = "data/1679928545_data10.npz"

    # Load the data
    statistics = np.load(filename)

    # Plot the X and Y positions over time
    plot(statistics["p"][:,0], statistics["p"][:,1], "Position Plot", "X [m]", "Y [m]")
    plot3d(statistics["p"][:,0], statistics["p"][:,1], statistics["p"][:,2], "Position Plot", "X [m]", "Y [m]", "Z [m]")

if __name__ == "__main__":
    main()


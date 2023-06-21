import math
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_waypoints(waypoints, connections, image_path, save_path):
    """
    Plots the connections between the given waypoints and overlays it onto the image at image_path.
    Saves the resulting plot to save_path.
    
    Parameters:
    - waypoints: a numpy array containing (x,y) pixel coordinates of a certain set of way-points
    - connections: a second array containing a single row of numbers which indicates the connection between the previously given way-points
    - image_path: the path to the image to overlay the plot on
    - save_path: the path to save the resulting plot to
    """
    # Load the image
    image = plt.imread(image_path)

    # Create a new figure and axis for the plot
    fig, ax = plt.subplots()

    # Plot the image
    ax.imshow(image)

    # Plot the connections between the waypoints
    for i in range(len(connections)-1):
        start_point = waypoints[connections[i]]
        end_point = waypoints[connections[i+1]]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')

    # Save the resulting plot
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
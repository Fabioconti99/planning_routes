import numpy as np
import os
import math
import matplotlib.pyplot as plt
import plot_panels

# Calculating of the Eucledia distance matrix modified
def euclidean_distance_mod(points, constraints):
    num_points = len(points)
    distances = np.zeros((num_points, num_points))
    tot_dist = 0 
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                distances[i,j] = 0
            else:
                point_i = points[i]
                point_j = points[j]
                dist = np.linalg.norm(point_i - point_j)
                distances[i,j] = dist
                
                if tot_dist < dist: 

                    tot_dist = dist

    for i in range(num_points):

        for j in range(num_points):
            if i == j:
                distances[i,j] = 0
            else:
                distances[i,j] = distances[i,j] + (tot_dist)

                for constraint in constraints:
                    if i in constraint and j in constraint:
                        distances[i,j] = distances[i,j] -(tot_dist)
                        distances[j,i] = distances[j,i] -(tot_dist)
                        break
    return distances

# Actual eucledia distance matrix without any differentiation
def euclidean_distance(points):
    num_points = len(points)
    distances = np.zeros((num_points, num_points))
    tot_dist = 0 
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                distances[i,j] = 0
            else:
                point_i = points[i]
                point_j = points[j]
                dist = np.linalg.norm(point_i - point_j)
                distances[i,j] = dist

    return distances

# Connection beteen 2 array of permutaiton
def connect_permutation(distances, permutation):

    total_distance = 0

    for i in range(len(permutation)-1):
        node1 = permutation[i]
        node2 = permutation[i+1]
        distance = distances[node1][node2]
        total_distance += distance

    return total_distance

# Finding the distance in meters between two points of the image
def get_distance(wp1, wp2):

    # approximate radius of Earth in meters
    R = 6373.0 * 1000

    # convert coordinates to radians
    lat1_rad = math.radians(wp1[0])
    lon1_rad = math.radians(wp1[1])
    lat2_rad = math.radians(wp2[0])
    lon2_rad = math.radians(wp2[1])

    # calculate differences between coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # apply Haversine formula to calculate distance
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

# Using the two distances to find the pixel size in meters anc calculate distances with it
def pixel_size(tl, br, im):

    p1 = np.array([tl[0],br[1]])

    width = get_distance(p1,tl)
    height = get_distance(br,p1)

    width = width/im.shape[0]
    height = height/im.shape[1]

    av = (height + width)/2
    return av






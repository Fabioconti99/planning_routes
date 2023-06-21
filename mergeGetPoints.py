# version inspection
import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.projects import point_rend

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

from PIL import Image


import math

# Get the extreams of the panles and the center point
def get_points_center(center_list,st):

    points = np.array([st[0][0],st[0][1]])
    constraints = np.zeros((1,2))
    #points = np.array([st[0],st[1]])
    center_points = np.zeros((1,2))

    n = 1
    for centers in center_list:

        center_points = np.vstack((center_points, np.array(centers[0])))

	    # Convert angle from degrees to radians
        alpha = math.radians(centers[2])

        radius = (max(centers[1][0],centers[1][1]))/2

        if centers[1][0] < centers[1][1]:
            alpha = alpha - math.pi/2

	    # Calculate the x and y coordinates of the first point
        x2 = centers[0][0] + radius * math.cos(math.pi + alpha)
        y2 = centers[0][1] + radius * math.sin(math.pi + alpha)
	    
	    # Calculate the x and y coordinates of the second point
        x1 = centers[0][0] + radius * math.cos(alpha)
        y1 = centers[0][1] + radius * math.sin(alpha)


        points = np.vstack((points,np.array([x2, y2])))
        points = np.vstack((points,np.array([x1, y1])))


        constraints = np.vstack((constraints,np.array([n, n+1])))

        n = n+2

    constraints = np.delete(constraints, 0, 0)
    center_points = np.delete(center_points, 0, 0)

	# Return the two points as a numpy array
    return points, constraints, center_points

# Obtain the lateral points of each panel of a certain cluster
def cluster_expansion(cluster_nodes, center_list,center_arr,st):

    points = np.array([st[0][0],st[0][1]])
    #points = np.array([st[0],st[1]])
    constraints = np.zeros(2)

    n = 1
    i = 0
    for centers in center_list:

        for k in range(cluster_nodes.shape[0]):

            if centers[0][0] == cluster_nodes[k,0] and centers[0][1] == cluster_nodes[k,1]:

                # Convert angle from degrees to radians
                alpha = math.radians(centers[2])

                radius = (max(centers[1][0],centers[1][1]))/2

                if centers[1][0] < centers[1][1]:
                    alpha = alpha - math.pi/2

                # Calculate the x and y coordinates of the first point
                x2 = centers[0][0] + radius * math.cos(math.pi + alpha)
                y2 = centers[0][1] + radius * math.sin(math.pi + alpha)
                
                # Calculate the x and y coordinates of the second point
                x1 = centers[0][0] + radius * math.cos(alpha)
                y1 = centers[0][1] + radius * math.sin(alpha)


                points = np.vstack((points,np.array([x2, y2])))
                points = np.vstack((points,np.array([x1, y1])))


                constraints = np.vstack((constraints,np.array([n, n+1])))

                n = n+2
                break

        i = i + 1

    constraints = np.delete(constraints, 0, 0)

    return points, constraints


# Get the length of the actual tour
def actual_perm(coords, subset, indexes):

    # Reorganize the subset of rows according to the given order
    reorganized_subset = subset[indexes]
    
    # Find the indices of the rows in `coords` that match the rows in the reorganized subset
    matching_indices = []

    for row in reorganized_subset:
        matching_indices.append(np.where((coords == row).all(axis=1))[0][0])
    
    return np.array(matching_indices)


# Get the top point of the group
def extract_points(points):
    # Calculate the distances between the first point and the other three points
    distances = np.linalg.norm(points[1:] - points[0], axis=1)

    # Find the index of the closest point to the first point
    closest_index = np.argmin(distances) + 1  # add 1 to account for zero-based indexing

    # Create the first pair of points
    pair1 = np.array([points[0], points[closest_index]])

    # Create the second pair of points
    pair2 = np.delete(points, [0, closest_index], axis=0)

    # Return the two pairs of points as 2x2 arrays
    return pair1, pair2


# divid the segment into sub-points 
def divide_segment(segment, k):

    # Calculate the length of the segment
    length = np.linalg.norm(segment[1] - segment[0])

    # Calculate the distances at which to divide the segment

    dist = np.zeros(k)

    for i in range(k):

    	dist[i] = (i+1) * length / (k+1)

    # Calculate the direction of the segment
    direction = (segment[1] - segment[0]) / length

    # Calculate the points that divide the segment
    points = np.zeros((k,2))

    for i in range(k):

	    points[i]= segment[0] + dist[i] * direction


    # Return the new segment as a numpy array

    return points


# Divide the segment but with another criterion
def divide_segment_double(segment, k):

	# k = number of points 

    # Calculate the length of the segment
    length = np.linalg.norm(segment[1] - segment[0])

    # Calculate the distances at which to divide the segment

    dist = np.zeros(k)

    n = 1

    for i in range(k):

    	dist[i] = n * length / (k*2)

    	n = n + 2


    # Calculate the direction of the segment
    direction = (segment[1] - segment[0]) / length

    # Calculate the points that divide the segment
    points = np.zeros((k,2))

    for i in range(k):

	    points[i]= segment[0] + dist[i] * direction


    # Return the new segment as a numpy array

    return points

# Find the closest point to a line described 
def closest_points_on_line(points, line):
    # normalize line direction
    u = line[1] - line[0]
    u = u / np.linalg.norm(u)

    # project points onto line
    v = points - line[0]
    proj = np.dot(v, u)
    closest = line[0] + proj[:, None] * u

    return closest


# Generate the array of constraints
def generate_arrays(arr1, k):
    n = arr1.shape[0]

    # First array
    first_arr = np.zeros((2*n-2, 2), dtype=int)
    for i in range(n-1):
        first_arr[i] = [k+1+i, k+i+2]
    for i in range(n, 2*n-1):
        first_arr[i-1] = [k+i+1, k+i+2]

    # Second array
    second_arr = np.zeros((n, 2), dtype=int)
    for i in range(n):
        second_arr[i] = [k+i+1, k+i+n+1]

    return first_arr, second_arr


# Mix up two arrays of points 
def interleave_arrays(a, b):

    assert a.shape == b.shape, "Arrays must have the same shape"

    n,_ = a.shape

    result = np.zeros((n*2, 2))

    k = 1
    j = 0

    result[0] = a[0]
    result[-1] = b[-1]

    r = int((n-1)/2)

    for i in range (r):

    	
    	result[4*i+1] = b[j]
    	result[4*i+2] = b[j+1]


    	result[4*i+3] = a[k]
    	result[4*i+4] = a[k+1]

    	j = j + 2 
    	k = k + 2

    return result


# Same thing but with even passages
def interleave_arrays_even(a, b):

    assert a.shape == b.shape, "Arrays must have the same shape"

    n,_ = a.shape

    result = np.zeros((n*2, 2))

    k = 1
    j = 0

    result[0] = a[0]
    result[-1] = a[-1]


    for i in range (n-1):

    	
    	result[1+(4*i)] = b[j]
    	result[2+(4*i)] = b[j+1]


    	result[3+(4*i)] = a[k]

    	try:

    		result[4+(4*i)] = a[k+1]
    	except:

    		break

    	j = j + 2 
    	k = k + 2

    return result




# Get the points order in the case of even passages
def get_points_l(center_list,st):

    points = np.array([st[0][0],st[0][1]])
    #points = np.array([st[0],st[1]])

    for centers in center_list:

	    # Convert angle from degrees to radians
        alpha = math.radians(centers[2])

        radius = (max(centers[1][0],centers[1][1]))/2

        if centers[1][0] < centers[1][1]:
            alpha = alpha - math.pi/2

        # Calculate the x and y coordinates of the first point
        x2 = centers[0][0] + radius * math.cos(math.pi + alpha)
        y2 = centers[0][1] + radius * math.sin(math.pi + alpha)
	    
	    # Calculate the x and y coordinates of the second point
        x1 = centers[0][0] + radius * math.cos(alpha)
        y1 = centers[0][1] + radius * math.sin(alpha)


        if abs(y2-y1)<abs(x2-x1):

	    	# horizontal case
            if x2<x1:
                points = np.vstack((points,np.array([x2, y2])))
            else: 
                points = np.vstack((points,np.array([x1, y1])))

        else:

	    	# vertical case
            if y2<y1:
                points = np.vstack((points,np.array([x1, y1])))
            else: 
                points = np.vstack((points,np.array([x2, y2])))

	# Return the two points as a numpy array
    return points

# Same thihing as expansion_plan but with even passages
def expansion_plan_even(perm, points, boxes, n_pass, dub,st):

    final_perm = np.array((st[0][0],st[0][1]))
    #final_perm = np.array((st[0],st[1]))

    for n in range(len(perm)-1):

        if perm[n+1] == 0 and (n!=0):
            
            final_perm = np.vstack((final_perm,[st[0][0],st[0][1]]))

        else: 
            pair1, pair2 = extract_points(boxes[perm[n+1]-1])

            diff = np.subtract(pair1[1,:],pair2[1,:])

            if abs(diff[0])>abs(diff[1]):

                # Horizontal layout

                if abs(pair2[0,0]) < abs(pair1[0,0]):

                    if dub:
                    	wp_1 = divide_segment_double(pair2, n_pass)
                    else: 
                    	wp_1 = divide_segment(pair2, n_pass)

                    wp_2 = closest_points_on_line(wp_1, pair1)

                else: 

                    if dub:
                    	wp_1 = divide_segment_double(pair1, n_pass)
                    else:
                    	wp_1 = divide_segment(pair1, n_pass)

                    wp_2 = closest_points_on_line(wp_1, pair2)

                try:
                    if abs(wp_1[0,1]-final_perm[-1, 1]) > abs(wp_1[-1,1]-final_perm[-1, 1]):
                        wp_1 = wp_1[::-1]
                        wp_2 = wp_2[::-1]
                except:
                    pass
            else: 

                # Vertical layout

                if abs(pair2[0,1]) < abs(pair1[0,1]):

                    if dub:
                    	wp_1 = divide_segment_double(pair1, n_pass)
                    else: 
                    	wp_1 = divide_segment(pair1, n_pass)

                    wp_2 = closest_points_on_line(wp_1, pair2)

                else: 
                    if dub: 
                    	wp_1 = divide_segment_double(pair2, n_pass)
                    else: 
                    	wp_1 = divide_segment(pair2, n_pass)

                    wp_2 = closest_points_on_line(wp_1, pair1)

                try:
                    if abs(wp_1[0,0]-final_perm[-1, 0]) > abs(wp_1[-1,0]-final_perm[-1, 0]):
                        wp_1 = wp_1[::-1]
                        wp_2 = wp_2[::-1]
                except:
                    pass

            temp = interleave_arrays_even(wp_1, wp_2)
            final_perm = np.vstack((final_perm,temp))

    perm = np.arange(len(final_perm))

    return final_perm, perm



# arrange per mutation given some points and constraints
def expansion_plan(perm, points, constr, boxes, n_pass, dub,st):

    final_perm = np.array((st[0][0],st[0][1]))

    for n in range(len(perm)-1):

        if perm[n] == 0 and n!=0:

            final_perm = np.vstack((final_perm,[st[0][0],st[0][1]]))

        if ([perm[n],perm[n+1]] in constr.tolist()) or ([perm[n+1],perm[n]] in constr.tolist()):

            # Get the box corresponding to the value of the n-th member fo the permutaiton

            pair1, pair2 = extract_points(boxes[math.ceil(perm[n]/2)-1])

            try: 
            	diff = np.subtract(points[perm[n],:],points[perm[n+1],:])
            except:
            	break

            if abs(diff[0])>abs(diff[1]):

                # Horizontal layout

                if abs(points[perm[n],0]-pair2[0,0]) < abs(points[perm[n],0]-pair1[0,0]):

                    if dub:
                    	wp_1 = divide_segment_double(pair2, n_pass)
                    else: 
                    	wp_1 = divide_segment(pair2, n_pass)

                    wp_2 = closest_points_on_line(wp_1, pair1)
                else: 

                    if dub:
                    	wp_1 = divide_segment_double(pair1, n_pass)
                    else: 
                    	wp_1 = divide_segment(pair1, n_pass)

                    wp_2 = closest_points_on_line(wp_1, pair2)

                try: 
                    if abs(wp_1[0,1]-final_perm[-1, 1]) > abs(wp_1[-1,1]-final_perm[-1, 1]):
                        wp_1 = wp_1[::-1]
                        wp_2 = wp_2[::-1]
                except: 
                    pass
            else: 

                # Vertical layout

                if abs(points[perm[n],1]-pair2[0,1]) < abs(points[perm[n+1],1]-pair1[0,1]):

                    if dub:
                    	wp_1 = divide_segment_double(pair2, n_pass)
                    else: 
                    	wp_1 = divide_segment(pair2, n_pass)

                    wp_2 = closest_points_on_line(wp_1, pair1)

                else: 

                    if dub:
                    	wp_1 = divide_segment_double(pair1, n_pass)
                    else: 
                    	wp_1 = divide_segment(pair1, n_pass)

                    wp_2 = closest_points_on_line(wp_1, pair2)
                try: 
                    if abs(wp_1[0,0]-final_perm[-1, 0]) > abs(wp_1[-1,0]-final_perm[-1, 0]):
                        wp_1 = wp_1[::-1]
                        wp_2 = wp_2[::-1]
                except:
                    pass

            temp = interleave_arrays(wp_1, wp_2)
            final_perm = np.vstack((final_perm,temp))

    perm = np.arange(len(final_perm))

    return final_perm, perm

# Fix the incline of the boxes. If the angle are similr 
# it will normalize it to the same vale
def angle_adjustment(rects):

    angs = []

    for rect in rects:

        if rect[1][0]<rect[1][1]:

            if rect[2]>0:
                angs.append(float(rect[2])-90)

            else:
                angs.append(float(rect[2])+90)

        else: 

            angs.append(float(rect[2]))

    angs = np.array(angs)

    std = np.std(angs)
    mean = np.mean(angs)

    rects_list = []


    for i in range(len(angs)):

        if rects[i][1][0]<rects[i][1][1]:
            
            if angs[i]< mean+std and angs[i] > mean-std:

                if rects[i][2]>0:
                    angs[i] = mean + 90

                else:
                    angs[i] = mean - 90

            else:
                if rects[i][2]>0:
                    angs[i] = angs[i]  + 90

                else:
                    angs[i] = angs[i] - 90

        else: 
            if angs[i]< mean+std and angs[i] > mean-std:

                angs[i] = mean

        rects_temp = list((rects[i][0],rects[i][1],angs[i]))


        rects_list.append(rects_temp)



    return tuple(rects_list)









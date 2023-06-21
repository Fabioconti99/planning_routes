# version inspection
import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.projects import point_rend

# TSP libraries
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing

# Kmenas libraries 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import sys
import math
import cv2
import json

# import of my libraries
import plot_panels
import info_distance as info
import mergeGetPoints as mgp
import get_images as gi
import get_csv_data as csv
import cplex_tsp_lib as cp



def main():

    # PATHS
    conf = json.load(open("./content/config.json", "r"))

    # path to kml file
    input_name = conf["paths"]["input_name"]
    kml_file = "../KML_input/"+input_name

    # path to image output
    folder_name = conf["paths"]["dataset_folder_name"]
    path = "../dataset/"+folder_name

    # output dir
    output_dir = "../KML_output"

    # skeleton file for path
    skeleton_file = "../KML_input/skeleton.kml"

    # top left, bottom right, and starting point
    tl, br, st = gi.get_poly(kml_file)


    # CONFIG DATA

    # flight conf
    th = conf["flight_type"]["th"]
    n_pass = conf["flight_type"]["n_pass"]
    dub = conf["flight_type"]["dub"]
    opened = conf["flight_type"]["opened"]
    n_loops = conf["flight_type"]["n_loops"]

    # drone config
    battery = conf["drone_config"]["battery"]

    # detection parameters 
    th_det = conf["detection_param"]["th_det"]
    th_overlap = conf["detection_param"]["th_overlap"]

    # TSP solution type
    tsp_type = conf["tsp_solution"]["tsp_type"]
    th_time = conf["tsp_solution"]["th_time"]


    # altitude param
    alt = conf["altitude"]

    # Picture name (and extraction image)
    pic = gi.get_img(tl,br, path)


    # Path to important locations
    image = path+"/"+pic
    output_image_track = "../dataset/"+folder_name+"/tracks/track_"+pic
    output_image_pred = "../dataset/"+folder_name+"/predictions/pred_"+pic
    output_image_graph = "../dataset/"+folder_name+"/graphs/graph_"+pic
    output_image_BB = "../dataset/"+folder_name+"/bbs/BB_"+pic
    output_mask= "../dataset/"+folder_name+"/masks/detections"+pic+".png"


    # below path applies to current installation location of Detectron2
    cfgFile = "detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"

    # model checkpoint path
    model_path = "detectron2_model/model_final.pth"

    # Option for extending np prints
    np.set_printoptions(threshold=np.inf)

    # Formatting the image
    im = cv2.imread(image)

    # Creating Detectron2 config
    cfg = get_cfg()

    # Add point-rend specific config
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(cfgFile)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2

    cfg.MODEL.WEIGHTS = model_path

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th_det
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = th_overlap

    cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy

    # create predictor
    predictor = DefaultPredictor(cfg)

    # make prediction
    output = predictor(im)

    # Plot prediction
    visualizer = Visualizer(
        im[:, :, ::-1],
        None, 
        scale=0.8, 
    )
    # Plt info
    out = visualizer.draw_instance_predictions(output["instances"].to("cpu"))
    plt.figure(figsize=(20,10))
    plt.imshow(out.get_image()[:, :, ::-1]);plt.title("Bounding box");

    # Save image
    plt.savefig((output_image_pred))


    # Info logs
    print("Inferance complete...")
    print("shape image: ", im.shape[0],im.shape[1])


    # Converting starting point to pixel
    st_pix = gi.coord_to_pix(image,st,tl,br)

    # Pixel size in meters
    pix_m = info.pixel_size(tl, br, im)

    # get the masks
    masks = np.asarray(output["instances"].pred_masks.to("cpu"))

    # Image copyformatting
    image_copy2 = cv2.imread (image)

    # Cycle variables
    rects = []
    boxes = []

    # number of pixel for 1 meter 
    num_pix = (th/pix_m)*2

    # Masks extraction and rectrangles contour approximation
    for mask in masks:

        mask = np.asarray(mask)
        mask = Image.fromarray(mask)
        mask.save(output_mask)

        detect = cv2.imread(output_mask, cv2.IMREAD_GRAYSCALE)
        contour, hierarchy = cv2.findContours(detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rect = cv2.minAreaRect(contour[0])
        rect_list = list(rect[1])

        if rect_list[0]>rect_list[1]:

            rect_list[0] = float(rect_list[0]) + num_pix
        else:
            rect_list[1] = float(rect_list[1]) + num_pix

        rect_temp = list(rect)
        rect_temp[1] = rect_list
        rect = tuple(rect_temp)

        rects.append(rect)
    
    # standardize the angle of the box
    rects = mgp.angle_adjustment(rects)

    # Corner extraction of the boxes
    for rect in rects:

        box = cv2.boxPoints(rect)
        boxes.append(box)
        cv2.drawContours(image_copy2,[box.astype('int')], 0, (0,255,0),2)

    # Extracting the necessary points'coordinates for even and odd number of passages over the panel 
    if n_pass%2 == 0:
        points = mgp.get_points_l(rects,st_pix)
        distance_matrix = info.euclidean_distance(points)

    else:
        points, arr_constraints, arr_center = mgp.get_points_center(rects,st_pix)
        distance_matrix = info.euclidean_distance_mod(points,arr_constraints)

    # Distance matrices
    distance_matrix_real = info.euclidean_distance(points)

    # Decision over having openned or closed cycles
    if opened:
        distance_matrix[:, 0] = 0


    # Drawing the image for each panel
    for i in range(len(points)):
        cv2.drawMarker(image_copy2, (points[i,0].astype(int), points[i,1].astype(int)), (0,0,255), cv2.MARKER_CROSS, 5, 5)

    # plot Info of the image 
    plt.figure(figsize=(20,10))
    plt.imshow(image_copy2[:,:,::-1]);plt.title("Bounding box")
    plt.savefig((output_image_BB))


    # variables for the loop
    n = 1
    k = 0

    # Full permutation
    total_perm = []

    # Same as points but with no initial row
    arr_points = np.delete(points, 0, 0)

    while k != n: 

        # K-means algorithm applied over the centers of the panels
        # to clusterize the area in sub-groups of panels, based on 
        # positioning. This is done to favor the completeness of a 
        # tour keeping into account the energy consumption. 

        km = KMeans(
            n_clusters = n, init='random',
            n_init=10, max_iter=300, 
            tol=1e-04, random_state=0
            )

        if n_pass%2 == 0:
            M_km = km.fit(arr_points)
        else:
            M_km = km.fit(arr_center)

        labels = M_km.labels_
        unique_labels = np.unique(labels)

        shared_row_indices = []

        # Solve TSP for each cluster of nodes created by the K-means function
        for label in unique_labels:

            indices = np.where(labels == label)[0]

            # Setting up the points for both the Even and Odd cases
            if n_pass%2 == 0:

                # All its needed is the left extream of the panel
                cluster_nodes = arr_points[indices]
                cluster_nodes_exp = cluster_nodes
                cluster_nodes_exp = np.vstack(([st_pix[0][0],st_pix[0][1]], cluster_nodes_exp))

                if (cluster_nodes == [st_pix[0][0],st_pix[0][1]]).any():
                    r = len(indices)-1

                else: 
                    r = len(indices)


            else:
                cluster_nodes = arr_center[indices]

                # Expansion of the nodes into couples. At the extreams of the panels.
                cluster_nodes_exp, constr_exp = mgp.cluster_expansion(cluster_nodes, rects, arr_center,st_pix)

                r = len(indices)

            boxes_temp = []


            for i in range(r):
                box = cv2.boxPoints(rects[indices[i]])
                boxes_temp.append(box)


            print("Number of nodes: "+ str(np.shape(cluster_nodes)[0])) 

            if tsp_type!="branch_and_cut":
                # Calculating the matrix of distances
                if n_pass%2 == 0:
                    cluster_exp_dist_mat_mod = info.euclidean_distance(cluster_nodes_exp)

                else:
                    cluster_exp_dist_mat_mod = info.euclidean_distance_mod(cluster_nodes_exp,constr_exp)
                    #print("cluster_nodes_exp:\n",cluster_nodes_exp,"constr_exp:\n",constr_exp)


            if tsp_type == "iterated_local_search":
                # Type of TSP cycle
                if opened:
                    cluster_exp_dist_mat_mod[:, 0] = 0
                # Solving TSP as an ITERATED LOCAL SEARCH
                dist_prev = 1000000000000
                for i in range(n_loops):
                    perm, dist = solve_tsp_local_search(cluster_exp_dist_mat_mod)

                    if dist_prev > dist:
                        dist_prev = dist
                        perm_cluster_exp = perm

            elif tsp_type == "tsp_simulated_annealing":
                perm_cluster_exp = solve_tsp_simulated_annealing(cluster_exp_dist_mat_mod)

            elif tsp_type == "branch_and_cut":

                if n_pass%2 == 0:
                    perm_cluster_exp = cp.tsp_cplex(cluster_nodes_exp, th_time)

                else:
                    perm_cluster_exp = cp.tsp_cplex_mod(cluster_nodes_exp,constr_exp, th_time)

            print("perm_cluster_exp info:")
            print(type(perm_cluster_exp))
            print(perm_cluster_exp)
            # Find the total distance to travel 
            if n_pass%2 == 0:
                wps, perm_temp = mgp.expansion_plan_even(perm_cluster_exp, cluster_nodes_exp, boxes_temp, n_pass, dub,st_pix)
            else:
                wps, perm_temp = mgp.expansion_plan(perm_cluster_exp, cluster_nodes_exp,constr_exp, boxes_temp, n_pass, dub,st_pix)

            # Adding the return point to the cycle
            perm_temp = np.hstack((perm_temp,0))

            # Creation of the distance matrix
            cluster_exp_dist_mat = info.euclidean_distance(wps)
            total_cost = info.connect_permutation(cluster_exp_dist_mat, perm_temp)

            # Verify if the distance is feasable with respect to the battery 
            if total_cost*pix_m < battery:

                k = k + 1

                print("used: "+str(k))
                perm_cluster_exp = mgp.actual_perm(points,cluster_nodes_exp,perm_cluster_exp)

                total_perm.extend(perm_cluster_exp)

                
                # Find the indices of rows in points or arr_center that are also in cluster_nodes
                if n_pass%2 == 0:
                    shared_row_index = np.where((arr_points[:, None, :] == cluster_nodes).all(-1).any(1))[0]

                else: 
                    shared_row_index = np.where((arr_center[:, None, :] == cluster_nodes).all(-1).any(1))[0]
                
                shared_row_indices.extend(shared_row_index)


        # Indexes checks 
        if k != n and k == 0:

            print("k = 0")
            print("n (not good): "+str(n)+"\n n going to try: "+str(n+1) )
            n = n + 1

        # If k<n only a few clusters were accepted. They'll be saved but the 
        # Remaining ones will be divided into sub-sets again starting from 1
        # to be checked for feasibility.
        if k != n and k != 0:

            print("k: "+str(k))
            print("n: "+str(n))

            k = 0
            n = 2

            print("n going to try: "+str(n))

            # Remove any shared rows from arr_center
            if n_pass%2 == 0:
                arr_points = np.delete(arr_points, shared_row_indices, axis=0)

            else: 
                arr_center = np.delete(arr_center, shared_row_indices, axis=0)

        # If the two index metch all the clusters where accepted
        if k == n:
            final_permutation = np.array(total_perm)

            if n_pass%2 == 0:
                arr_points = np.delete(arr_points, shared_row_indices, axis=0)

            else: 
                arr_center = np.delete(arr_center, shared_row_indices, axis=0)

            print("n=k: "+str(n))

    print("Final permuatation:")

    # Calculating the final permutation wpoints
    if n_pass%2 == 0:
        wp, final_perm = mgp.expansion_plan_even(final_permutation, points, boxes, n_pass, dub, st_pix)
    else:
        wp, final_perm = mgp.expansion_plan(final_permutation, points, arr_constraints, boxes, n_pass, dub,st_pix)

    # Final permutation
    final_perm = np.append(final_perm,[0])

    wp = np.vstack((wp,wp[0])) 

    final_permutation = np.append(final_permutation,[0])
    print("final_permutation:",str(final_permutation))
    print("final_perm:",str(final_perm))

    # Get the geo-coordinates of the points
    geo_coordinates = gi.get_geographic_coordinates2(image, tl, br, wp)


    # Get CSV info
    if alt:
        csv.get_csv_alt(final_permutation,n_pass,st,geo_coordinates)
    else:
        csv.get_csv(wp,n_pass,st,st_pix,geo_coordinates)


    # Get the poly-line in kml format
    gi.get_kml_line(geo_coordinates,output_dir,kml_file,skeleton_file)
    plot_panels.plot_waypoints(wp, final_perm, image, (output_image_track))


    plt.show()

   

if __name__ =="__main__":
    main()

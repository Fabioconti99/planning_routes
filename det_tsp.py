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

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pykml import parser

import os
import sys
import math
import cv2
from shutil import move

import plot_panels
import info_distance as info
import mergeGetPoints as mgp
import get_images as gi

def main():

    # PATHS
    # path to kml file
    kml_file = "../KML_input/input.kml"

    # path to image output
    path = "../dataset/casella_20230419"

    kml_file = "../KML_input/input.kml"

    # output dir
    output_dir = "../KML_output"

    # skeleton file for path
    skeleton_file = "../KML_input/skeleton.kml"


    #tl=[ 42.90477704470378, 12.080922046093901]
    #br=[ 42.903797161795865, 12.08216765415183]

    tl=[ 8.63970255, 44.78174367]
    br=[ 8.6413979, 44.77998829]


    print(tl,br)

    th = 1

    pic = gi.get_img(tl,br, path)

    print("path to image:",path+"/"+pic)
 
    image = path+"/"+pic

    json_file = "../test/panel_"+pic+"_file.json"
    output_image_track = "../dataset/casella_20230419/tracks/track_"+pic
    output_image_pred = "../dataset/casella_20230419/predictions/pred_"+pic
    output_image_graph = "../dataset/casella_20230419/graphs/graph_"+pic
    output_image_BB = "../dataset/casella_20230419/bbs/BB_"+pic
    output_data = "../test/track_"+pic+"_data.txt"
    output_mask= "../dataset/casella_20230419/masks/detections"+pic+".png"

    # below path applies to current installation location of Detectron2
    cfgFile = "detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"

    # model checkpoint path
    model_path = "detectron2_model/model_final.pth"

    # modality:
    try: 
        n_pass = int(input("Number of passages over the panels: "))
    except:
        print("wrong value n_pass")
        exit()

    # double:

    try:
        dub = bool(input("Modality: "))
    except:
        print("wrong value dub")
        exit()

    np.set_printoptions(threshold=np.inf)

    im = cv2.imread(image)

    # create config
    cfg = get_cfg()

    # Add point rend speofoc config
    point_rend.add_pointrend_config(cfg)

    

    cfg.merge_from_file(cfgFile)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

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

    out = visualizer.draw_instance_predictions(output["instances"].to("cpu"))

    plt.figure(figsize=(20,10))

    plt.imshow(out.get_image()[:, :, ::-1]);plt.title("Bounding box");

    plt.savefig((output_image_pred))



    print("Inferance complete...")

    print("shape image: ", im.shape)

    # Pixel size in meters
    pix_m = info.pixel_size(tl, br, im)


    # get the masks
    masks = np.asarray(output["instances"].pred_masks.to("cpu"))

    # getting the merged data 
    merged = mgp.merge_boolean_arrays(masks)

    # making it into an array to keep track of the size
    merged = np.asarray(merged)

    print("shape merged: ", merged.shape)

    # from array to image
    merged = Image.fromarray(merged)

    # save the image
    merged.save(output_mask)

    # import it as a gray scale image
    detections = cv2.imread("../dataset/casella_20230419/detections"+pic+".png", cv2.IMREAD_GRAYSCALE)

    # print the size of the image
    print("shape merged image: ", detections.shape)


    # find the countours of the image
    contours, hierarchy = cv2.findContours(detections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours found = {}".format(len(contours)))


    image_copy2 = cv2.imread (image)

    rects = []

    boxes = []

    # number of pixel for 1 meter 
    num_pix = (th/pix_m)*2

    for contour in contours:

        rect = cv2.minAreaRect(contour) 
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
    print(rects[15],rects[13])
    rects = mgp.angle_adjustment(rects)

    

    for rect in rects:

        box = cv2.boxPoints(rect)
        boxes.append(box)
        cv2.drawContours(image_copy2,[box.astype('int')], 0, (0,255,0),2)

    #for rec in rects:
    #    print(rec[2])

    # If the number of passages over the panels is even, the identification of the key points will be the left side 
    if n_pass%2 == 0:

        points = mgp.get_points_l(rects)
        distance_matrix = info.euclidean_distance(points)

    # If the number of passages over the panels is odd, the identification of the key points will be the center  
    else:

        points, arr_constraints, arr_center = mgp.get_points_center(rects)
        distance_matrix = info.euclidean_distance_mod(points,arr_constraints)

    for i in range(len(points)):

        cv2.drawMarker(image_copy2, (points[i,0].astype(int), points[i,1].astype(int)), (0,0,255), cv2.MARKER_CROSS, 1, 5)

    plt.figure(figsize=(20,10))
    plt.imshow(image_copy2[:,:,::-1]);plt.title("Bounding box")
    plt.savefig((output_image_BB))


    distance_matrix[:, 0] = 0


    for j in range(1):

        best_dist_3 = 100000000000000.0
        #best_dist_6 = 0.0

        best_perm_3 = np.array([0.0, 0.0])
        #best_perm_6 = np.array([0.0, 0.0])

        dist_arr_3 = np.array([31000])
        #dist_arr_6 = np.array([0.0])

        n = 0

        for i in range(1):

            permutation_3, d = solve_tsp_local_search(distance_matrix)


            if n_pass%2 == 0:
                wps, perm_temp = mgp.expansion_plan_even(permutation_3, points, boxes, n_pass, dub)
            else:
                wps, perm_temp = mgp.expansion_plan(permutation_3, points,arr_constraints, boxes, n_pass, dub)


            distance_matrix_real = info.euclidean_distance(wps)
            distance_3 = info.connect_permutation(distance_matrix_real, perm_temp)


            if distance_3 < best_dist_3: 

                best_dist_3 = distance_3
                best_perm_3 = permutation_3

            dist_arr_3 = np.hstack((dist_arr_3, np.array([best_dist_3])))

            print("Loop "+str(n))
            print(" d_3: "+str(distance_3))

            n = n + 1


        dist_arr_3 = dist_arr_3[1:]

        print ("Best distance: "+ str(best_dist_3))
        print ("Best permutation: "+ str(best_perm_3))

        f = open (output_data, "a")
        f.write("\n Best distance: "+ str(best_dist_3))
        f.write("\n Best permutation: "+ str(best_perm_3))

        f.close()



        if n_pass%2 == 0:
            wp, final_perm = mgp.expansion_plan_even(best_perm_3, points, boxes, n_pass, dub)
        else:
            wp, final_perm = mgp.expansion_plan(best_perm_3, points, arr_constraints, boxes, n_pass, dub)


        #print(wp)
        geo_coordinates = gi.get_geographic_coordinates2(image, tl, br, wp)

        gi.get_kml_line(geo_coordinates,output_dir,kml_file,skeleton_file)


        plot_panels.plot_waypoints(wp, final_perm, image, (output_image_track))
        #plot_panels.plot_waypoints(points, best_perm_3, image, (output_image_track))


        # visualize your prediction

        fig = plt.figure(figsize=(20, 10))

        plt.plot(dist_arr_3)

        # Add labels and title
        plt.xlabel('iterations')
        plt.ylabel('Pixel distance value')
        plt.title('Distances')
        plt.savefig((output_image_graph))

        plt.show()
if __name__ == "__main__":

    main()




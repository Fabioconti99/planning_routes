import sys
import os
from shutil import move
import math
import json

import info_distance as info

from PIL import Image
from pykml import parser
from pykml.factory import KML_ElementMaker as KML
from lxml import etree
import numpy as np 


#lat == y
#lon == x

# Parameters for files location
conf = json.load(open("./content/config.json", "r"))

shift_y = conf["plot_param"]["shift_y"]
shift_x = conf["plot_param"]["shift_x"]

# Find the geographic coordinates of a some given points
def get_geographic_coordinates2(pic, tl, br, pixel_coords):
    global shift_y
    global shift_x

    img = Image.open(pic)

    geo_coords = np.zeros((int(np.size(pixel_coords,0)), int(np.size(pixel_coords,1))))

    k = 0
    for wp in pixel_coords:

        x = tl[1] + (wp[0]-shift_x)/(img.size[0]-256)*(br[1]-tl[1])
        y = tl[0] + (wp[1]-shift_y)/(img.size[1]-256)*(br[0]-tl[0])

        geo_coords[k] = [y, x]
        k = k + 1

    k = 0
    return geo_coords

# From geographic coordinates to lixel coordinates of a given point
def coord_to_pix(pic,st,tl,br):
    global shift_y
    global shift_x

    img = Image.open(pic)

    pix = np.zeros((1,2))

    pix[0][0] = shift_x+(((st[1]-tl[1])*(img.size[0]-256))/(br[1]-tl[1]))
    pix[0][1] = shift_y+(((st[0]-tl[0])*(img.size[1]-256))/(br[0]-tl[0]))

    return pix

# It returns the top_left and bottom_right points of the polygon
def get_poly(kml_file):

    # Load the KML file and extract the polygon
    kml_doc = parser.parse(open(kml_file, 'rb'))
    poly = kml_doc.getroot().Document.Folder.Placemark[0].Polygon.outerBoundaryIs.LinearRing.coordinates.text.split(',')

    coord = kml_doc.getroot().Document.Folder.Placemark[1].Point.coordinates.text.split(',')

    # 
    extreams = []
    for i in range(len(poly)-3): 

        if  " " in str(poly[i+1]) :

            poly[i+1] = poly[i+1].split(" ")

            extreams.append(float(poly[i+1][1]))
        else:
            extreams.append(float(poly[i+1]))

    extreams = np.array(extreams).reshape(int((len(poly)-3)/2), 2)

    print("extreams:\n",extreams)
    tl = np.array([np.max(extreams[:,0]), np.min(extreams[:,1])])
    br = np.array([np.min(extreams[:,0]), np.max(extreams[:,1])])
    print("tl:",tl)
    print("br:",br)
    st = np.array([float(coord[1]),float(coord[0])])

    #return extreams
    return tl, br, st

# Extractong the Imge described by the extreams form google earth
def get_img(tl, br, path):

    cmd = "./executables/sat-img.out gsat {} {} {} {} {}".format(
        min(tl[1], br[1]),
        max(tl[0], br[0]),
        max(tl[1], br[1]),
        min(tl[0], br[0]),
        path
    )
    print(cmd)

    name = "map-{}-{}-{}-{}.jpg".format(
        tl[0], tl[1], br[0], br[1]
    )

    images = [x for x in os.listdir(path) if x[-4:] == ".jpg"]

    if not name in images: 
        #lancio il comando che genera l'immagine da google
        os.system(cmd)
        move("{}/report.jpg".format(path), "{}/{}".format(path, name))

    print("Image generated")

    return name

# Returns the ouput kml with a polyline describing the final permutation
def get_kml_line(geo_coordinates,output_dir,kml_file,skeleton_file):

    kml_doc = parser.parse(open(kml_file, 'rb'))
    name = str(kml_doc.getroot().Document.name.text)
    name = name.replace(".kml']", "")
    name = name.replace("['", "")
    name = name + "_output"

    geocoods = ""
    for coords in geo_coordinates: 
        
        geocoods = geocoods+(str(coords[1]))
        geocoods = geocoods+(",")
        geocoods = geocoods+(str(coords[0]))
        geocoods = geocoods+(",0 ")


    f = open(skeleton_file, "r")
    file_str = f.read()
    file_str = file_str.replace("<name></name>", "<name>"+name+"</name>")
    file_str = file_str.replace("<coordinates></coordinates>", "<coordinates>"+geocoods+"</coordinates>")

    f.close()

    f = open(output_dir+"/"+name+".kml", 'w')
    f.write(file_str)
    f.close()
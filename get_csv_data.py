import rasterio
import os
import json

# Initialization of som variable from the command json file
conf = json.load(open("./content/config.json", "r"))

tif_name = conf["paths"]["tif_name"]
csv_name = conf["paths"]["csv_name"]

tif_name = "./content/"+tif_name
csv_name = "../CSV_output/"+csv_name

dem = rasterio.open(tif_name)

# Extractiong the altitude from the .tif file
def get_altitude(lon, lat):

    global dem

    row, col = dem.index(lon, lat)
    dem_data = dem.read(1)

    try:
        altitude = str(dem_data[row,col])
        
    except: 
        altitude = "NaN"

    return altitude

# Get the data of the Way-points written onto an output csv with altitude
def get_csv_alt(final_permutation,n_pass,st,geo_coordinates):

    global csv_name
    with open(csv_name, "w") as file:
        file.write("id,num,lat,lon\n")
        n = 0
        k = 0
        n_vela = 0
        num_pass = 1

        for geo in geo_coordinates:
            lat = geo[0]
            lon = geo[1]

            if final_permutation[k] == 0 :

                file.write("{},{},{},{},{}\n".format("Start","NaN",str(st[1]),str(st[0]),str(get_altitude(st[1], st[0]))))

            else:

                if n%(n_pass*2) == 0:

                    n_vela = n_vela+1
                    num_pass = 1

                file.write("{},{},{},{},{}\n".format(str(n_vela),str(num_pass),str(lon),str(lat),str(get_altitude(lon, lat))))
                num_pass = num_pass+1

                n = n + 1
            k = k + 1


# Get the data of the Way-points written onto an output csv
def get_csv(wp,n_pass,st,st_pix,geo_coordinates):

    global csv_name
    with open(csv_name, "w") as file:
        file.write("id,num,lat,lon\n")
        n = 0
        k = 0
        n_vela = 0
        num_pass = 1

        for geo in geo_coordinates:
            lat = geo[0]
            lon = geo[1]

            if wp[k][0] == st_pix[0][0] and wp[k][1] == st_pix[0][1]:

                file.write("{},{},{},{}\n".format("Start","NaN",str(st[1]),str(st[0])))

            else:

                if n%(n_pass*2) == 0:

                    n_vela = n_vela+1
                    num_pass = 1

                file.write("{},{},{},{}\n".format(str(n_vela),str(num_pass),str(lon),str(lat)))
                num_pass = num_pass+1

                n = n + 1
            k = k + 1
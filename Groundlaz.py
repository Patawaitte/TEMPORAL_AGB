'''
Created on 01/11/2022

@author: Pauline Perbet
http://awsassets.panda.org/downloads/annex_5__fusionexercises_tgo_wwf_workshop_1.pdf
'''


import os, sys
import subprocess as sp
import glob
import geopandas as gpd
import pandas as pd

chemin = "C:/Lidar/"
laz_filepath = chemin + "LAZ_synchron/"
#name = glob.glob(laz_filepath+"Polygon_"+num_polygone[i]+"*")

for filename in os.listdir(laz_filepath):
    f = os.path.join(laz_filepath, filename)
    print(filename)
    print('f',f)

    Ground = "C:/Lidar/FUSION/GroundFilter.exe " + chemin + "LAZ_synchron_ground/" + filename[:-4]+ "_Ground.laz " + "5 " + f
    sp.call(Ground)
    print("Ground")
    Ground_surface = "C:/Lidar/FUSION/GridSurfaceCreate.exe " + chemin + "LAZ_synchron_ground/" + filename[:-4] + "_Ground_surface.dtm " + "1 m m 1 0 0 0 " + chemin + "LAZ_synchron_ground/" + filename[:-4] + "_Ground.laz"

    sp.call(Ground_surface)
    print("Ground_surface")
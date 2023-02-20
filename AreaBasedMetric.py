'''
Created on 16/01/2023

@author: Pauline Perbet
http://awsassets.panda.org/d
'''

import os, sys
import subprocess as sp
import glob
from tqdm import tqdm
#import geopandas as gpd
#import pandas as pd
# Path to Laz and output

chemin = "C:/Lidar/"
laz_filepath = "D:/2017/2pt5/laz/"
#name = glob.glob(laz_filepath+"Polygon_"+num_polygone[i]+"*")
Output_filepath = chemin + "Output_2017_2pt5/"

for filename in tqdm(os.listdir(laz_filepath)[48400+9000:]):

    f = os.path.join(laz_filepath, filename)
    print(filename)
    print('f',f)


    Ground = "C:/Lidar/FUSION/GroundFilter.exe " + Output_filepath + filename[:-4]+ "_Ground.laz " + "5 " + f
    sp.call(Ground)
    Ground_surface = "C:/Lidar/FUSION/GridSurfaceCreate.exe " + Output_filepath + filename[:-4] + "_Ground_surface.dtm " + "1 m m 1 0 0 0 " + Output_filepath + filename[:-4]+ "_Ground.laz "

    sp.call(Ground_surface)

    GridMetrics = "C:/Lidar/FUSION/GridMetrics.exe  /nointensity  /outlier:2,35 /minht:2 " + Output_filepath + filename[:-4] + "_Ground_surface.dtm " +"2 30 "+ Output_filepath+filename[:-4]+"gridmetrics.csv " + laz_filepath + filename[:-4]+ ".laz "
    sp.call(GridMetrics)

    #GridMetrics_first = "C:/Lidar/FUSION/GridMetrics.exe /first /nointensity  /outlier:2,35 /minht:2 " + Output_filepath + filename[:-4] + "_Ground_surface.dtm " +"2 30 "+ Output_filepath+filename[:-4]+"gridmetrics_first.csv " + laz_filepath + filename[:-4]+ ".laz "
    #sp.call(GridMetrics_first)

    #CSV2Grid= "C:/Lidar/FUSION/CSV2Grid.exe " + Output_filepath+filename[:-4]+"gridmetrics_all_returns_elevation_stats.csv " +"  7 "+  Output_filepath+filename[:-4]+ "elevmax.asc"
    # sp.call(CSV2Grid)


    os.remove(Output_filepath + filename[:-4]+ "_Ground.laz")

print('alldone')
import os
import os.path
import shutil
import glob
import pandas as pd
import geopandas as gpd

textFile = ("C:\\Lidar\\shp\\index_08451.shp")
sourceFolder = ("C:\\Lidar\\Output_2017_grid\\")
destinationFolder = ("C:\\Lidar\\Output_2017_grid_08451")

f= gpd.read_file(textFile)
#f = open(textFile, "r").readlines()
for i in f['ID_TUILE']:
    print(i)
    #ListFile= glob.glob(os.path.join(sourceFolder,"**",i.strip()),recursive=True)
    #ListFile= glob.glob(os.path.join(sourceFolder,"**","**","**",i.strip()+".laz"),recursive=True)
    ListFile= glob.glob(sourceFolder+i+"*.tif")
    print(sourceFolder+i+"*.tif")
    print(ListFile)
    if len(ListFile):
        print(ListFile[0],destinationFolder,os.path.basename(ListFile[0]))
        destinationfile=os.path.join(destinationFolder,os.path.basename(ListFile[0]))
        shutil.copyfile(ListFile[0],destinationfile)
    else:
        print(i,"-File not found")
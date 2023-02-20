'''
Created on 01/11/2022

@author: Pauline Perbet
Extraction of raster metrics...
Faire boucles pour toute les métrics... GDAL/gdalbuildvrt.exe -separate
Faire boucles pour toutes les dalles...
'''

import os, sys
import subprocess as sp
import glob

chemin = "C:/Lidar/"
laz_filepath = chemin +"sample/"

Ground = "C:/Lidar/FUSION/GroundFilter.exe C:/Lidar/sample/tile_248000_5252750_ground.laz 1 C:/Lidar/sample/tile_248000_5252750.laz"
Ground_surface =chemin+"FUSION/GridSurfaceCreate.exe C:/Lidar/sample/tile_248000_5252750_ground.dtm 1 m m 0 0 0 0 C:/Lidar/sample/tile_248000_5252750_ground.laz"


sp.call(Ground)
sp.call(Ground_surface)
metrics =chemin+"FUSION/gridmetrics.exe /minht:2 /nointensity  C:/Lidar/sample/tile_248000_5252750_ground.dtm 2 30 C:/Lidar/sample/tile_248000_5252750_metrics.csv C:/Lidar/sample/tile_248000_5252750.laz"
sp.call(metrics)

tiff =chemin+"FUSION/CSV2Grid.exe C:/Lidar/sample/tile_248000_5252750_metrics_all_returns_elevation_stats.csv 6 elev6248000_5252750_30m.asc"
tiff2 =chemin+"FUSION/CSV2Grid.exe C:/Lidar/sample/tile_248000_5252750_metrics_all_returns_elevation_stats.csv 7 elev7248000_5252750_30m.asc"

sp.call(tiff)
sp.call(tiff2)

#Merge_Data= chemin+"FUSION/MergeRaster.exe combined248000_5252750.asc elev6.asc elev7.asc"

#sp.call(Merge_Data)



#for name in glob.glob(laz_filepath+"*"):
#    name_ = name.replace('/', '\\')
#    head, tail = os.path.split(name_)
#    num=tail[:-4]
#    print(name_)

    #metrics =chemin+"FUSION/gridmetrics.exe "+ "/minht:2  /outlier:2,30 /nointensity  /nointdtm 3 30 " + head[:-6] + num+".csv" + name_
    #sp.call(metrics)


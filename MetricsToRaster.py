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
input_filepath = "D:/Output_2017_2pt5/"
#name = glob.glob(laz_filepath+"Polygon_"+num_polygone[i]+"*")
Output_filepath = chemin + "Output_2017_grid/"

for filename in tqdm(os.listdir(input_filepath)):

    f = os.path.join(input_filepath, filename)
    print(filename)
    print('f',f)


    CSV2Grid7= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  7 "+  Output_filepath+filename[:-4]+ "elevmax.tif"
    sp.call(CSV2Grid7)

    CSV2Grid8= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  8 "+  Output_filepath+filename[:-4]+ "elevmin.tif"
    sp.call(CSV2Grid8)

    CSV2Grid25= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  25 "+  Output_filepath+filename[:-4]+ "ElevP05.tif"
    sp.call(CSV2Grid25)

    CSV2Grid26= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  26 "+  Output_filepath+filename[:-4]+ "ElevP10.tif"
    sp.call(CSV2Grid26)

    CSV2Grid27= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  27 "+  Output_filepath+filename[:-4]+ "ElevP20.tif"
    sp.call(CSV2Grid27)

    CSV2Grid28= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  28 "+  Output_filepath+filename[:-4]+ "ElevP25.tif"
    sp.call(CSV2Grid28)

    CSV2Grid29= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  29 "+  Output_filepath+filename[:-4]+ "ElevP30.tif"
    sp.call(CSV2Grid29)

    CSV2Grid36= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  36 "+  Output_filepath+filename[:-4]+ "ElevP90.tif"
    sp.call(CSV2Grid36)

    CSV2Grid38= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  38 "+  Output_filepath+filename[:-4]+ "ElevP99.tif"
    sp.call(CSV2Grid38)

    CSV2Grid49= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  49"+  Output_filepath+filename[:-4]+ "P_first_r_above2.tif"
    sp.call(CSV2Grid49)

    CSV2Grid54= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  54"+  Output_filepath+filename[:-4]+ "P_first_r_abovemean.tif"
    sp.call(CSV2Grid54)

    CSV2Grid51= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  51"+  Output_filepath+filename[:-4]+ "All_r_above2_Total_first_r.tif"
    sp.call(CSV2Grid51)

    CSV2Grid58= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  58"+  Output_filepath+filename[:-4]+ "All_r_abovemean_Total_first_r.tif"
    sp.call(CSV2Grid58)

    CSV2Grid17= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  17"+  Output_filepath+filename[:-4]+ "ElevL1.tif"
    sp.call(CSV2Grid17)

    CSV2Grid14= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  14"+  Output_filepath+filename[:-4]+ "Elevskewness.tif"
    sp.call(CSV2Grid14)

    CSV2Grid50= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  50"+  Output_filepath+filename[:-4]+ "P_all_r_above2.tif"
    sp.call(CSV2Grid50)

    CSV2Grid56= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  56"+  Output_filepath+filename[:-4]+ "P_all_r_abovemean.tif"
    sp.call(CSV2Grid56)

    CSV2Grid41= "C:/Lidar/FUSION/CSV2Grid.exe " + input_filepath+filename +"  41"+  Output_filepath+filename[:-4]+ "Return3countabove2.tif"
    sp.call(CSV2Grid41)

print('alldone')
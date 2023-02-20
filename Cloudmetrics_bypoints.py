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
#Path to Laz and output

for crs in range(2946, 2952+1):
    chemin = "C:/Lidar/"
    laz_filepath = chemin +"LAZ_synchron/"
    outputclip = chemin +"clip_synchro/"
    ground =  chemin +"LAZ_synchron_ground/"

    #Getting the sample area bounding box from shapefile points
    r = 11.28
    df = gpd.read_file("C:/Lidar/shp/Placette_synchrone_1an_lazname.shp")

    df.crs = {'init' : 'epsg:6622'}
    df=df.to_crs('epsg:'+str(crs))
    df['x']  = df.geometry.x
    df['y']  = df.geometry.y
    print(df['x'])
    df['Xmin'] = (df.geometry.x)-r
    df['Ymin'] = (df.geometry.y)-r
    df['Xmax'] = (df.geometry.x)+r
    df['Ymax'] = (df.geometry.y)+r

    df['coordplot'] = df[['Xmin', 'Ymin', 'Xmax', 'Ymax']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    #Iterate over the shapefile
    df = df.reset_index()
    print(df.head())
    for index, row in df.iterrows():
        print(row['idmes'])
        #namelaz=(row['filenam'].split("/"))[3] #Get the name of the intersept laz data
        namelaz=(row['laz_name']+'_dc.laz')
        print(laz_filepath+namelaz)
        nameclip=(str(row['idmes'])+'.laz')
        print('nameclip',nameclip)

        clip =  "C:/Lidar/FUSION/clipdata.exe /shape:1 /height /dtm:" + ground + namelaz[:-4] + "_Ground_surface.dtm "+laz_filepath+namelaz+ " "+outputclip+"clip_"+nameclip+ " "+row['coordplot']
        #clip =  "C:/Lidar/FUSION/polyclipdata.exe C:/Lidar/shp/Points_1sample_buff.shp C:/Lidar/stand.las C:/Lidar/sample/tile_248000_5252750.laz"

        sp.call(clip)
        print('pathclip',  chemin + outputclip+"clip_"+nameclip+ ".laz")


    cloudmetric = "C:/Lidar/FUSION/CloudMetrics.exe /new /id  /above:2 /minht:2  "+outputclip+"*.laz"+" "+chemin+"samplemetrics_allreturn_noint.csv"
    sp.call(cloudmetric)

    print('done')


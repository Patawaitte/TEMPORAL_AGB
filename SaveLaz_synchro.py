import os
import os.path
import shutil
import glob
import pandas as pd

textFile = ("C:\\Lidar\\lazname_synchro.csv")
sourceFolder = ("D:\\")
destinationFolder = ("C:\\Lidar\\LAZ_synchron")

f= pd.read_csv(textFile)
#f = open(textFile, "r").readlines()
for i in f['laz_name']:
    print(i)
    #ListFile= glob.glob(os.path.join(sourceFolder,"**",i.strip()),recursive=True)
    #ListFile= glob.glob(os.path.join(sourceFolder,"**","**","**",i.strip()+".laz"),recursive=True)
    ListFile= glob.glob(sourceFolder+"\\**\\**\\**\\"+i+"_dc.laz",recursive=True)
    print(sourceFolder+"**\\**\\**\\"+i+".laz")
    print(ListFile)
    if len(ListFile):
        print(ListFile[0],destinationFolder,os.path.basename(ListFile[0]))
        destinationfile=os.path.join(destinationFolder,os.path.basename(ListFile[0]))
        shutil.copyfile(ListFile[0],destinationfile)
    else:
        print(i,"-File not found")
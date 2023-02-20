import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from glob import glob
import os, fnmatch
import numpy as np
import subprocess
from osgeo import gdal
from rasterio.crs import CRS



my_wkt="""PROJCRS["NAD_1983_Canada_Lambert",BASEGEOGCRS["NAD83",DATUM["North American Datum 1983", ELLIPSOID["GRS 1980",6378137,298.257222101004, LENGTHUNIT["metre",1]],ID["EPSG",6269]],PRIMEM["Greenwich",0,ANGLEUNIT["Degree",0.0174532925199433]]],CONVERSION["unnamed", METHOD["Lambert Conic Conformal (2SP)", ID["EPSG",9802]],PARAMETER["Latitude of false origin",0,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8821]], PARAMETER["Longitude of false origin",-95,ANGLEUNIT["Degree",0.0174532925199433], ID["EPSG",8822]], PARAMETER["Latitude of 1st standard parallel",49, ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",77,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1], ID["EPSG",8827]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1], LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]"""
dst_projection = CRS.from_wkt(my_wkt)

inFolder= '/home/beland/PycharmProjects/MyProjectDisturb/results/COMPUTECANADA_trans/Clip6_trans/'
outFolder= '/home/beland/PycharmProjects/MyProjectDisturb/results/COMPUTECANADA_trans/Clip6_trans_good/'



# if not os.path.exists(outFolder):
#     os.mkdir(outFolder)
# # # # # CORRIGE DALLE EXPORTER POUR FAIRE MOSAIK
# def findRasters(path, filter):
#     for root, dirs, files in os.walk(path, filter):
#         for file in fnmatch.filter(files, filter):
#             yield os.path.join(root, file)
# for raster in findRasters (inFolder, '*.tif'):
#     (infilepath, infilename)= os.path.split (raster)
#     outRaster= outFolder+ infilename
#     print(raster)
#     print(outRaster)
# #
#     warp= 'gdalwarp -tr 30 30 '+raster+' '+ outRaster
#     os.system(warp)


#warp= 'gdalwarp -tr 30 30 /home/beland/PycharmProjects/MyProjectDisturb/results/COMPUTECANADA/Clip4/result_2048_2048_dis_type_2.tif /home/beland/PycharmProjects/MyProjectDisturb/results/COMPUTECANADA/Clip4/good_result_2048_2048_dis_type_2.tif'
#os.system(warp)


# # # # # MOSAIK par dalle
# listclip=['Clip1','Clip2','Clip3','Clip4','Clip5','Clip6', 'Clip7', 'Clip20', 'Clip21', 'Clip30', 'Clip31']
# listclip=['Clip1','Clip2','Clip3','Clip4','Clip5','Clip6', 'Clip10', 'Clip20', 'Clip21', 'Clip30', 'Clip31']
#
# for Clipnum in listclip:
#
#   # Clipnum='Clip31'
#   listresult=['date_1','date_2','type_1','type_2','sev_1','sev_2']
#
#
#   for x in listresult:
#
#     RESULT ="/home/beland/PycharmProjects/MyProjectDisturb/results/COMPUTECANADA_trans/"+Clipnum+"_trans_"+x+".tif"
#
#     clipmosaic = glob('/home/beland/PycharmProjects/MyProjectDisturb/results/COMPUTECANADA_trans/'+Clipnum+'_trans_good/*'+x+'.tif')
#     # files_string = " ".join(files_to_mosaic)
#     #files_to_mosaic = glob('/home/beland/PycharmProjects/Data_biomass/BIOMASS/Zone03/*.tif')
#
#     vrt = gdal.BuildVRT("/home/beland/PycharmProjects/MyProjectDisturb/results/COMPUTECANADA_trans/merged.vrt", clipmosaic,  VRTNodata= '-99' )
#     #gdal.Translate("/home/beland/PycharmProjects/Data_biomass/Mosaic_biomass/30m_NAD83_landsat/MOSAIC_zone3_30M_BIOMASS_.tif", vrt)
#     gdal.Translate(RESULT, vrt)
#
#     Image = gdal.Open(RESULT, 1)  # open image in read-write mode
#     Band = Image.GetRasterBand(1)
#
#     gdal.SieveFilter(srcBand=Band, maskBand=None, threshold=3, connectedness=8, dstBand=Band)
#     del Image, Band
#     vrt = None


# # # # # # MOSAIKall
listresult=['date_1','date_2','type_1','type_2','sev_1','sev_2']

for x in listresult:
  RESULT ="/home/beland/PycharmProjects/MyProjectDisturb/results/COMPUTECANADA_trans/Partialallclip/Partialall_"+x+".tif"

  clipmosaic = glob('/home/beland/PycharmProjects/MyProjectDisturb/results/COMPUTECANADA_trans/*'+x+'.tif')
  # files_string = " ".join(files_to_mosaic)
  #files_to_mosaic = glob('/home/beland/PycharmProjects/Data_biomass/BIOMASS/Zone03/*.tif')

  vrt = gdal.BuildVRT("/home/beland/PycharmProjects/MyProjectDisturb/results/COMPUTECANADA_trans/merged.vrt", clipmosaic,  VRTNodata= '-99' )
  #gdal.Translate("/home/beland/PycharmProjects/Data_biomass/Mosaic_biomass/30m_NAD83_landsat/MOSAIC_zone3_30M_BIOMASS_.tif", vrt)
  gdal.Translate(RESULT, vrt)

  Image = gdal.Open(RESULT, 1)  # open image in read-write mode
  Band = Image.GetRasterBand(1)

  gdal.SieveFilter(srcBand=Band, maskBand=None, threshold=3, connectedness=8, dstBand=Band)
  del Image, Band
  vrt = None

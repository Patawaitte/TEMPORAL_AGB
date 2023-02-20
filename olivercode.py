'''
Created on 26 oct. 2018

@author: Olivier Matte
'''
import os, sys
import subprocess as sp
import glob

chemin = "C:/Document_/Traitement_Lidar_AGB/"
laz_filepath = chemin +"Donnees_Laz/"

Output_filepath = chemin
num_polygone=["%03d"%i for i in range(300)]

for i in range (0,300) :
    name = glob.glob(laz_filepath+"Polygon_"+num_polygone[i]+"*")

    if name != [] and len(name)>1 :
        nom = "Polygon_"+num_polygone[i]+name[1][57:65]
        
        Merge_Data= chemin+"FUSION/MergeData.exe " + "C:/Document_/Traitement_Lidar_AGB/Donnees_Laz/"+ nom +"*.laz " + Output_filepath +"merge_Laz/"+nom+"_merge.laz"
        print Merge_Data
        sp.call(Merge_Data)

        Filter = chemin+"FUSION/FilterData.exe outlier 1 1 "+ chemin +"filter/"+ nom +"_Filter.laz " + chemin +"merge_Laz/"+nom+"_merge.laz"
        sp.call(Filter)
        
        Ground = chemin+"FUSION/GroundFilter.exe " + chemin +"ground/"+ nom + "_Ground.laz " + "5 " + chemin +"filter/"+ nom + "_Filter.laz"
        Ground_surface =chemin+"FUSION/GridSurfaceCreate.exe "+ chemin +"ground/"+ nom + "_Ground_surface.dtm " + "1 m m 1 " + nom[16:] +" 0 0 " + chemin +"ground/"+ nom + "_Ground.laz"
        
        sp.call(Ground)
        sp.call(Ground_surface)
        
        Canopy = chemin+"FUSION/CanopyModel.exe " + "/median:3 /ground:" + chemin +"ground/"+ nom + "_Ground_surface.dtm " + chemin +"canopy/"+ nom + "_Canopy.dtm " + "1 M M 1 " + nom[16:] +" 0 0 " + chemin +"filter/"+ nom + "_Filter.laz "
        DTM2ASCII = chemin+"FUSION/DTM2ASCII.exe " + chemin + "canopy/"+ nom + "_Canopy.dtm"
        ASCII2TIF = chemin+"GDAL/gdal_translate.exe -of GTiff "  + chemin + "canopy/"+ nom + "_Canopy.asc " + chemin + "canopy/"+ nom + "_Canopy.tif"
        Resolution_5m = chemin+"GDAL/gdalwarp.exe   -tr 5 5 -overwrite -dstnodata 0 "+ chemin + "canopy/"+ nom + "_Canopy.tif " + chemin + "canopy/"+ nom + "_Canopy_5m.tif"
        Resolution_30m = chemin+"GDAL/gdalwarp.exe  -tr 30 30 -overwrite -dstnodata 0 "+ chemin + "canopy/"+ nom + "_Canopy.tif " + chemin + "canopy/"+ nom + "_Canopy_30m.tif"
        
        sp.call(Canopy)
        sp.call(DTM2ASCII)
        sp.call(ASCII2TIF)
        sp.call(Resolution_5m)
        sp.call(Resolution_30m)
        #info = "gdalInfo " + chemin + "canopy/"+ nom + "Canopy_5m.tif"
        #sp.call(info)
        
        
        Cover = chemin+"FUSION/Cover.exe /all" + " " + chemin +"ground/"+ nom + "_Ground_surface.dtm" + " " + chemin +"cover/"+ nom + "_Cover.dtm" + " " + "20 5 M M 1 " + nom[16:] +" 0 0" + " " + chemin +"filter/"+ nom + "_Filter.laz" 
        DTM2ASCII =chemin+"FUSION/DTM2ASCII.exe" + " " + chemin + "cover/"+ nom + "_Cover.dtm"
        ASCII2TIF = chemin+"GDAL/gdal_translate.exe -of GTiff" + " " + chemin + "cover/"+ nom + "_Cover.asc" + " " + chemin + "cover/"+ nom + "_Cover.tif"
        Resolution_5m = chemin +"GDAL/gdalwarp.exe -tr 5 5 -overwrite -dstnodata 0" + " " + chemin + "cover/"+ nom + "_Cover.tif" + " " + chemin +"cover/"+ nom + "_Cover_5m.tif"
        Resolution_30m = chemin +"GDAL/gdalwarp.exe -tr 30 30 -overwrite -dstnodata 0" + " " + chemin + "cover/"+ nom + "_Cover.tif" + " " + chemin +"cover/" + nom + "_Cover_30m.tif"
        
        sp.call(Cover)
        sp.call(DTM2ASCII)
        sp.call(ASCII2TIF)
        sp.call(Resolution_5m)
        sp.call(Resolution_30m)
        
        
        input_file_path_A = chemin +"cover/"+ nom + "_Cover_30m.tif"
        input_file_path_B = chemin +"chm/"+ nom + "_chm.tif"
        output_file_path_vrt = chemin +"cover/"+ nom + "_Cover_Canopy.vrt"
        output_file_path = chemin +"cover/"+ nom + "_Cover_resid.tif"
        calc_expr = '--calc=((A)-(1/(1+(exp(12.3529)*B**(-4.1108)))))/100'
        Build_VRT = chemin+"GDAL/gdalbuildvrt.exe -separate -allow_projection_difference  "+ output_file_path_vrt + " " + input_file_path_A + " " + input_file_path_B
        Resolution_30m = chemin+"GDAL/gdalwarp.exe -tr 30 30 -overwrite" + " " + chemin + "cover/" + nom + "_Cover_resid.tif" + " " + chemin + "cover/"+ nom + "_Cover_resid_30m.tif"
        
        sp.call(Build_VRT)
        sp.call([sys.executable,chemin + 'GDAL/gdal_calc.py', '-A' , output_file_path_vrt, '--A_band=1', '-B' , output_file_path_vrt, '--A_band=2', '--outfile=' + output_file_path, calc_expr , "--overwrite",'--type=Float64',  '--NoDataValue=0'])
        sp.call(Resolution_30m)
        
        
        input_file_path_A = chemin +"cover/"+ nom + "_Cover_resid.tif"
        input_file_path_B = chemin +"chm/"+ nom + "_chm.tif"
        output_file_path_vrt = chemin +"ACD/"+ nom + "_Cover_resid_Canopy.vrt"
        output_file_path = chemin +"ACD/"+ nom + "_ACD_estim.tif"
        calc_expr = '--calc=((0.62369*A**(1.63899))*(1+(1.983*(B)))**(1.081))'
        Build_VRT = chemin+"GDAL/gdalbuildvrt.exe -separate -allow_projection_difference "+ output_file_path_vrt + " " + input_file_path_A + " " + input_file_path_B
        Resolution_30m = chemin+"GDAL/gdalwarp.exe -tr 30 30 -overwrite" + " " + chemin + "ACD/" + nom + "_ACD_estim.tif" + " " + chemin + "ACD/"+ nom + "_ACD_estim_30m.tif"

        sp.call(Build_VRT)
        sp.call([sys.executable, chemin+'GDAL/gdal_calc.py', '-A', output_file_path_vrt, '--A_band=1', '-B', output_file_path_vrt, '--A_band=2', '--outfile='+output_file_path , calc_expr , "--overwrite", '--type=Float64', '--NoDataValue=0'])
        sp.call(Resolution_30m)
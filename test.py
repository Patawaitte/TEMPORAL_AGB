import rasterio
import numpy as np
import rioxarray
from pyproj import Transformer
import xarray as xr
from glob import glob
import pandas as pd
import os
from dask.distributed import Client, LocalCluster, Lock
from dask.utils import SerializableLock
import threading
import time
start_time = time.time()
dffinal = pd.DataFrame()

# prepare different n_wins window offsets to be read
n_wins = 200
col_offs = [int(i) for i in np.linspace(580185.0, 590305.0, n_wins)]
print(col_offs)
row_offs = [int(i) for i in np.linspace( 990185.0, 890285.0, n_wins)]
print(row_offs)
h, w = 256, 256

bounds=(580185.0,990185.0,580305.0, 990285.0)
IMG_PATH="C:/Lidar/tiff/*.tif"
# List of .tiff files
lt_bands = glob(IMG_PATH)
lt_bands.sort() # sorted list
#img = rioxarray.open_rasterio('C:/Lidar/first.tif', masked=True).sel(band=1)
print(lt_bands)
for col_off, row_off in zip(col_offs, row_offs):
    print(slice(row_off, h))
    print(slice(col_off, w))

    for band in lt_bands:
        name = os.path.splitext(os.path.basename(band))[0]
        #print(name)
        #with rioxarray.open_rasterio(band, chunks='auto',  parse_coordinates=False, masked=True, lock=threading.Lock()).rio.clip_box(*bounds) as b:
        #with rioxarray.open_rasterio(band).isel(x=slice(row_off, h), y=slice(col_off, w)).load() as b:
        
        with rioxarray.open_rasterio(band).rolling({band: 3, "center": True}).load() as b:
            #print(b)
            b = b.squeeze().drop("spatial_ref").drop("band")
            b.name = "data"
            res = b.to_dataframe().reset_index()
            #print(res.head(13))
            dffinal['x']=res['x']
            dffinal['y']=res['y']
            dffinal[name]=res['data']
            print(dffinal)

print("Process finished --- %s seconds ---" % (time.time() - start_time))

#https://gis.stackexchange.com/questions/394455/how-to-find-coordinates-of-pixels-of-a-geotiff-image-with-python
#rds = rioxarray.open_rasterio("C:/Lidar/tiff/first.tif")
#rds = rds.squeeze().drop("spatial_ref").drop("band")

#rds.name = "data"
#res = rds.to_dataframe().reset_index()
#print(res.head(13))






#file_name = 'C:/Lidar/first.tif'
#with rasterio.open(file_name) as src:
#    band1 = src.read(1)
#print('Band1 has shape', img.shape)
#height = img.shape[0]
#width = img.shape[1]
#cols, rows = np.meshgrid(np.arange(width), np.arange(height))
#xs, ys = rasterio.transform.xy(img.transform, rows, cols)
#lons= np.array(xs)
#lats = np.array(ys)
#print('lons ', lons)
#print('lats ', lats)

#print('lons shape', lons.shape)


#for iy, ix in np.ndindex(lons.shape):
#    print(lons[iy, ix],  lats[iy, ix],)

#    rds = rioxarray.open_rasterio("C:/Lidar/first.tif")
#    #transformer = Transformer.from_crs("EPSG:6622", rds.rio.crs, always_xy=True)
#    #xx, yy = transformer.transform(lons[iy, ix], lats[iy, ix])

#    # get value from grid
#    value = rds.sel(x=lons[iy, ix], y=lats[iy, ix], method="nearest").values
#    print(value)
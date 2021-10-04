import cv2
import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np

from osgeo import gdal, osr,ogr


def GetExtent(ds):
    """ Return list of corner coordinates from a gdal Dataset """
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    return (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)

def ReprojectCoords(coords,src_srs,tgt_srs):
    """ Reproject a list of x,y coordinates. """
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

###################################################################################


# PlanetScope cropped scene and labels
scene_id = '17027-29735-16'
label_file = 'labels.npz'
planet_img = f'{scene_id}.jpg'

# Read labels
x = np.load(label_file, mmap_mode='r')

# Print all of the tile entries in the NPZ
# print(x.files)
# Print the bounding box coordinates and numerical class ID of the tile's associated annotations
print(x[scene_id])

# Calculate bbox properties
width = x[scene_id][0][2] - x[scene_id][0][0]
height = x[scene_id][0][3] - x[scene_id][0][1]

# Read the cropped Planet scene
ds = gdal.Open(planet_img)
obsdat = ds.ReadAsArray()
obsdat = np.moveaxis(obsdat, 0, -1)

# Overplot bbox's on Panet scene
plt.axes()
plt.imshow(obsdat)
for ind in range(x[scene_id].shape[0]):
    width = x[scene_id][ind][2] - x[scene_id][ind][0]
    height = x[scene_id][ind][3] - x[scene_id][ind][1]
    bbox = plt.Rectangle((x[scene_id][ind][0], x[scene_id][ind][1]), width, height, fc='None', ec='r')
    plt.gca().add_patch(bbox)

plt.title('Planet - Labeled Marine Debris scene')
# plt.savefig('Planet_labeled.png', dpi=400, bbox_inches='tight')
plt.show()

###################################################################################


# Sentinel 2 data
S2_fname ='L1C_T16QED_A005314_20180313T162454.tif'

s2 = gdal.Open(S2_fname)
s2dat = s2.ReadAsArray()
s2dat = np.moveaxis(s2dat, 0, -1)

s2_width = s2.RasterXSize
s2_height = s2.RasterYSize
ext = GetExtent(s2)

src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(s2.GetProjection())
#tgt_srs=osr.SpatialReference()
#tgt_srs.ImportFromEPSG(4326)
tgt_srs = src_srs.CloneGeogCS()

geo_ext = ReprojectCoords(ext, src_srs, tgt_srs)

s2_lt = np.linspace(geo_ext[2][0], geo_ext[0][0], num=s2_height)
s2_ll = np.linspace(geo_ext[0][1], geo_ext[1][1], num=s2_width)
s2_longitude, s2_latitude = np.meshgrid(s2_ll, s2_lt, indexing='ij')
 

###################################################################################

# PlanetScope tiff image
P_fname ='marine_plastic_20180313T154259_1008_3B_Visual.tif'

pf = gdal.Open(P_fname)
pdat = pf.ReadAsArray()
pdat = np.moveaxis(pdat, 0, -1)

width = pf.RasterXSize
height = pf.RasterYSize
gt = pf.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width*gt[4] + height*gt[5] 
maxx = gt[0] + width*gt[1] + height*gt[2]
maxy = gt[3] 

p2_latitude = np.linspace(miny, maxy, num=height)
p2_longitude = np.linspace(minx, maxx, num=width)


# Read geojson to get the bounds of the bbox
df = gpd.read_file('20180313_154259_1008.geojson')

P_fname_sw = 'marine_plastic_20180313T154259_1008_3B_AnalyticMS.tif'
pf_sw = gdal.Open(P_fname)
pdat_sw = pf_sw.ReadAsArray()
pdat_sw = np.moveaxis(pdat_sw, 0, -1)

# Check S2 on the map
f = plt.figure()
plt.imshow(pdat, extent=[p2_longitude.min(), p2_longitude.max(), p2_latitude.min(), p2_latitude.max()])
# plt.imshow(s2dat, extent=[s2_longitude.min(), s2_longitude.max(), s2_latitude.min(), s2_latitude.max()]) 
# for ind in range(np.array(df.geometry).size):
#     bounds = np.array(df.geometry)[ind].bounds
#     width = label_bounds[1] - label_bounds[0]
#     height = label_bounds[3] - label_bounds[1]
#     bbox = plt.Rectangle((label_bounds[0], label_bounds[1]), width, height, fc='None', ec='r')
#     plt.gca().add_patch(bbox) 
# plt.savefig('Planet_geojson_polygons.jpeg', dpi=300) 
plt.show()

label_bounds = np.array(df.geometry)[16].bounds

# Every croped tile is 256x256
label_lons = np.linspace(label_bounds[1], label_bounds[0], 256)
label_lats = np.linspace(label_bounds[3], label_bounds[1], 256)

#Get the corners of bounding box that correpsond to S2
# S2lon_min_id = np.argmin(abs(s2_longitude - label_lons[-1]))
# S2lon_max_id = np.argmax(abs(s2_longitude - label_lons[-1]))
# S2lat_min_id = np.argmin(abs(s2_latitude - label_lats[-1]))
# S2lat_max_id = np.argmax(abs(s2_latitude - label_lats[-1]))

ll_ids = np.where(np.logical_and(s2_longitude > label_bounds[0], s2_longitude < label_bounds[2]))
lt_ids = np.where(np.logical_and(s2_latitude > label_bounds[1], s2_latitude < label_bounds[3]))

coor_ids = np.where(np.logical_and(s2_longitude > label_bounds[0], np.logical_and(s2_longitude < label_bounds[2], np.logical_and(s2_latitude > label_bounds[1],s2_latitude < label_bounds[3]))))


S2_cropped_lon = s2_longitude[coor_ids]
S2_cropped_lat = s2_latitude[coor_ids]

# Check S2 on the map
f = plt.figure()
plt.imshow(s2dat, extent=[s2_longitude.min(), s2_longitude.max(), s2_latitude.min(), s2_latitude.max()]) 
plt.scatter(S2_cropped_lon, S2_cropped_lat, s=8, c='r')   
plt.show()



###################################################################################


# Get the marine debris boundary within a bbox
img = cv2.imread(planet_img)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(img,kernel,iterations = 1)

edges = cv2.Canny(img,100,130)

gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)

kernel = np.ones((7,7),np.uint8)
tophat = cv2.morphologyEx(gradient, cv2.MORPH_TOPHAT, kernel)
plt.axes()
plt.imshow(tophat)

plt.title('Edge Detection')
# plt.savefig('Planet_edge_detection.png', dpi=400, bbox_inches='tight')

plt.show()

###################################################################################


s2Dir = 'S2B_MSIL1C_20180313T161029_N0206_R140_T16QED_20180313T194611.SAFE/GRANULE/L1C_T16QED_A005314_20180313T162454/IMG_DATA/'
s2_files = sorted(glob.glob(s2Dir + '*.jp2'))

 # NDVI = (NIR - RED) / (NIR + RED), where
 # RED is B4, 664.5 nm
 # NIR is B8, 835.1 nm

s2_b4 = gdal.Open(s2_files[3])
s2data_b4 = s2_b4.ReadAsArray() / 10000

s2_b8 = gdal.Open(s2_files[7])
s2data_b8 = s2_b8.ReadAsArray() / 10000

s2_NDVI = (s2data_b8 - s2data_b4) / (s2data_b8 + s2data_b8)

# Read and save RGB channels
ch = [3,2,1]
s2data = np.zeros((10980, 10980, 3))
for i in range(3):
    print(s2_files[ch[i]])
    s2f = gdal.Open(s2_files[ch[i]])
    print(s2f.ReadAsArray().shape)
    print('\n')
    s2data[:,:,i] = s2f.ReadAsArray()

s2_ref = np.clip(s2data/ 10000., 0, 1)

s2_width = s2f.RasterXSize
s2_height = s2f.RasterYSize
ext = GetExtent(s2f)

src_srs = osr.SpatialReference()
src_srs.ImportFromWkt(s2f.GetProjection())
#tgt_srs=osr.SpatialReference()
#tgt_srs.ImportFromEPSG(4326)
tgt_srs = src_srs.CloneGeogCS()

geo_ext = ReprojectCoords(ext, src_srs, tgt_srs)

s2_lt = np.linspace(geo_ext[2][0], geo_ext[0][0], num=s2_height)
s2_ll = np.linspace(geo_ext[0][1], geo_ext[1][1], num=s2_width)
s2_longitude, s2_latitude = np.meshgrid(s2_ll, s2_lt, indexing='ij')

# Check S2 on the map
# s2dat_im = np.moveaxis(s2data, 0, -1)
f = plt.figure()
plt.imshow(s2_ref, extent=[s2_longitude.min(), s2_longitude.max(), s2_latitude.min(), s2_latitude.max()]) 
# plt.scatter(S2_cropped_lon, S2_cropped_lat, s=8, c='r')   
plt.show()

fig = plt.figure(figsize=(9, 7))
axes = plt.axes(projection=ccrs.PlateCarree())

axes.set_extent((-87, -86, 16, 17))

img = axes.contourf(
    s2_longitude,
    s2_latitude,
    s2_NDVI,
    cmap='gist_ncar',
    levels=np.arange(-3, 1,0.1),
    transform=ccrs.PlateCarree(),
)

cbar = plt.colorbar(img, orientation="vertical", pad=0.02, shrink=0.70)
cbar.set_label('NDVI')

add_coast(axes)
axes.set_title("Sentinel 2 NDVI", loc="left")
axes.set_title('2018/03/13', loc="right")
plt.show()
# plt.savefig(
#     f'../figures/{simulation_out}/RSHORT/RSHORT_{simulation_out}_{pd.to_datetime(str(olam_out["time"].values[0])).strftime("%m%d%Y_%H%M%S")}.jpeg',
#     dpi=400,
#     bbox_inches="tight",
# )
plt.close()



ids = np.where(np.logical_and(s2_longitude > p2_longitude.min(), np.logical_and(s2_longitude < p2_longitude.max(), np.logical_and(s2_latitude < p2_latitude.max, s2_latitude > p2_latitude.min()))))














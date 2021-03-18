import geopandas as gpd

g = gpd.read_file('xView_training_china.geojson')

class_ids = [11,12,13,15,19,21,23,24,25,26,28,29,32,40,44,45,47,49,50,51,52,59,60,61,62,63,64,65,79]

g_filt = g[g.type_id.isin(class_ids)]

g_filt.to_file('xView_training_china_select.geojson', driver='GeoJSON')   

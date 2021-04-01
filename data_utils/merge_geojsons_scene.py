import os
import pandas as pd
import geopandas as gpd
import numpy


root_dir = ''

df = pd.read_csv(os.path.join(root_dir, 'filename_sceneID.csv'))

dfs = dict(tuple(df.groupby('Scene_ID')))

dfsl = []

merged_dir = os.path.join(root_dir, 'merged')
if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)

for s in df.Scene_ID.unique().tolist():
    dfsl.append(dfs[s])
    dfss = dfs[s]
    dfss.to_csv(merged_dir, s, '.csv')
    gs = dfss.Filename 
    gsl = []
    for g in gs:
        in_file = os.path.join(root_dir, 'geojsons','marine-debris-geojsons',g[:-8],g)
        geodf = gpd.read_file(in_file)
        gsl.append(geodf)
    mgdf = gpd.GeoDataFrame( pd.concat( gsl, ignore_index=True) )
    #print(mgdf)
    out_file = os.path.join(merged_dir, s, '.geojson')
    mgdf.to_file(out_file, driver="GeoJSON")



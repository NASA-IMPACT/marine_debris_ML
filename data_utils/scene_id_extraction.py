import os
import pandas as pd
import pygeoj as pyj

folder = '/Users/ashah/Desktop/Plastic Pollution/marine-debris-geojsons'
#Create empty dataframe to store filenames and corresponding sceneIDs
dfObj = pd.DataFrame(columns=['Filename', 'Scene_ID'])

#Walking through the folder to extract filename and filepath
for subdir, dirs, files in os.walk(folder):
    for filename in files:
        filepath = subdir + os.sep + filename
       
       #Try and except for extracting the Planet sceneID from the shapefile name
        try: 
            geojson = pyj.load(str(filepath))
            for feature in geojson:
                shapefile_label = feature.properties['label']
                
                scene_id = feature.properties['label'][15:]

            #import ipdb; ipdb.set_trace();
            if 'T' in scene_id:
                    scene_id = scene_id.replace('T', '_')
            dfObj = dfObj.append({'Filename': filename, 'Scene_ID': scene_id}, ignore_index=True)
            print(dfObj)
            scene_id_csv = dfObj.to_csv('filename_sceneID.csv', index=False, header=True)

        except:
            pass
            #print(filepath)
        

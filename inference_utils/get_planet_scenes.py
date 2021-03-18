"""
Script to get all the scene ids from available scenes, aoi and date range for skysat
Author: @developmentseed
Run:

    python3 get_planet_scenes.py --geojson ny.geojson \
            --site ny \
            --start_date 2019,8,1 \
            --end_date 2020,8,19 \
            --cloud_cover 0.10
"""

from planet import api
import os, sys, argparse, json
from datetime import datetime


client = api.ClientV1()

parser = argparse.ArgumentParser()
parser.add_argument('--site', help="name of location to query for, e.g. la, sf, nyc")
parser.add_argument('--geojson', help="geojson of location to query for, e.g. la.geojson")
parser.add_argument('--start_date', help="start date in format of: year, month, day")
parser.add_argument('--end_date', help="start date in format of: year, month, day")
parser.add_argument('--cloud_cover', help='cloud cover in float, e.g. 0.05 for under 5%')

args = parser.parse_args()
print(args)

site = args.site
geojson = args.geojson
start_date = args.start_date
end_date = args.end_date
cloud_cover = args.cloud_cover

with open(geojson, 'r') as geo:
  geojson = json.load(geo)
  geojson = [feat['geometry'] for feat in geojson['features']]

def write_txt(tile_file, out_tex):
    """write tile in format of 'scene_id' to a txt file
    """
    with open(out_tex, 'w') as out:
        for tile in tile_file:
            out.write(tile)
            out.write('\n')

# start_date, end_date:  2019-08-01T00:00:00.000Z 2020-08-19T00:00:00.000Z

year_s, mon_s, day_s = start_date.split(',')
year_e, mon_e, day_e = end_date.split(',')

start_date_ = datetime(year=int(year_s), month=int(mon_s), day=int(day_s))
end_date_ = datetime(year=int(year_e), month=int(mon_e), day=int(day_e))

# build a filter for the AOI
query = api.filters.and_filter(
  api.filters.geom_filter(geojson[0]),
  api.filters.range_filter('cloud_cover', gt=0),
  api.filters.range_filter('cloud_cover', lt=float(cloud_cover)),
  api.filters.date_range("acquired", gte=start_date_, lte=end_date_)
)

# we are requesting SkySatScene imagery
item_types = ['SkySatScene']
request = api.filters.build_search_request(query, item_types)
# this will cause an exception if there are any API related errors
results = client.quick_search(request)

scenes_list=[]

# items_iter returns an iterator over API response pages
for item in results.items_iter(range(len(results.items))):
  # each item is a GeoJSON feature
  #sys.stdout.write('%s\n' % item['id'])
  scenes_list.append(item['id'])

write_txt(scenes_list, site+'_ss.txt')




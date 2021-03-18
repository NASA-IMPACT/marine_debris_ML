"""
Script to get all the tiles from available scenes, aoi and date range for Planetscope or Skysat

Author: @developmentseed

Run:
    python3 get_planet_tiles.py --geojson=supersites.geojson \
            --api_key=xxxxx \
            --collections=PSScene3Band \
            --start_date=2020,1,1 \
            --end_date=2020,1,10 \
            --cloud_cover=0.05 \
            --zoom=16 \
            --out_tex=test.txt
"""
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
import json
import requests
import mercantile
import numpy as np
import argparse
from datetime import date

from planet import api

client = api.ClientV1()


def stats(geometry, collections, start_date, end_date, cc, PL_API_KEY):
    """Retrieve Stats

    ----
    Args:
        collections: ['PSOrthoTile', 'REOrthoTile', 'PSScene3Band', 'PSScene4Band', 'SkySatScene']
        geometry: geojson for the sites
        start_date: "2020-04-01T00:00:00.000Z"
        end_date: same format as start_date
        cc: cloud cover in 0.05 (5%)
    """
    # filter for items the overlap with our chosen geometry
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": geometry
    }

    # filter images acquired in a certain date range
    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte": start_date,
        "lte": end_date
      }
    }

    # filter any images which are more than 50% clouds
    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "lte": cc
      }
    }

    config = {
      "type": "AndFilter",
      "config": [geometry_filter, cloud_cover_filter, date_range_filter]
    }

    # Stats API request object
    stats_endpoint_request = {
      "interval": "day", "item_types": collections, "filter": config
    }

    # build a filter for the AOI
    query = api.filters.and_filter(
      api.filters.geom_filter(geometry),
      api.filters.range_filter('cloud_cover', gt=0),
      api.filters.range_filter('cloud_cover', lt=float(cc)),
      api.filters.date_range("acquired", gte=start_date, lte=end_date)

    )

    # we are requesting <collect type> imagery
    item_types = collections
    request = api.filters.build_search_request(query, item_types)
    # this will cause an exception if there are any API related errors
    results = client.quick_search(request)

    return results.get()


def search(geometry, collections, start_date, end_date, cc, PL_API_KEY
):
    """Search for Data."""
    print('PL_API_KEY: ', PL_API_KEY)
    print('geometry: ', geometry)
    print('collections: ', collections)
    print('start_date, end_date: ', start_date, end_date)
    print('cc: ', cc)


    # filter for items the overlap with our chosen geometry
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": geometry
    }

    # filter images acquired in a certain date range
    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte": start_date,
        "lte":end_date
      }
    }

    # filter any images which are more than 50% clouds
    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "lte": cc
      }
    }

    config = {
      "type": "AndFilter",
      "config": [geometry_filter, cloud_cover_filter, date_range_filter]
    }

    # Stats API request object
    stats_endpoint_request = {
      "interval": "day",
      "item_types": collections,
      "filter": config
    }

    # fire off the POST request #'https://api.planet.com/data/v1/quick-search',
    result = requests.post('https://api.planet.com/data/v1/quick-search', 
               auth=HTTPBasicAuth(PL_API_KEY, ''), json=stats_endpoint_request)

    # build a filter for the AOI
    query = api.filters.and_filter(
      api.filters.geom_filter(geometry),
      api.filters.range_filter('cloud_cover', gt=0),
      api.filters.range_filter('cloud_cover', lt=float(cc)),
      api.filters.date_range("acquired", gte=start_date, lte=end_date)

    )

    # we are requesting <collect type> imagery
    item_types = collections
    request = api.filters.build_search_request(query, item_types)
    # this will cause an exception if there are any API related errors
    results = client.quick_search(request)


    return results.get() 


def get_scene_ids_aoi(site_aois, start_date, end_date, cc, collections, api_key):
    """get scene ids
    """
    results = {feat["properties"]["label"] : stats(feat["geometry"], collections, start_date, end_date, cc, api_key)
           for feat in site_aois["features"]}
    results_PS = {feat["properties"]["label"] : search(feat["geometry"], collections, start_date, end_date, cc, api_key)
                  for feat in site_aois["features"]}

    results_PS = {feat["properties"]["label"] : search(feat["geometry"], collections, start_date, end_date, cc, api_key)
                  for feat in site_aois["features"]}

    aois_scene_ids = [{
        aoi: [[f['id'], f['geometry']['coordinates']]for f in results_PS[aoi]['features']]
    } for aoi in results_PS.keys()]

    print(f"Total of {collections} scenes per sites")
    for k, r in results.items():
        print(k)
        im = r.get("buckets", [])
        total = sum([f["count"] for f in im])
        print(total)
        print("---")

    return aois_scene_ids


def revert_coordinates(coordinates):
    """convert coordinates to bbox
    Args:
        coordinates(list): geometry coordiantes of the aoi
    Return:
        bbox(list): [xmin, ymin, xmax, ymax]
    """
    coordinates = np.asarray(coordinates)
    lats = coordinates[:,:,1]
    lons = coordinates[:,:,0]
    bbox = [lons.min(), lats.min(), lons.max(), lats.max()]
    return bbox

def tile_indices(bbox, ZOOM_LEVEL):
    """get mercantile bounds
    Args:
        bbox(list): [xmin, ymin, xmax, ymax] of the aoi
        ZOOM_LEVEL(int): zoom level, e.g. 16
    Returns:
        tile_bounds (list): tile bounds of AOI

    """
    start_x, start_y, _ = mercantile.tile(bbox[0], bbox[3], ZOOM_LEVEL)
    end_x, end_y, _ = mercantile.tile(bbox[2], bbox[1], ZOOM_LEVEL)
    tile_bounds = [[start_x, end_x], [start_y, end_y]]
    return tile_bounds

def get_tile_xy(geojson, start_date, end_date, cc, collections, ZOOM_LEVEL, api_key):
    """get tile range for pss scenes

    Args:
        geojson (geojson): polygons of the AOIs;
        start_date(iso date): date in iso format, e.g. "2020-04-01T00:00:00.000Z"
        end_date(iso date): date in iso format, e.g. "2020-04-03T00:00:00.000Z"
        cc(float): cloud cover, e.g. 0.05;
        collections(list): planet image products, e.g.['PSOrthoTile','PSScene3Band','SkySatScene']

    """
    scene_tiles_range = []
    aoi_scene_coverage = get_scene_ids_aoi(geojson, start_date, end_date,
                         cc, collections, api_key)
    for aoi in aoi_scene_coverage:
        for item in aoi.values():
            for scene_id, coor in item:
                bbox = revert_coordinates(coor)
                tiles = tile_indices(bbox, ZOOM_LEVEL)
                scene_tiles_range.append([scene_id, tiles])
    return scene_tiles_range

def get_tiles(tile_rangs):
    """get each scene id and the tile x y bounds

    Args:
        tile_rangs(list): save scene id and x y bounds
    ###########tile_range#######
    #[['20200529_003832_100d', [[53910, 53961], [24896, 24921]]]]#

    Returns:
        tile_xyz(list): a list contains scene id, x, y, z.
    ###########################
    """
    tile_xyz = []
    for item in tile_rangs:
        for x_bound in range(item[1][0][0], item[1][0][1] + 1):
            for y_bound in range(item[1][1][0], item[1][1][1]):
                tile_xyz.append(f'{item[0]}-{x_bound}-{y_bound}-16')

    return tile_xyz

def write_txt(tile_file, out_tex):
    """write tile in format of 'pss_scene_id-x-y-z' to a txt file
    """
    with open(out_tex, 'w') as out:
        for tile in tile_file:
            out.write(tile)
            out.write('\n')

def main(geojson, api_key, collections, start_date, end_date, cloud_cover, zoom, out_tex):
    """get all the tiles

    """
    with open(geojson, 'r') as geo:
        geojson = json.load(geo)
    collections = [collections]
    year_s, mon_s, day_s = start_date.split(',')
    year_e, mon_e, day_e = end_date.split(',')
    start_date = f'{date(int(year_s), int(mon_s), int(day_s))}T00:00:00.000Z'
    end_date = f'{date(int(year_e), int(mon_e), int(day_e))}T00:00:00.000Z'
    zoom = int(zoom)
    print("#"*40)
    print(f'start date is {start_date} and end date is {end_date}\n')
    tile_range = get_tile_xy(geojson, start_date, end_date, float(cloud_cover), collections,
                            int(zoom), api_key)
    tiles_pss_aois = get_tiles(tile_range)
    print(f'total output tiles are {len(tiles_pss_aois)}\n')
    write_txt(tiles_pss_aois, out_tex)
    print(f'write all the tiles in {out_tex} at current directory \n')
    print("#"*40)


def parse_arg(args):
    desc = "get_planet_tiles"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--geojson', help="aoi API endpoit in https://")
    parse0.add_argument('--api_key', help="planet api key")
    parse0.add_argument('--collections', help="Planet product as a list")
    parse0.add_argument('--start_date', help="start date in format of: year, month, day")
    parse0.add_argument('--end_date', help="start date in format of: year, month, day")
    parse0.add_argument('--cloud_cover', help='cloud cover in float, e.g. 0.05 for under 5%')
    parse0.add_argument('--zoom', help='OSM zoom level, e.g. 16')
    parse0.add_argument('--out_tex', help='txt name to save all the output tiles')
    return vars(parse0.parse_args(args))


def cli():
    args = parse_arg(sys.argv[1:])
    main(**args)

if __name__ == "__main__":
    cli()


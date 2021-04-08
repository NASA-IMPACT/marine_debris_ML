from planet import api
import os

client = api.ClientV1(api_key=os.getenv("PL_API_KEY"))

import json

def p(data):
    print(json.dumps(data, indent=2))

LA_geom = {
  "type": "Polygon",
  "coordinates": [
    [
      [-117.66703694,33.42673544],
      [-117.07333302,34.14299552],
      [-117.80010186,34.30197535],
      [-118.67592739,34.34392384],
      [-118.68741566,33.73867555],
      [-117.66703694,33.42673544]
    ]
  ]
}

Beijing_geom = {
  "type": "Polygon",
  "coordinates": [
    [
      [115.84,39.62],
      [116.85,39.62],
      [116.85,40.22],
      [115.84,40.22],
      [115.84,39.62]
    ]
  ]
}

Ghent_geom = {
  "type": "Polygon",
  "coordinates": [
    [
      [3.64539683,51.09482029],
      [3.66610478,51.07164212],
      [3.74584324,51.06663625],
      [3.79612713,51.11582801],
      [3.84588693,51.17990464],
      [3.82746305,51.21762622],
      [3.85833337,51.28454634],
      [3.81774134,51.28873095],
      [3.7221739,51.12261565],
      [3.64539683,51.09482029]
    ]
  ]
}

Dunkirk_geom = {
  "type": "Polygon",
  "coordinates": [
    [
      [2.08355962,51.03423481],
      [2.14826632,50.96553938],
      [2.41646888,51.02097784],
      [2.38289168,51.07488218],
      [2.32298564,51.08773119],
      [2.15844656,51.05891125],
      [2.08355962,51.03423481]
    ]
  ]
}

NewYork_geom = {
  "type": "Polygon",
  "coordinates": [
    [
      [-71.74516,41.54467],
      [-74.43395,41.54943],
      [-74.43219,40.47812],
      [-71.74516,40.48343],
      [-71.74516,41.54467]
    ]
  ]
}

SF_geom = {
  "type": "Polygon",
  "coordinates": [
    [
      [-122.63570045,38.31172386],
      [-122.53518996,37.11988178],
      [-121.53519174,37.17901736],
      [-121.64821141,38.35512939],
      [-122.63570045,38.31172386]
    ]
  ]
}

Tokyo_geom = {
  "type": "Polygon",
  "coordinates": [
    [
      [139.37,35.33],
      [140.19,35.33],
      [140.19,35.85],
      [139.37,35.85],
      [139.37,35.33]
    ]
  ]
}

from planet.api import filters
from datetime import datetime

start_date_01 = datetime(year=2020, month=1, day=1)
end_date_01 = datetime(year=2020, month=1, day=31)
start_date_02 = datetime(year=2020, month=2, day=1)
end_date_02 = datetime(year=2020, month=2, day=29)
start_date_03 = datetime(year=2020, month=3, day=1)
end_date_03 = datetime(year=2020, month=3, day=31)
start_date_04 = datetime(year=2020, month=4, day=1)
end_date_04 = datetime(year=2020, month=4, day=30)
start_date_05 = datetime(year=2020, month=5, day=1)
end_date_05 = datetime(year=2020, month=5, day=31)


aois = ['LA', 'Beijing', 'Ghent', 'Dunkirk', 'NewYork', 'SF', 'Tokyo']
geoms = [LA_geom, Beijing_geom, Ghent_geom, Dunkirk_geom, NewYork_geom, SF_geom, Tokyo_geom]
months = ['01', '02', '03', '04', '05']
seds = [[start_date_01, end_date_01],[start_date_02, end_date_02], [start_date_03, end_date_03], [start_date_04, end_date_04], [start_date_05, end_date_05]]
da = dict(zip(aois, geoms))
dd = dict(zip(months, seds))

for aoi, g in da.items():
    for month, se in dd.items():
        date_filter = filters.date_range('acquired', gte=se[0], lte=se[1])
        cloud_filter = filters.range_filter('cloud_cover', lte=0.05)

        geometry_filter = api.filters.geom_filter(g) #, "geometry")


        and_filter = filters.and_filter(date_filter, cloud_filter, geometry_filter) #, area_filter)

        p(and_filter)

        item_types = ["PSScene3Band"]
        req = filters.build_search_request(and_filter, item_types)

        p(req)

        res = client.quick_search(req)

        for item in res.items_iter(4):
            print(item['id'], item['properties']['item_type'])

        with open('results_{}_supersite_{}.json'.format(aoi, month),'w') as f:
            res.json_encode(f,10000)

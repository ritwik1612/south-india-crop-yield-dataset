# Preserved raw sources

| File | Source | Contains |
| --- | --- | --- |
| government/crop_statistics.csv | Government crop-statistics distribution originally named APY.csv | Crop, state, district, crop year, season, area and production |
| boundaries/india_districts.geojson.zip | GADM 4.1 administrative level 2 | District boundary geometry and names |
| weather/daily_weather.csv | Assembled NASA POWER Daily API extract | Daily weather at matched district coordinates |

These are three source families, not five independent datasets.
The former separate yield/area/fraction tables are temporary join products;
the new builder does not keep them as user-facing datasets.

Source descriptions and license terms:
- [Government district-season crop statistics](https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997)
- [GADM](https://gadm.org/data.html) and [license](https://gadm.org/license.html)
- [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)

Keep files read-only. The weather extract includes derived vapour pressure;
ET0 maps to NASA EVLAND, an evapotranspiration proxy. This folder's files retain
the exact contents of their previous project copies, with clearer names.

NASA's RE-community radiation units were verified from API metadata:
RAD is kWh/m2/day, and RAD_TOTAL is kWh/m2. EVLAND is land evaporation
(mm/day); its ET0 column name is retained for schema correspondence.
See ../../docs/preprocessing/weather_units.json for the recorded metadata.

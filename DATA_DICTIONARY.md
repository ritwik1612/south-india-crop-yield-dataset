# Data dictionary — South India Four-State Crop Yield Dataset v1

## Geographic and target convention

`IDREGION` is a deterministic `IND_<STATE>_<DISTRICT>` key. `FYEAR` is the
calendar crop year from the Government of India source. Area is hectares;
production is tonnes; yield is tonnes/hectare. The primary first-stage target
is rice yield, with `SEASON` retained to avoid mixing Kharif, Rabi and other
crop cycles.

## File contents

| File | Rows | Purpose |
| --- | ---: | --- |
| INDIA_CROP_STATS_MASTER.csv | 66,172 | All valid reported crop records for the four states, 1997–2023. |
| YIELD_NUTS2_SOUTH_INDIA_RICE.csv | 3,554 | Rice target records; `YIELD = PRODUCTION_TONNES / CROP_AREA_HA`. |
| CROP_AREA_NUTS2_SOUTH_INDIA_RICE.csv | 3,554 | Rice harvested/cultivated area interface. |
| AREA_FRACTIONS_NUTS2_SOUTH_INDIA_RICE.csv | 3,554 | District share of state-season rice area. |
| CENTROIDS_NUTS2_SOUTH_INDIA.csv | 70 | Verified current GADM district centroid crosswalk. |
| METEO_DAILY_NUTS2_SOUTH_INDIA.csv | 588,000 | 70 centroids × 8,400 days, 2001–2023. |
| METEO_DEKADS_NUTS2_SOUTH_INDIA.csv | 57,960 | Ten-day aggregation of the daily weather file. |

## Meteorological fields

`TMAX`, `TMIN`, `TAVG` are degrees C; `WSPD` metres/second; `PREC` millimetres
per day (or sum for dekads); `RAD` kWh/m²/day; `RELH` percent; `PS` kPa.
`VPRES` is derived vapour pressure in hPa from temperature and relative
humidity. `ET0` is retained for Netherlands-schema compatibility but contains
NASA POWER `EVLAND` evapotranspiration (mm/day) and must be treated as a proxy,
not a locally calibrated FAO-56 reference ET.

## Required joins and modelling cohort

Join targets to weather through `IDREGION`. The source preserves every valid
record from all four states. Only 2,383 of the 3,554 rice label rows currently
have an exact or documented historical district-to-GADM centroid crosswalk.
Use that 2,383-row cohort for the first model. Do not drop the remaining rows:
the unmatched file is the audit trail for completing the district crosswalk.

## Provenance

- Crop labels: Ministry of Agriculture & Farmers Welfare, [district-wise,
  season-wise crop production statistics](https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997).
- Boundaries: [GADM 4.1](https://gadm.org/data.html), administrative level 2.
- Weather: [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), queried at district centroids.

## Not present yet

MODIS FAPAR/NDVI, SoilGrids/SRTM aggregates, and WOFOST runs are deliberately
not represented by fabricated columns. They will be added as separate tables
after the district crosswalk is reviewed.

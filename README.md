# South India Four-State Crop Yield Dataset v1

## What is included

This package contains 66,172 source-backed district/crop/season/year records
for Andhra Pradesh, Karnataka, Tamil Nadu and Telangana, and 3,554 rice
yield records. Crop yield is calculated as reported production (tonnes) divided by
reported area (hectares), expressed as tonnes per hectare.

## Source and licence

Primary labels are from the Ministry of Agriculture & Farmers Welfare's district-wise,
season-wise crop production statistics, downloaded in this project as the APY.csv
distribution. Administrative geometry is GADM 4.1. Preserve both source citations in
any report. The source master is retained unchanged in meaning, with whitespace
normalised and invalid zero-area records excluded.

## Netherlands-schema correspondence

| Netherlands interface | South India v1 counterpart | Status |
| --- | --- | --- |
| YIELD_NUTS2 | YIELD_NUTS2_SOUTH_INDIA_RICE.csv | included |
| CROP_AREA_NUTS2 | CROP_AREA_NUTS2_SOUTH_INDIA_RICE.csv | included |
| AREA_FRACTIONS_NUTS2 | AREA_FRACTIONS_NUTS2_SOUTH_INDIA_RICE.csv | included |
| CENTROIDS_NUTS2 | CENTROIDS_NUTS2_SOUTH_INDIA.csv | included where GADM match is exact |
| METEO_DAILY / METEO_DEKADS | NASA POWER reanalysis at district centroids | included for 70 matched districts |
| REMOTE_SENSING (FAPAR) | MODIS FAPAR district aggregation | pending download |
| SOIL / GAES | SoilGrids plus SRTM district aggregation | pending download |
| WOFOST | WOFOST crop-model simulations | generated later, not publicly precomputed |

## Important limitations

This is an actual label/area plus weather dataset, not a completed multimodal feature set. Do not
impute or invent weather, FAPAR, soil, elevation, or WOFOST values. Historical
district names differ from current boundaries; use UNMATCHED_DISTRICTS_REQUIRING_REVIEW.csv
to create a documented crosswalk before spatial feature extraction.

## Weather file, when present

`METEO_DAILY_NUTS2_SOUTH_INDIA.csv` contains NASA POWER reanalysis values at matched
district centroids for 2001-01-01 to 2023-12-31. It keeps the Netherlands daily
column names. VPRES is derived vapour pressure (hPa) from TAVG and RELH. ET0 is a
NASA POWER `EVLAND` evapotranspiration proxy (mm/day), not a claim that it is a
locally calibrated FAO-56 reference ET. PS is additionally retained as surface
pressure (kPa). Rows exist only for the documented centroid-matched cohort.

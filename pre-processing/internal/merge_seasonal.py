"""Create one source-backed master training table from the project CSV inputs.

The output has one row per verified district, rice crop year and season. It does
not fabricate observations: it joins labels, crop area, area share, centroids
and season-specific aggregates of real NASA POWER daily weather.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import date
from pathlib import Path


SEASON_MONTHS = {
    "Kharif": {6, 7, 8, 9, 10, 11},
    "Rabi": {1, 2, 3, 4, 11, 12},
    "Summer": {3, 4, 5, 6},
    "Autumn": {9, 10, 11, 12},
    "Winter": {12, 1, 2, 3},
    "Whole Year": set(range(1, 13)),
}
WEATHER_FIELDS = ("TMAX", "TMIN", "TAVG", "VPRES", "WSPD", "PREC", "ET0", "RAD", "RELH", "PS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the one-table South India master training dataset.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-weather-days", type=int, default=60)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def numeric(value: str | None) -> float:
    try:
        return float(value or "")
    except ValueError:
        return float("nan")


def main() -> None:
    args = parse_args()
    source = args.data_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_csv(source / "YIELD_NUTS2_SOUTH_INDIA_RICE.csv")
    centroids = {row["IDREGION"]: row for row in read_csv(source / "CENTROIDS_NUTS2_SOUTH_INDIA.csv")}
    areas = {
        (row["IDREGION"], row["FYEAR"], row["SEASON"]): row
        for row in read_csv(source / "CROP_AREA_NUTS2_SOUTH_INDIA_RICE.csv")
    }
    fractions = {
        (row["IDREGION"], row["FYEAR"], row["SEASON"]): row
        for row in read_csv(source / "AREA_FRACTIONS_NUTS2_SOUTH_INDIA_RICE.csv")
    }

    records: dict[tuple[str, int, str], dict[str, str]] = {}
    month_lookup: dict[tuple[str, int, int], list[tuple[str, int, str]]] = defaultdict(list)
    for label in labels:
        try:
            key = (label["IDREGION"], int(label["FYEAR"]), label["SEASON"])
        except (KeyError, ValueError):
            continue
        if key[0] not in centroids or key[2] not in SEASON_MONTHS or not math.isfinite(numeric(label.get("YIELD"))):
            continue
        records[key] = label
        for month in SEASON_MONTHS[key[2]]:
            weather_year = key[1] - 1 if key[2] in {"Rabi", "Winter"} and month in {11, 12} else key[1]
            month_lookup[(key[0], weather_year, month)].append(key)

    aggregation: dict[tuple[str, int, str], dict[str, float]] = {}
    for key in records:
        aggregation[key] = {"count": 0.0, "gdd": 0.0, **{field: 0.0 for field in WEATHER_FIELDS}}

    with (source / "METEO_DAILY_NUTS2_SOUTH_INDIA.csv").open(newline="", encoding="utf-8-sig") as handle:
        for weather in csv.DictReader(handle):
            try:
                observation_date = date.fromisoformat(weather["DATE"])
            except (KeyError, ValueError):
                continue
            matching_keys = month_lookup.get((weather.get("IDREGION", ""), observation_date.year, observation_date.month), [])
            if not matching_keys:
                continue
            values = {field: numeric(weather.get(field)) for field in WEATHER_FIELDS}
            if not all(math.isfinite(value) for value in values.values()):
                continue
            for key in matching_keys:
                target = aggregation[key]
                target["count"] += 1
                target["gdd"] += max(values["TAVG"] - 10.0, 0.0)
                for field, value in values.items():
                    target[field] += value

    output_rows: list[dict[str, object]] = []
    for key, label in records.items():
        weather = aggregation[key]
        if weather["count"] < args.minimum_weather_days:
            continue
        region, year, season = key
        centroid = centroids[region]
        area = areas.get((region, str(year), season), {})
        fraction = fractions.get((region, str(year), season), {})
        count = weather["count"]
        tmax_mean, tmin_mean, tavg_mean = (weather[name] / count for name in ("TMAX", "TMIN", "TAVG"))
        prec_total, et0_total, rad_total = (weather[name] for name in ("PREC", "ET0", "RAD"))
        row: dict[str, object] = {
            "CROP": label["CROP"], "IDREGION": region, "STATE": centroid["STATE"], "DISTRICT": centroid["DISTRICT"],
            "FYEAR": year, "SEASON": season, "YIELD": numeric(label["YIELD"]),
            "PRODUCTION_TONNES": numeric(label["PRODUCTION_TONNES"]), "CROP_AREA": numeric(area.get("CROP_AREA")),
            "AREA_FRACTION": numeric(fraction.get("FRACTION")), "CENTROID_X": numeric(centroid["CENTROID_X"]),
            "CENTROID_Y": numeric(centroid["CENTROID_Y"]), "MATCH_STATUS": centroid["MATCH_STATUS"],
            "WEATHER_DAY_COUNT": int(count), "TMAX_MEAN": tmax_mean, "TMIN_MEAN": tmin_mean, "TAVG_MEAN": tavg_mean,
            "VPRES_MEAN": weather["VPRES"] / count, "WSPD_MEAN": weather["WSPD"] / count,
            "PREC_TOTAL": prec_total, "ET0_TOTAL": et0_total, "RAD_TOTAL": rad_total,
            "RELH_MEAN": weather["RELH"] / count, "PS_MEAN": weather["PS"] / count, "GDD_BASE10": weather["gdd"],
            # Source-backed engineered parameters; no values are synthesized.
            "TEMP_RANGE_MEAN": tmax_mean - tmin_mean,
            "WATER_BALANCE_PROXY": prec_total - et0_total,
            "RAIN_PER_GDD": prec_total / weather["gdd"] if weather["gdd"] else 0.0,
            "RADIATION_PER_GDD": rad_total / weather["gdd"] if weather["gdd"] else 0.0,
            "CROP_AREA_LOG1P": math.log1p(max(numeric(area.get("CROP_AREA")), 0.0)),
        }
        output_rows.append(row)

    output_rows.sort(key=lambda row: (int(row["FYEAR"]), str(row["STATE"]), str(row["DISTRICT"]), str(row["SEASON"])))
    destination = args.output_dir / "south_india_rice_master_training.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Created {destination} with {len(output_rows):,} rows and {len(output_rows[0]):,} columns.")


if __name__ == "__main__":
    main()


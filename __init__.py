"""species_verifier.core

Core plausibility-check logic.

What it does
------------
Given:
  - claimed_species: a scientific name string
  - boundary_geojson: a GeoJSON Polygon or Feature (EPSG:4326 lon/lat)

It returns a JSON-serializable report that *assesses plausibility* of the claim using:
  1) GBIF taxonomy match + occurrence search in/near the polygon
  2) ESA WorldCover land cover proportions in the polygon bbox
  3) Sentinel-2 NDVI summary in the polygon bbox

This is not definitive species identification; it is evidence-based plausibility scoring.

Dependencies
------------
- Always required: requests, shapely, pyproj
- For satellite steps: numpy, pystac-client, planetary-computer, odc-stac, xarray, rasterio, rioxarray

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from shapely.geometry import Polygon, shape

GBIF_BASE = "https://api.gbif.org/v1"
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass
class PlausibilityWeights:
    gbif_in_polygon: float = 0.45
    habitat_tree_cover: float = 0.25
    ndvi_signal: float = 0.30


# -----------------------------
# Geo helpers
# -----------------------------

def _extract_geometry(boundary_geojson: Dict[str, Any]) -> Dict[str, Any]:
    if boundary_geojson.get("type") == "Feature":
        return boundary_geojson["geometry"]
    return boundary_geojson


def _load_polygon_from_geojson(boundary_geojson: Dict[str, Any]) -> Polygon:
    geom = _extract_geometry(boundary_geojson)
    geom_shape = shape(geom)

    if geom_shape.geom_type == "Polygon":
        return geom_shape
    if geom_shape.geom_type == "MultiPolygon":
        # choose largest polygon by area (simple, predictable)
        return max(list(geom_shape.geoms), key=lambda g: g.area)

    raise ValueError(f"Unsupported geometry type: {geom_shape.geom_type}")


def _polygon_to_wkt_lonlat(poly: Polygon) -> str:
    # GeoJSON is assumed to be EPSG:4326 lon/lat
    return poly.wkt


# -----------------------------
# GBIF helpers
# -----------------------------

def _gbif_species_match(scientific_name: str) -> Dict[str, Any]:
    url = f"{GBIF_BASE}/species/match"
    params = {"name": scientific_name, "strict": "false"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _gbif_occurrence_search(params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GBIF_BASE}/occurrence/search"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _gbif_occurrences_in_polygon(taxon_key: int, polygon_wkt: str, max_records: int = 900) -> Tuple[int, list]:
    """Return (total_count, collected_results_up_to_max_records).

    GBIF occurrence search is paged; max limit per request is 300.
    """

    per_page = 300
    max_records = max(per_page, int(max_records))

    collected = []
    offset = 0
    total_count = None

    while offset < max_records:
        resp = _gbif_occurrence_search(
            {
                "taxonKey": taxon_key,
                "geometry": polygon_wkt,
                "hasCoordinate": "true",
                "limit": per_page,
                "offset": offset,
            }
        )

        if total_count is None:
            total_count = int(resp.get("count", 0))

        results = resp.get("results", [])
        if not results:
            break

        collected.extend(results)
        offset += per_page

        # stop if we've already fetched all results
        if offset >= total_count:
            break

    return int(total_count or 0), collected[:max_records]


def _gbif_occurrences_nearby_bbox(taxon_key: int, centroid_lon: float, centroid_lat: float, radius_km: float = 50.0, max_records: int = 900) -> Tuple[int, list]:
    # crude bbox around centroid
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.1, math.cos(math.radians(centroid_lat))))
    minx, miny = centroid_lon - dlon, centroid_lat - dlat
    maxx, maxy = centroid_lon + dlon, centroid_lat + dlat

    bbox_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]).wkt
    return _gbif_occurrences_in_polygon(taxon_key, bbox_poly, max_records=max_records)


# -----------------------------
# Satellite helpers (Planetary Computer)
# -----------------------------

def _satellite_available() -> bool:
    try:
        import numpy  # noqa: F401
        import pystac_client  # noqa: F401
        import planetary_computer  # noqa: F401
        from odc.stac import stac_load  # noqa: F401
        return True
    except Exception:
        return False


def _pc_search_best_item(collection: str, poly: Polygon, datetime: str, extra_query: Optional[Dict[str, Any]] = None):
    import pystac_client
    import planetary_computer as pc

    catalog = pystac_client.Client.open(PC_STAC_URL)
    bbox = poly.bounds

    kwargs: Dict[str, Any] = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": datetime,
    }
    if extra_query:
        kwargs["query"] = extra_query

    items = list(catalog.search(**kwargs).items())
    if not items:
        return None

    # sign assets so URLs are accessible
    items = [pc.sign(i) for i in items]

    def cloud_key(it):
        return it.properties.get("eo:cloud_cover", 999.0)

    return sorted(items, key=cloud_key)[0]


def _pc_worldcover_class_proportions(poly: Polygon, year: int = 2021) -> Optional[Dict[str, Any]]:
    import numpy as np
    from odc.stac import stac_load

    dt = f"{year}-01-01/{year}-12-31"
    item = _pc_search_best_item("esa-worldcover", poly, dt)
    if item is None:
        return None

    ds = stac_load(
        [item],
        bands=["map"],
        bbox=poly.bounds,
        crs="EPSG:4326",
        chunks={},
    )

    arr = ds["map"].isel(time=0).values
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None

    uniq, counts = np.unique(arr.astype("int32"), return_counts=True)
    total = counts.sum()

    return {
        "year": year,
        "class_counts": {int(k): int(v) for k, v in zip(uniq, counts)},
        "class_proportions": {int(k): float(v) / float(total) for k, v in zip(uniq, counts)},
    }


def _pc_sentinel2_ndvi_stats(poly: Polygon) -> Optional[Dict[str, Any]]:
    import numpy as np
    from odc.stac import stac_load

    # Broad window; you can tighten this or make it user-configurable
    dt = "2024-01-01/2026-12-31"
    item = _pc_search_best_item("sentinel-2-l2a", poly, dt)
    if item is None:
        return None

    cloud = item.properties.get("eo:cloud_cover")

    ds = stac_load(
        [item],
        bands=["B04", "B08"],
        bbox=poly.bounds,
        crs="EPSG:4326",
        resolution=10,
        chunks={},
        dtype="uint16",
        nodata=0,
    )

    red = ds["B04"].isel(time=0).astype("float32").values / 10000.0
    nir = ds["B08"].isel(time=0).astype("float32").values / 10000.0

    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, np.nan)

    ndvi = ndvi[~np.isnan(ndvi)]
    if ndvi.size == 0:
        return None

    return {
        "item_datetime": item.datetime.isoformat() if item.datetime else None,
        "cloud_cover": cloud,
        "ndvi_mean": float(np.mean(ndvi)),
        "ndvi_median": float(np.median(ndvi)),
        "vegetated_fraction_ndvi_gt_0_4": float(np.mean(ndvi > 0.4)),
    }


# -----------------------------
# Scoring
# -----------------------------

def _score_plausibility(
    gbif_count_in_poly: int,
    worldcover: Optional[Dict[str, Any]],
    ndvi: Optional[Dict[str, Any]],
    weights: PlausibilityWeights = PlausibilityWeights(),
) -> Tuple[float, Dict[str, float]]:
    # 1) GBIF evidence, log-scaled
    gbif_score = min(1.0, math.log1p(gbif_count_in_poly) / math.log1p(50))

    # 2) Habitat signal from WorldCover (generic: more dominance of a single class => higher certainty)
    habitat_score = 0.5
    if worldcover and worldcover.get("class_proportions"):
        props = worldcover["class_proportions"]
        habitat_score = min(1.0, 0.3 + max(props.values()))

    # 3) NDVI signal
    ndvi_score = 0.5
    if ndvi:
        veg = float(ndvi.get("vegetated_fraction_ndvi_gt_0_4", 0.0))
        med = float(ndvi.get("ndvi_median", 0.0))
        ndvi_score = min(1.0, 0.5 * veg + 0.5 * max(0.0, min(1.0, (med - 0.1) / 0.6)))

    overall = (
        weights.gbif_in_polygon * gbif_score
        + weights.habitat_tree_cover * habitat_score
        + weights.ndvi_signal * ndvi_score
    )

    return overall, {
        "gbif_score": gbif_score,
        "habitat_score": habitat_score,
        "ndvi_score": ndvi_score,
    }


def _verdict(score: float) -> str:
    if score >= 0.70:
        return "plausible"
    if score >= 0.40:
        return "uncertain"
    return "implausible"


# -----------------------------
# Public API
# -----------------------------

def run_plausibility_check(
    claimed_species: str,
    boundary_geojson: Dict[str, Any],
    skip_satellite: bool = False,
    worldcover_year: int = 2021,
    nearby_radius_km: float = 50.0,
    gbif_max_records: int = 900,
) -> Dict[str, Any]:
    """Run plausibility check and return report dict."""

    poly = _load_polygon_from_geojson(boundary_geojson)
    centroid = poly.centroid
    polygon_wkt = _polygon_to_wkt_lonlat(poly)

    # 1) GBIF taxonomy normalisation
    sp_match = _gbif_species_match(claimed_species)
    taxon_key = sp_match.get("usageKey") or sp_match.get("speciesKey")
    if taxon_key is None:
        raise RuntimeError(f"GBIF could not resolve species: {claimed_species}. Match response: {sp_match}")

    taxon_key = int(taxon_key)

    # 2) GBIF occurrences
    in_count, in_results = _gbif_occurrences_in_polygon(taxon_key, polygon_wkt, max_records=gbif_max_records)
    near_count, _ = _gbif_occurrences_nearby_bbox(
        taxon_key,
        centroid.x,
        centroid.y,
        radius_km=nearby_radius_km,
        max_records=min(gbif_max_records, 900),
    )

    # 3) Satellite
    worldcover = None
    ndvi = None
    sat_errors = []

    if not skip_satellite:
        if not _satellite_available():
            sat_errors.append(
                "Satellite dependencies not installed. Install extras listed in requirements.txt, or tick 'Skip satellite steps'."
            )
        else:
            try:
                worldcover = _pc_worldcover_class_proportions(poly, year=int(worldcover_year))
            except Exception as e:
                sat_errors.append(f"WorldCover error: {e}")

            try:
                ndvi = _pc_sentinel2_ndvi_stats(poly)
            except Exception as e:
                sat_errors.append(f"Sentinel-2 NDVI error: {e}")

    # 4) Score
    score, components = _score_plausibility(in_count, worldcover, ndvi)

    return {
        "inputs": {
            "claimed_species": claimed_species,
            "boundary_bbox": list(poly.bounds),
            "boundary_centroid_lonlat": [centroid.x, centroid.y],
        },
        "gbif": {
            "species_match": sp_match,
            "taxon_key": taxon_key,
            "occurrences_in_polygon_count": int(in_count),
            "occurrences_nearby_count": int(near_count),
            "occurrence_sample_in_polygon": in_results[:25],
        },
        "satellite": {
            "worldcover": worldcover,
            "sentinel2_ndvi": ndvi,
            "errors": sat_errors,
        },
        "plausibility": {
            "score_0_to_1": float(score),
            "component_scores": components,
            "verdict": _verdict(score),
            "interpretation": (
                "Plausibility assessment based on public occurrence + habitat/vegetation signals. "
                "Not a definitive species identification for an individual tree without ground truth or lab data."
            ),
        },
    }

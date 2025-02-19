# Copyright (c) Meta Platforms, Inc. and affiliates.

import json, math, urllib3
from http.client import responses
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd

import mercantile
import pandas as pd
from mapbox_vector_tile import decode

from shapely.geometry import shape

from .. import logger
from ..utils.geo import BoundaryBox

OSM_URL = "https://api.openstreetmap.org/api/0.6/map.json"

FB_TILE_SERVER_URL = (
    # "https://www.internalfb.com/intern/maps/vtp/s1/20250217080099/{z}/{x}/{y}/"
    "https://external.xx.fbcdn.net/maps/vtp/s1/77/{z}/{x}/{y}/?locale=en_US"
)


def get_osm(
    boundary_box: BoundaryBox,
    cache_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if not overwrite and cache_path is not None and cache_path.is_file():
        return json.loads(cache_path.read_text())

    (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_
    query = {"bbox": f"{left},{bottom},{right},{top}"}

    logger.info("Calling the OpenStreetMap API...")
    result = urllib3.request("GET", OSM_URL, fields=query, timeout=10)
    if result.status != 200:
        error = result.info()["error"]
        raise ValueError(f"{result.status} {responses[result.status]}: {error}")

    if cache_path is not None:
        cache_path.write_bytes(result.data)
    return result.json()


# https://gis.stackexchange.com/questions/401541/decoding-mapbox-vector-tiles/460173#460173
def pixel2deg(xtile, ytile, zoom, xpixel, ypixel, extent=4096):
    xtile = xtile + (xpixel / extent)
    ytile = ytile + ((extent - ypixel) / extent)
    lon_deg = (xtile / 2**zoom) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / 2**zoom)))
    lat_deg = math.degrees(lat_rad)
    return (lon_deg, lat_deg)


def vector_tiles_to_geodataframe(
    bbox: BoundaryBox,
    zoom: int = 16,
):
    (south, west), (north, east) = bbox.min_, bbox.max_
    tiles = mercantile.tiles(west, south, east, north, zooms=[zoom])

    features = []
    _columns = ["group", "label", "geometry"]

    for t in tiles:
        result = urllib3.request(
            "GET",
            FB_TILE_SERVER_URL.format(x=t.x, y=t.y, z=zoom),
            fields={},
            timeout=10,
        )

        if result.status != 200:
            error = result.info()["error"]
            raise ValueError(f"{result.status} {responses[result.status]}: {error}")

        decoded_tile = decode(
            tile=result.data,
            default_options={
                "transformer": lambda x, y: pixel2deg(t.x, t.y, zoom, x, y)
            },
        )

        # Clip Dataframe and add relevant columns based on properties
        if decoded_tile.get("building"):
            buildings = gpd.GeoDataFrame(
                [
                    f.get("properties")
                    for f in decoded_tile.get("building").get("features")
                ],
                geometry=[
                    shape(f.get("geometry"))
                    for f in decoded_tile.get("building").get("features")
                ],
            )

            if "isDetail" in buildings.columns:
                buildings = buildings[buildings.isDetail != "true"]

            buildings = buildings.clip(mask=[west, south, east, north])
            buildings["group"] = "building"
            buildings["label"] = "none"

            features.append(buildings[pd.notnull(buildings.group)][_columns])

        if decoded_tile.get("road"):
            roads = gpd.GeoDataFrame(
                [f.get("properties") for f in decoded_tile.get("road").get("features")],
                geometry=[
                    shape(f.get("geometry"))
                    for f in decoded_tile.get("road").get("features")
                ],
            )

            roads = roads.clip(mask=[west, south, east, north])
            roads["group"] = roads["class"].apply(
                lambda _cls: "path" if _cls in {"pedestrian"} else "road"
            )
            roads["label"] = roads["class"]

            features.append(roads[pd.notnull(roads.group)][_columns])

        if decoded_tile.get("landuse"):
            landuse = gpd.GeoDataFrame(
                [
                    f.get("properties")
                    for f in decoded_tile.get("landuse").get("features")
                ],
                geometry=[
                    shape(f.get("geometry"))
                    for f in decoded_tile.get("landuse").get("features")
                ],
            )

            landuse = landuse.clip(mask=[west, south, east, north])
            landuse["group"] = landuse["class"].apply(
                lambda _cls: "grass" if _cls in {"greenspace"} else None
            )
            landuse["label"] = landuse["class"]

            features.append(landuse[pd.notnull(landuse.group)][_columns])

    return pd.concat(features)

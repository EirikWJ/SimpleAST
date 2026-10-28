import sys

import geopandas as gpd
from shapely.geometry import box

from sim.marast.marastPath import REPO_ROOT

# Writes a map of open water with no obstacles, in the same format as MAR-AST's maps.
# Run from the repo root:  uv run python -m sim.marast.makeOpenWaterMap [size] [border_gap]
#
# The open water is the square 0 <= north, east <= size [m]. MAR-AST takes the map's edge
# from the land polygons, so the square is surrounded by four thin land strips, border_gap m
# outside it. A ship that leaves the square therefore runs aground (or goes outside the
# map) and stops. The strips are separate polygons, since MAR-AST only reads the outer
# outline of each polygon: a ring with a hole would make the whole square land.

def make_open_water_map(size = 10000, border_gap = 200, border_width = 500, fileName = None):
    if fileName is None:
        fileName = f'open_water_{round(size/1000)}km.gpkg'
    inner, outer = -border_gap, size + border_gap
    edge = border_width
    strips = [box(inner - edge, inner - edge, outer + edge, inner),  # South (box takes min x, min y, max x, max y, x = east)
              box(inner - edge, outer, outer + edge, outer + edge),  # North
              box(inner - edge, inner, inner, outer),                # West
              box(outer, inner, outer + edge, outer)]                # East
    crs = "EPSG:3857"  # Same as MAR-AST's maps. Coordinates are used as metres
    layers = {
        "frame_3857": [box(inner - edge, inner - edge, outer + edge, outer + edge)],
        "ocean_3857": [box(inner, inner, outer, outer)],
        "land_3857": strips,
        "coast_3857": strips,
    }
    path = REPO_ROOT / "maps" / fileName
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # Start from an empty file, so no old layers are left
    for layer, geometries in layers.items():
        gpd.GeoDataFrame(geometry=geometries, crs=crs).to_file(path, layer=layer, driver="GPKG")
    print(f'Wrote {path}')
    return path

if __name__ == "__main__":
    make_open_water_map(*(float(arg) for arg in sys.argv[1:3]))

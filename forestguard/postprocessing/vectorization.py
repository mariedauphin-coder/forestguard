"""Convert binary raster masks to georeferenced GeoJSON polygons."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import geopandas as gpd
from typing import Any


class MaskVectorizer:
    """
    Converts a binary (H, W) numpy mask + rasterio transform to a
    GeoDataFrame of polygons, optionally reprojected to WGS-84.
    """

    def __init__(
        self,
        simplify_tolerance: float = 10.0,   # metres
        output_crs: str = "EPSG:4326",
    ) -> None:
        self.simplify_tolerance = simplify_tolerance
        self.output_crs = output_crs

    def vectorize(
        self,
        mask: np.ndarray,
        transform: Affine,
        src_crs: Any,
    ) -> gpd.GeoDataFrame:
        """
        Parameters
        ----------
        mask      : (H, W) uint8, 1 = deforested
        transform : rasterio Affine transform of the mask
        src_crs   : rasterio CRS or EPSG string of the source raster

        Returns
        -------
        GeoDataFrame with columns: geometry, area_ha, centroid_lon, centroid_lat
        """
        polygons = []
        for geom_dict, value in shapes(mask, mask=mask.astype(bool), transform=transform):
            if value == 0:
                continue
            geom = shape(geom_dict)
            if self.output_crs and str(src_crs) != self.output_crs:
                geom_dict_reproj = transform_geom(
                    src_crs, self.output_crs, mapping(geom)
                )
                geom = shape(geom_dict_reproj)
            if self.simplify_tolerance > 0:
                geom = geom.simplify(self.simplify_tolerance / 111_320)  # deg/m approx
            if not geom.is_empty:
                polygons.append(geom)

        if not polygons:
            return gpd.GeoDataFrame(
                columns=["geometry", "area_ha", "centroid_lon", "centroid_lat"],
                geometry="geometry",
                crs=self.output_crs,
            )

        merged = unary_union(polygons)
        geoms = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]

        rows = []
        for g in geoms:
            area_ha = self._area_ha(g)
            centroid = g.centroid
            rows.append(
                {
                    "geometry": g,
                    "area_ha": round(area_ha, 4),
                    "centroid_lon": round(centroid.x, 6),
                    "centroid_lat": round(centroid.y, 6),
                }
            )

        return gpd.GeoDataFrame(rows, geometry="geometry", crs=self.output_crs)

    @staticmethod
    def _area_ha(geom) -> float:
        """Approximate area in hectares from WGS-84 geometry (crude, use projected CRS for accuracy)."""
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        area, _ = geod.geometry_area_perimeter(geom)
        return abs(area) / 10_000

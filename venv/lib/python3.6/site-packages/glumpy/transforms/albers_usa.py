# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
"""
An alias for ConicEqualArea, with USA-centric defaults:

   scale:      1285
   translate:  (960, 500)
   rotation:   96°, 0°
   center:     -0.6°, 38.7°
   parallels:  29.5°, 45.5°

This makes it suitable for displaying the United States, centered around
Hutchinson, Kansas in a 960×500 area. The central meridian and parallels are
specified by the USGS in the 1970 National Atlas.  Albers projection

See: https://github.com/mbostock/d3/blob/master/src/geo/conic-equal-area.js
     http://mathworld.wolfram.com/AlbersEqual-AreaConicProjection.html
     http://en.wikipedia.org/wiki/Albers_projection
"""
from . conic_equal_area import ConicEqualArea

# ESRI:102003
Lower48 = ConicEqualArea(scale=1285,
                         parallels = (29.5, 45.5),
                         rotate = (96,0),
                         translate = (0,0),
                         center = (0.01, -0.6))

# EPSG:3338
Alaska = ConicEqualArea(scale=1285*.35,
                        parallels = (55, 65),
                        rotate = (154, 0),
                        translate = (0,0),
                        center = (.35, -.8))

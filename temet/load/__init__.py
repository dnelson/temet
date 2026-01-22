"""Handle loading of all known data types from stored files.

Covers direct simulation outputs (snapshots, group catalogs) as well as
post-processing data catalogs (auxCats), as well as observational data.
"""

from . import (
    auxcat,
    auxcat_fields,
    data,
    groupcat,
    groupcat_fields_aux,
    groupcat_fields_custom,
    groupcat_fields_post,
    simtxt,
    snap_fields,
    snap_fields_custom,
    snapshot,
)

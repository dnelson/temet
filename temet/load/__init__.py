""" Handle loading of all known data types from stored files.
Covers direct simulation outputs (snapshots, group catalogs) as well as
post-processing data catalogs (auxCats), as well as observational data.
"""
from . import auxcat
from . import data
from . import groupcat
from . import groupcat_fields_custom
from . import groupcat_fields_aux
from . import groupcat_fields_post
from . import simtxt
from . import snapshot
from . import snap_fields
from . import snap_fields_custom

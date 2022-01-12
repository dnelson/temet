""" Handle loading of all known data types from stored files.
Covers direct simulation outputs (snapshots, group catalogs) as well as
post-processing data catalogs (auxCats), as well as observational data.
"""
from . import auxcat
from . import data
from . import groupcat
from . import snapshot

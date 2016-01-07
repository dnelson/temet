"""
util/helper.py
  General helper functions, small algorithms, basic I/O, etc.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np

def getIDIndexMap(ids):
	""" Return an array of size max(ids)-min(ids) such that array[ID-min(ids)] is the 
		index of the original array ids where ID is found (assumes a one to one mapping, 
		not repeated indices as in the case of parentIDs for tracers).
	"""
	minid = np.min(ids)
	maxid = np.max(ids)

	dtype = 'uint32'
	if ids.size >= 2e9:
		dtype = 'uint64'

	# Paul's direct indexing approach (pretty fast)
	arr = np.zeros( maxid-minid+1, dtype=dtype )
	arr[ids-minid] = np.arange( ids.size, dtype=dtype )

	# C-style loop approach (good for sparse IDs)
	#arr = ulonarr(maxid-minid+1)
	#for i=0ULL,n_elements(ids)-1L do arr[ids[i]-minid] = i

	# looped where approach (never a good idea)
	#arr = l64indgen(maxid-minid+1)
	#for i=minid,maxid do begin
	#  w = where(ids eq i,count)
	#  if (count gt 0) then arr[i] = w[0]
	#endfor

	# reverse histogram approach (good for dense ID sampling, maybe better by factor of ~2)
	#arr = l64indgen(maxid-minid+1)
	#h = histogram(ids,rev=rev,omin=omin)
	#for i=0L,n_elements(h)-1 do if (rev[i+1] gt rev[i]) then arr[i] = rev[rev[i]:rev[i+1]-1]

	return arr, minid

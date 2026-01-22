"""
Cosmological simulations - auxiliary catalogs of time-evolution (merger trees, tracers, etc).
"""

import numpy as np
from scipy.signal import medfilt


def mergerTreeQuant(sP, pSplit, treeName, quant, smoothing=None):
    """For every subhalo, compute an assembly/related quantity using a merger tree.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      treeName (str): specify merger tree.
      quant (str): specify subhalo quantity (available in merger tree) or custom user-defined quantity.
      smoothing (None or list): if not None, then smooth the quantity along the time dimension
        according to the tuple specification ``[method,windowSize,windowVal,order]`` where
        ``method`` should be ``mm`` (moving median of ``windowSize``), ``ma`` (moving average
        of ``windowSize``), or ``poly`` (poly fit of ``order`` N). The window size is given
        in units of ``windowVal`` which can be only ``snap``.

    Returns:
      a 2-tuple composed of

      - **result** (:py:class:`~numpy.ndarray`): 1d array, value for each subhalo.
      - **attrs** (dict): metadata.
    """
    assert quant in ["zForm", "isSat_atForm", "rad_rvir_atForm", "dmFrac_atForm"]
    assert pSplit is None  # not implemented

    def _ma(X, windowSize):
        """Running mean. Endpoints are copied unmodified in bwhalf region."""
        r = np.zeros(X.size, dtype=X.dtype)
        bwhalf = int(windowSize / 2.0)

        cumsum = np.cumsum(np.insert(X, 0, 0))
        r[bwhalf:-bwhalf] = (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize
        r[0:bwhalf] = X[0:bwhalf]
        r[-bwhalf:] = X[-bwhalf:]
        return r

    def _mm(X, windowSize):
        """Running median."""
        return medfilt(X, windowSize)

    # prepare catalog metadata
    desc = "Merger tree quantity (%s)." % quant
    if smoothing is not None:
        desc += " Using smoothing [%s]." % "_".join([str(s) for s in smoothing])
    select = "All Subfind subhalos."

    # load snapshot and subhalo information
    redshifts = sP.snapNumToRedshift(all=True)

    nSubsTot = sP.numSubhalos
    ids = np.arange(nSubsTot, dtype="int32")  # currently, always process all

    # choose tree fields to load, and validate smoothing request
    groupFields = None
    fields = ["SnapNum"]

    if quant == "zForm":
        fields += ["SubhaloMass"]
        mpb_valkey = "SubhaloMass"

        dtype = "float32"
        assert len(smoothing) == 3
        assert smoothing[2] == "snap"  # todo: e.g. Gyr, scalefac
    elif quant == "isSat_atForm":
        fields += ["SubfindID", "SubhaloGrNr"]
        groupFields = ["GroupFirstSub"]
        mpb_valkey = "SubfindID"

        dtype = "int16"
        assert smoothing is None
    elif quant == "rad_rvir_atForm":
        fields += ["SubhaloGrNr", "SubhaloPos"]
        groupFields = ["GroupPos", "Group_R_Crit200"]
        mpb_valkey = "SubhaloGrNr"

        dtype = "float32"
        assert smoothing is None
    elif quant == "dmFrac_atForm":
        fields += ["SubhaloMass", "SubhaloMassType"]
        mpb_valkey = "SubhaloMassType"
        dmPtNum = sP.ptNum("dm")

        dtype = "float32"
        assert smoothing is None

    if groupFields is not None:
        # we also need group properties, at all snapshots, so load now
        cacheKey = "mtq_%s" % quant
        if cacheKey in sP.data:
            # if computing these auxcats over many snapshots, use sP.setSnap() to preserve mtq_* cache
            print("Loading [%s] from sP.data cache..." % cacheKey)
            groups = sP.data[cacheKey]
        else:
            groups = {}
            prevSnap = sP.snap

            for snap in sP.validSnapList():
                sP.setSnap(snap)
                if snap % 10 == 0:
                    print("%d%%" % (float(snap) / len(sP.validSnapList()) * 100), end=", ")
                groups[snap] = sP.groupCat(fieldsHalos=groupFields, sq=False)["halos"]

            sP.setSnap(prevSnap)
            sP.data[cacheKey] = groups
            print("Saved [%s] into sP.data cache." % cacheKey)

    # allocate return, NaN indicates not computed (e.g. not in tree at sP.snap)
    r = np.zeros(nSubsTot, dtype=dtype)

    if np.issubdtype(dtype, np.integer):
        r.fill(-1)
    else:
        r.fill(np.nan)

    # load all trees at once
    mpbs = sP.loadMPBs(ids, fields=fields, treeName=treeName)

    # loop over subhalos
    printFac = 100.0 if sP.res > 512 else 10.0

    for i in range(nSubsTot):
        if i % int(nSubsTot / printFac) == 0 and i <= nSubsTot:
            print("   %4.1f%%" % (float(i + 1) * 100.0 / nSubsTot))

        if i not in mpbs:
            continue  # subhalo ID i not in tree at sP.snap

        # todo: could generalize here into generic reduction operations over a given tree field
        # e.g. 'max', 'min', 'mean' of 'SubhaloSFR', 'SubhaloGasMetallicity', ... in addition to
        # more specialized calculations such as formation time
        loc_vals = mpbs[i][mpb_valkey]
        loc_snap = mpbs[i]["SnapNum"]

        # smoothing
        if smoothing is not None:
            if loc_snap.size < smoothing[1] + 1:
                continue

            if smoothing[0] == "mm":  # moving median window of size N snapshots
                loc_vals = _mm(loc_vals, windowSize=smoothing[1])

            if smoothing[0] == "ma":  # moving average window of size N snapshots
                loc_vals = _ma(loc_vals, windowSize=smoothing[1])

            if smoothing[0] == "poly":  # polynomial fit of Nth order
                coeffs = np.polyfit(loc_snap, loc_vals, smoothing[1])
                loc_vals = np.polyval(coeffs, loc_snap)  # resample to original X-pts

        # general quantities
        # (currently none)

        # custom quantities: 'at formation' (at end of MPB)
        if quant == "isSat_atForm":
            # subpar = loc_vals[-1]
            subid = loc_vals[-1]  # mpbs[i]['SubfindID'][-1]
            subgrnr = mpbs[i]["SubhaloGrNr"][-1]
            subgrnr_snap = loc_snap[-1]
            grfirstsub = groups[subgrnr_snap][subgrnr]

            if grfirstsub == subid:
                # at the MPB last snapshot, GroupFirstSub[SubhaloGrNr] points to this same subhalo,
                # as recorded by SubfindID at this snapshot, so we are a central
                r[i] = 0
            else:
                # GroupFirstSub points elsewhere
                r[i] = 1

        if quant == "rad_rvir_atForm":
            sub_pos = mpbs[i]["SubhaloPos"][-1, :]
            subgrnr = loc_vals[-1]
            subgrnr_snap = loc_snap[-1]

            par_pos = groups[subgrnr_snap]["GroupPos"][subgrnr, :]
            par_rvir = groups[subgrnr_snap]["Group_R_Crit200"][subgrnr]
            dist = sP.periodicDists(sub_pos, par_pos)

            if dist == 0.0:
                # mostly the case for centrals
                r[i] = 0.0
            elif par_rvir == 0.0:
                # can be zero for small groups (why?)
                r[i] = np.inf
            else:
                r[i] = dist / par_rvir

        if quant == "dmFrac_atForm":
            sub_masstype = loc_vals[-1, :]
            sub_mass = mpbs[i]["SubhaloMass"][-1]

            r[i] = sub_masstype[dmPtNum] / sub_mass

        # custom quantities
        if quant == "zForm":
            # where does half of max of [smoothed] total mass occur?
            halfMaxVal = loc_vals.max() * 0.5

            # if smoothing[0] == 'poly': # root find on the polynomial coefficients (not so simple)
            # coeffs[-1] -= halfMaxVal # shift such that we find the M=halfMaxVal not M=0 roots
            # roots = np.polynomial.polynomial.polyroots(coeffs[::-1]) # there are many
            w = np.where(loc_vals >= halfMaxVal)[0]
            assert len(w)  # by definition

            # linearly interpolate between snapshots
            snap0 = loc_snap[w].min()
            ind0 = w.max()  # lowest snapshot where mass exceeds halfMaxVal
            ind1 = ind0 + 1  # lower snapshot (earlier in time)

            assert snap0 == loc_snap[ind0]

            if ind0 == loc_vals.size - 1:
                # only at first tree entry
                z_form = redshifts[loc_snap[ind0]]
            else:
                assert ind0 >= 0 and ind0 < loc_vals.size - 1
                assert ind1 > 0 and ind1 <= loc_vals.size - 1

                z0 = redshifts[loc_snap[ind0]]
                z1 = redshifts[loc_snap[ind1]]
                m0 = loc_vals[ind0]
                m1 = loc_vals[ind1]

                assert m0 >= halfMaxVal and m1 <= halfMaxVal

                # linear interpolation, find redshift where mass=halfMaxVal
                z_form = (halfMaxVal - m0) / (m1 - m0) * (z1 - z0) + z0
                assert z_form >= z0 and z_form <= z1

            assert z_form >= 0.0
            r[i] = z_form

    subhaloIDsTodo = np.arange(nSubsTot, dtype="int32")

    attrs = {"Description": desc.encode("ascii"), "Selection": select.encode("ascii"), "subhaloIDs": subhaloIDsTodo}

    return r, attrs


def tracerTracksQuant(sP, pSplit, quant, op, time, norm=None):
    """For every subhalo, compute a assembly/accretion/related quantity using the tracker track catalogs.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      quant (str): specify tracer tracks catalog quantity.
      op (str): statistical operation to apply across all the child tracers of the subhalo.
      time (str or None): sample quantity at what time? e.g. 'acc_time_1rvir'.
      norm (str or None): normalize quantity by a second quantity, e.g. 'tvir_tacc'.

    Returns:
      a 2-tuple composed of

      - **result** (:py:class:`~numpy.ndarray`): 1d array, value for each subhalo.
      - **attrs** (dict): metadata.
    """
    from ..tracer.evolution import (
        ACCMODES,
        accMode,
        accTime,
        mpbValsAtAccTimes,
        mpbValsAtRedshifts,
        tracersMetaOffsets,
        tracersTimeEvo,
        trValsAtAccTimes,
    )
    from ..tracer.montecarlo import defParPartTypes

    assert pSplit is None  # not implemented
    assert op in ["mean"]  # ,'sample']
    assert quant in ["angmom", "entr", "temp", "acc_time_1rvir", "acc_time_015rvir", "dt_halo"]
    assert time is None or time in ["acc_time_1rvir", "acc_time_015rvir"]
    assert norm is None or norm in ["tvir_tacc", "tvir_cur"]

    def _nansum(x):
        """Helper."""
        N = np.count_nonzero(~np.isnan(x))

        r = np.nan
        if N > 0:
            r = np.nansum(x)

        return r, N

    # prepare catalog metadata
    desc = "Tracer tracks quantity (%s) using [%s] over [%s]." % (quant, op, time)
    select = "All Subfind subhalos."

    # load snapshot and subhalo information
    subhaloIDsTodo = np.arange(sP.numSubhalos, dtype="int32")

    nSubsDo = len(subhaloIDsTodo)

    # allocate return, NaN indicates not computed (e.g. not in tree at sP.snap)
    nTypes = len(defParPartTypes) + 1  # store separately by z=0 particle type, +1 for 'all' combined
    nModes = len(ACCMODES) + 1  # store separately by mode, +1 for 'all' combined

    r = np.zeros((nSubsDo, nTypes, nModes), dtype="float32")
    N = np.zeros((nSubsDo, nTypes, nModes), dtype="int32")  # accumulate bin counts
    r.fill(np.nan)

    # load 1D reduction of the tracer tracks corresponding to the requested quantity, per tracer,
    # taking into account the time specification. note that 'accretion times' as the quantity
    # are already just 1D, while the others need to be selected at a specific time, or averaged
    # over a specific time window, reducing the (Nsnaps,Ntr) shaped tracks into a 1D (Ntr) shape
    if quant == "acc_time_1rvir":
        assert time is None and norm is None
        tracks = accTime(sP, rVirFac=1.0)

    elif quant == "acc_time_015rvir":
        assert time is None and norm is None
        tracks = accTime(sP, rVirFac=0.15)

    elif quant == "dt_halo":
        assert time is None and norm is None
        age_universe_rad100rvir = sP.units.redshiftToAgeFlat(accTime(sP, rVirFac=1.0))
        age_universe_rad015rvir = sP.units.redshiftToAgeFlat(accTime(sP, rVirFac=0.15))
        tracks = age_universe_rad015rvir - age_universe_rad100rvir
        assert np.nanmin(tracks) >= 0.0  # negative would contradict definition

        # handful of zeros, set to nan
        w = np.where(tracks == 0.0)[0]
        if len(w):
            print(" Note: setting [%d] of [%d] dt_halo==0 to nan." % (len(w), tracks.size))
            tracks[w] = np.nan
    else:
        if time is None:
            # full tracks (what are we going to do with these?)
            tracks = tracersTimeEvo(sP, quant, all=True)  # snapStep=None
            # do e.g. a mean across all time
            assert norm is None
            print("todo: finish")

        elif time == "acc_time_1rvir":
            tracks = trValsAtAccTimes(sP, quant, rVirFac=1.0)

        elif time == "acc_time_015rvir":
            tracks = trValsAtAccTimes(sP, quant, rVirFac=0.15)

    assert tracks.ndim == 1

    # normalization?
    if norm is not None:
        # what MPB property to normalize by, and take it at which time (e.g. Tvir at tAcc)
        norm_field, norm_time = norm.split("_")

        if norm_time == "tacc":
            norm_vals = mpbValsAtAccTimes(sP, norm_field, rVirFac=1.0)
        if norm_time == "z0":
            norm_vals = mpbValsAtRedshifts(sP, norm_field, 0.0)
        if norm_time == "cur":
            norm_vals = mpbValsAtRedshifts(sP, norm_field, sP.redshift)

        assert tracks.shape == norm_vals.shape
        if tracks.max() > 20.0 and norm_vals.max() < 20.0:
            assert 0  # check log of norm_vals

        tracks /= norm_vals

    # load mode decomposition, metadata offsets
    mode = accMode(sP)

    meta, nTracerTot = tracersMetaOffsets(sP, all="Subhalo")

    # loop over subhalos
    for i, subhaloID in enumerate(subhaloIDsTodo):
        if i % np.max([1, int(nSubsDo / 100)]) == 0 and i <= nSubsDo:
            print("   %4.1f%%" % (float(i + 1) * 100.0 / nSubsDo))

        # loop over modes
        for modeNum, modeName in enumerate(ACCMODES.keys()):
            modeVal = ACCMODES[modeName]
            # print('   [%s] %s (val=%d)' % (modeNum,modeName,modeVal))

            # loop over partTypes
            for j, ptName in enumerate(defParPartTypes):
                # slice starting/ending indices for tracers local to this subhalo
                i0 = meta[ptName]["offset"][subhaloID]
                i1 = i0 + meta[ptName]["length"][subhaloID]

                if i1 == i0:
                    continue  # zero length of this type

                # mode segregation
                if modeNum < len(ACCMODES):
                    w_mode = np.where(mode[i0:i1] == modeVal)[0]
                else:
                    w_mode = np.arange(i1 - i0)  # 'all', use all

                if w_mode.size == 0:
                    continue  # no tracers of this mode in this subhalo

                # local slice, mode dependent
                loc_vals = tracks[i0:i1][w_mode]

                # should never overwrite anything
                assert np.isnan(r[i, j, modeNum])
                assert N[i, j, modeNum] == 0

                # store intermediate value (e.g. sum for mean) and counts for later calculation of op
                if op == "mean":
                    r[i, j, modeNum], N[i, j, modeNum] = _nansum(loc_vals)

                # print(j,ptName,N[i,j,modeNum],r[i,j,modeNum],i1-i0,w_mode.size)

            # do op for all part types together, for this mode
            assert np.isnan(r[i, j + 1, modeNum])
            assert N[i, j + 1, modeNum] == 0

            if op == "mean":
                r[i, j + 1, modeNum], _ = _nansum(r[i, 0 : j + 1, modeNum])
                N[i, j + 1, modeNum] = np.sum(N[i, 0 : j + 1, modeNum])
                # print(j+1,'all',N[i,j+1,modeNum],r[i,j+1,modeNum])

        # do op for all modes together, for each partType+all
        # print('   [%s] ALL MODES' % (modeNum+1))

        for j in range(len(defParPartTypes) + 1):
            assert np.isnan(r[i, j, modeNum + 1])  # should never overwrite anything
            assert N[i, j, modeNum + 1] == 0

            r[i, j, modeNum + 1], _ = _nansum(r[i, j, 0 : modeNum + 1])
            N[i, j, modeNum + 1] = np.sum(N[i, j, 0 : modeNum + 1])
            # print(j,' - ',N[i,j,modeNum+1],r[i,j,modeNum+1])

    # now, normalize element by element
    w = np.where(N == 0)
    assert len(np.where(np.isnan(r[w]))[0]) == len(w[0])  # all zero counts should have nan value

    w = np.where(N > 0)
    assert len(np.where(np.isnan(r[w]))[0]) == 0  # all nonzero counts should have finite value

    r[w] /= N[w]

    attrs = {"Description": desc.encode("ascii"), "Selection": select.encode("ascii")}
    #'accModes'     : ACCMODES, # encoding errors, would need to do more carefully
    #'parPartTypes' : defParPartTypes}

    return r, attrs

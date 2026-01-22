"""
Load AREPO .txt diagnostic and output files (by converting and cachine to HDF5).
"""

import glob
import os
import subprocess
from os.path import expanduser, isdir, isfile

import h5py
import numpy as np

from ..util.helper import evenlySample, tail


def sfrTxt(sim):
    """Load and parse sfr.txt."""
    nPts = 2000

    # cached? in sim object or on disk?
    if "sfrd" in sim.data:
        return sim.data["sfrd"]

    saveFilenames = sorted(glob.glob(sim.derivPath + "sfrtxt_*.hdf5"))

    if len(saveFilenames):
        print(" Loaded: [%s]" % saveFilenames[-1].simlit(sim.derivPath)[1])
        r = {}
        with h5py.File(saveFilenames[-1], "r") as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # check possible locations for 'sfr.txt' file
    path = sim.simPath + "sfr.txt"
    if not isfile(path):
        path = sim.simPath + "txt-files/sfr.txt"
    if not isfile(path):
        path = sim.derivPath + "sfr.txt"

    # does not exist? create similar, but snapshot time-simaced, data
    if not isfile(path):
        print("WARNING: Cannot find [sfr.txt] file, deriving data from snapshots.")
        keys = ["scaleFac", "totSfrRate"]  # ['totalSm','sfrMsunPerYr','totalSumMassStars']
        r = {k: [] for k in keys}

        sim = sim.copy()
        for snap in sim.validSnapList():
            sim.setSnap(snap)
            print(f"[{snap = :3d}] at z = {sim.redshift:5.2f}")

            r["scaleFac"].append(sim.units.scalefac)
            r["totSfrRate"].append(np.sum(sim.gas("sfr")))  # msun/yr

        r["scaleFac"] = np.array(r["scaleFac"])
        r["totSfrRate"] = np.array(r["totSfrRate"])

    else:
        # columns: All.Time, total_sm, totsfrrate, rate_in_msunperyear, total_sum_mass_stars, cum_mass_stars
        data = np.loadtxt(path)

        r = {
            "scaleFac": evenlySample(data[:, 0], nPts, logsimace=True),
            "totalSm": evenlySample(data[:, 1], nPts, logsimace=True),  # from probabilistic formula
            "totSfrRate": evenlySample(data[:, 2], nPts, logsimace=True),  # sum of TimeBinSfr (i.e. gas cells)
            "sfrMsunPerYr": evenlySample(data[:, 3], nPts, logsimace=True),  # totalSm divided by timestep
            "totalSumMassStars": evenlySample(data[:, 4], nPts, logsimace=True),  # actual star particle mass formed
            "cumMassStars": evenlySample(data[:, 5], nPts, logsimace=True),
        }  # sum of above (somehow strange)

    r["redshift"] = 1.0 / r["scaleFac"] - 1.0
    r["sfrd"] = r["totSfrRate"] / sim.boxSizeCubicComovingMpc  # a constant cMpc^3 (equals pMpc^3 at z=0)

    # save
    saveFilename = sim.derivPath + "sfrtxt_%.2f.hdf5" % r["scaleFac"].max()
    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]
    print(" Saved: [%s]" % saveFilename.simlit(sim.derivPath)[1])

    sim.data["sfrd"] = r  # attach to sim as cache

    return r


def getCpuTxtLastTimestep(filePath):
    """Parse cpu.txt for last timestep number and number of CPUs/tasks and total CPU hours."""
    # hardcode Illustris-1 finalized data and complicated txt-files
    if "L75n1820FP/" in filePath:
        return 1.0, 912915, 8192, 0
    if filePath == expanduser("~") + "/sims.illustris/L75n910FP/output/cpu.txt":
        return 1.0, 876580, 4096, 0
    if filePath == expanduser("~") + "/sims.illustris/L75n455FP/output/cpu.txt":
        return 1.0, 268961, 128, 0
    if filePath == expanduser("~") + "/sims.TNG/L75n1820TNG/output/cpu.txt":
        return 1.0, 11316835, 10752, 0
    if filePath == expanduser("~") + "/sims.TNG/L205n2500TNG/output/cpu.txt":
        return 1.0, 6203063, 24000, 0
    if filePath == expanduser("~") + "/sims.TNG/L35n2160TNG_halted/output/cpu.txt":
        return 0.149494, 2737288, 16320, 0

    if not isfile(filePath):
        return 0, 0, 0, 0

    lines = tail(filePath, 100).split("\n")[::-1]
    for i, line in enumerate(lines):
        if "Step " in line:
            maxSize = int(line.split(", ")[0].split(" ")[1]) + 1
            maxTime = float(line.split(", ")[1].split(" ")[1])
            numCPUs = np.int32(line.split(", ")[2].split(" ")[1])
            cpuHours = float(lines[i - 2].split()[3]) * numCPUs / 60**2
            break

    return maxTime, maxSize, numCPUs, cpuHours


def loadCpuTxt(basePath, keys=None, hatbMin=0, skipWrite=False):
    """Load and parse Arepo cpu.txt, save into hdf5 format.

    If hatbMin>0, then save only timesteps with active time bin above this value.
    """
    saveFilename = basePath + "data.files/cpu.hdf5"

    if not isdir(basePath + "data.files/"):
        saveFilename = basePath + "postprocessing/cpu.hdf5"

    filePath = basePath + "output/cpu.txt"
    if not isfile(filePath):
        filePath = basePath + "output/txt-files/cpu.txt"
    if not isfile(filePath):
        filePath = basePath + "cpu.txt"
    if not isfile(filePath):
        print("WARNING: Failed to find [%s]." % filePath)
        return None

    r = {}

    cols = None

    # load save if it exists already
    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            read_keys = keys if keys is not None else f.keys()

            # check size and ending time
            maxTimeSaved = f["time"][()].max()
            maxStepSaved = f["step"][()].max()
            maxTimeAvail, maxStepAvail, _, _ = getCpuTxtLastTimestep(filePath)

            if maxTimeAvail > maxTimeSaved * 1.001:
                # recalc for new data
                print(
                    "recalc [%f to %f] [%d to %d] %s"
                    % (maxTimeSaved, maxTimeAvail, maxStepSaved, maxStepAvail, basePath)
                )
                os.remove(saveFilename)
                return loadCpuTxt(basePath, keys, hatbMin)

            for key in read_keys:
                if key not in f:
                    continue  # e.g. hydro fields in DMO runs
                r[key] = f[key][()]
            r["numCPUs"] = f["numCPUs"][()]
    else:
        # determine number of timesteps in file, and number of CPUs
        _, maxSize, r["numCPUs"], _ = getCpuTxtLastTimestep(filePath)

        maxSize = int(maxSize * 1.2)  # since we filter empties out anyways, let file grow as we read

        printPath = "/".join(basePath.split("/")[-3:])
        print("[%s] maxSize: %d numCPUs: %d, loading..." % (printPath, maxSize, r["numCPUs"]))

        r["step"] = np.zeros(maxSize, dtype="int32")
        r["time"] = np.zeros(maxSize, dtype="float32")
        r["hatb"] = np.zeros(maxSize, dtype="int16")

        # parse
        f = open(filePath)

        step = None
        hatbSkip = False

        # chunked load
        while 1:
            lines = f.readlines(100000)
            if not lines:
                break

            for line in lines:
                line = line.strip()

                # timestep header
                if line[0:4] == "Step":
                    line = line.split(",")
                    hatb = int(line[4].split(": ")[1])

                    if hatb < hatbMin and hatb > 0:
                        hatbSkip = True  # skip until next timestep header
                    else:
                        hatbSkip = False  # proceed normally

                    step = int(line[0].split(" ")[1])
                    time = float(line[1].split(": ")[1])

                    if step % 100000 == 0 and step > 0:
                        print(" [%d] %8.6f hatb=%d %s" % (step, time, hatb, hatbSkip))

                    continue

                if hatbSkip:
                    continue
                if line == "" or line[0:4] == "diff":
                    continue

                # normal line
                line = line.split()

                name = line[0]

                # names with a space
                offset = 0
                if line[1] in ["vel", "zone", "surface", "search"]:
                    name = line[0] + "_" + line[1]
                    offset = 1
                name = name.replace("/", "_")

                # timings
                if name not in r:
                    r[name] = np.zeros((maxSize, 4), dtype="float32")

                # how many columns (how old is this file)?
                if cols is None:
                    cols = len(line) - 1
                else:
                    if cols != len(line) - 1:
                        # corrupt line
                        r[name][step, :] = np.nan
                        continue

                if cols == 4:
                    r[name][step, 0] = float(line[1 + offset].strip())  # diff time
                    r[name][step, 1] = float(line[2 + offset].strip()[:-1])  # diff percentage
                    r[name][step, 2] = float(line[3 + offset].strip())  # cumulative time
                    r[name][step, 3] = float(line[4 + offset].strip().replace("%", ""))  # cumulative percentage

                if cols == 3:
                    r[name][step, 0] = float(line[1 + offset].strip())  # diff time
                    r[name][step, 2] = float(line[2 + offset].strip())  # cumulative time
                    r[name][step, 3] = float(line[3 + offset].strip()[:-1])  # cumulative percentage

                r["step"][step] = step
                r["time"][step] = time
                r["hatb"][step] = hatb

        f.close()

        # compress (remove empty entries)
        w = np.where(r["hatb"] > 0)

        for key in r.keys():
            if key == "numCPUs":
                continue
            if r[key].ndim == 1:
                r[key] = r[key][w]
            if r[key].ndim == 2:
                r[key] = r[key][w, :]

        # write into hdf5
        if skipWrite:
            return r

        with h5py.File(saveFilename, "w") as f:
            for key in r.keys():
                f[key] = r[key]

    return r


def loadTimebinsTxt(basePath):
    """Load and parse Arepo timebins.txt, save into hdf5 format."""
    filePath = basePath + "output/timebins.txt"

    saveFilename = basePath + "data.files/timebins.hdf5"
    if not isdir(basePath + "data.files/"):
        saveFilename = basePath + "postprocessing/timebins.hdf5"

    r = {}

    def _getTimebinsLastTimestep():
        if not isfile(filePath):
            return 0.0, 0.0, 0.0, 0.0  # no recalculate

        lines = tail(filePath, 30)
        binNums = []
        for line in lines.split("\n")[::-1]:
            if "Sync-Point " in line:
                sp, time, redshift, ss, dloga = line.split(", ")
                maxStepAvail = int(sp.split(" ")[1])
                maxTimeAvail = float(time.split(": ")[1])
                break
            if "bin=" in line:
                binNums.append(int(line.split("bin=")[1].split()[0]))

        binMin = np.min(binNums) - 10  # leave room in case true minimum not represented in final timestep
        binMax = np.max(binNums) + 20  # leave room
        nBins = binMax - binMin + 1
        return maxStepAvail, maxTimeAvail, nBins, binMin

    # load save if it exists already
    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            # check size and ending time
            maxTimeSaved = f["time"][()].max()
            maxStepSaved = f["step"][()].max()

            maxStepAvail, maxTimeAvail, _, _ = _getTimebinsLastTimestep()

            if maxTimeAvail > maxTimeSaved * 1.001:
                # recalc for new data
                print(
                    "recalc [%f to %f] [%d to %d] %s"
                    % (maxTimeSaved, maxTimeAvail, maxStepSaved, maxStepAvail, basePath)
                )
                os.remove(saveFilename)
                return loadTimebinsTxt(basePath)

            for key in f:
                r[key] = f[key][()]
    else:
        # determine number of timesteps in file, ~maximum number of bins, and allocate
        maxSize, lastTime, nBins, binMin = _getTimebinsLastTimestep()

        print("[%s] maxSize: %d lastTime: %.2f (nBins=%d), loading..." % (basePath, maxSize, lastTime, nBins))

        r["step"] = np.zeros(maxSize, dtype="int32")
        r["time"] = np.zeros(maxSize, dtype="float32")
        r["bin_num"] = np.zeros(nBins, dtype="int16") - 1
        r["bin_dt"] = np.zeros(nBins, dtype="float32")

        r["active"] = np.zeros((nBins, maxSize), dtype="bool")
        r["n_grav"] = np.zeros((nBins, maxSize), dtype="int64")
        r["n_hydro"] = np.zeros((nBins, maxSize), dtype="int64")
        r["avg_time"] = np.zeros((nBins, maxSize), dtype="float32")
        r["cpu_frac"] = np.zeros((nBins, maxSize), dtype="float32")

        # parse
        f = open(filePath)

        step = None

        # chunked load
        while 1:
            lines = f.readlines(100000)
            if not lines:
                break

            for line in lines:
                line = line.strip()
                if line == "" or line[0:8] == "Occupied" or line[0:4] == "----":
                    continue
                if line[0:13] == "Total active:" or line[0:15] == "PM-Step. Total:":
                    continue

                # timestep header
                # Sync-Point 71887, Time: 1, Redshift: 2.22045e-16, Systemstep: 9.25447e-06, Dloga: 9.25451e-06
                if line[0:10] == "Sync-Point":
                    sp, time, redshift, ss, dloga = line.split(", ")

                    step = int(sp.split(" ")[1]) - 1  # skip 0th step
                    time = float(time.split(": ")[1])

                    if step % 10000 == 0:
                        redshift = 1 / time - 1.0
                        print(" [%8d] t=%7.5f z=%6.3f" % (step, time, redshift))

                    # per step
                    r["step"][step] = step + 1  # actual sync-point number
                    r["time"][step] = time  # scalefactor

                    continue

                # normal line
                #    bin=42      15          14      0.000018509027       16          15            0.03      7.8%
                # X  bin=41       1           1      0.000009254513        1           1 <          0.02     11.1%
                line = line.replace("<", "").replace("*", "")  # delete
                line = line.replace("bin= ", "bin=")  # delete space if present
                line = line.split()

                active = False
                offset = 0
                if line[0] == "X":
                    active = True
                    offset = 1

                binNum = int(line[0 + offset][4:])
                n_grav = int(line[1 + offset])
                n_hydro = int(line[2 + offset])
                dt = float(line[3 + offset])
                # cum_grav = int(line[4+offset])
                # cum_hydro = int(line[5+offset])
                # A, D flagged using '<' and '*' have been removed
                avg_time = float(line[6 + offset])
                cpu_frac = float(line[7 + offset][:-1])  # remove trailing '%'

                if binNum == 0:
                    continue  # dt=0

                saveInd = binNum - binMin
                assert saveInd >= 0

                # per bin per step
                r["active"][saveInd, step] = active
                r["n_grav"][saveInd, step] = n_grav
                r["n_hydro"][saveInd, step] = n_hydro
                r["avg_time"][saveInd, step] = avg_time
                r["cpu_frac"][saveInd, step] = cpu_frac

                # per bin
                r["bin_num"][saveInd] = binNum
                r["bin_dt"][saveInd] = dt

        f.close()

        # condense (remove unused timebin spots)
        w = np.where(r["bin_num"] > 0)[0]

        for key in ["active", "n_grav", "n_hydro", "avg_time", "cpu_frac"]:
            r[key] = r[key][w, :]
        for key in ["bin_num", "bin_dt"]:
            r[key] = r[key][w]

        # write into hdf5
        with h5py.File(saveFilename, "w") as f:
            for key in r.keys():
                f[key] = r[key]

    return r


def loadMemoryTxt(basePath):
    """Parse the memory.txt file for a run (prelimiany)."""
    largest_alloc = 0.0
    largest_alloc_wo_generic = 0.0

    filePath = basePath + "output/memory.txt"

    if not isfile(filePath):
        filePath = basePath + "output/txt-files/memory.txt"

    if not isfile(filePath):
        return largest_alloc, largest_alloc_wo_generic  # no recalculate

    # load entire file
    with open(filePath) as f:
        lines = f.readlines()

    # find maximum values across all timesteps
    for line in lines:
        if "MEMORY:" not in line:
            continue

        alloc1 = float(line.split(" ")[5])  # MB
        alloc2 = float(line.split(" ")[-2])  # MB

        if alloc1 > largest_alloc:
            largest_alloc = alloc1
        if alloc2 > largest_alloc_wo_generic:
            largest_alloc_wo_generic = alloc2

    return largest_alloc, largest_alloc_wo_generic


def blackhole_details_mergers(sim, overwrite=False):
    """Convert the blackhole_details/ and blackhole_mergers/ files into HDF5 format."""
    cmd1 = 'cat blackhole_details_*.txt | sed -r "s/^BH=//" > blackhole_details.txt'
    cmd2 = "cat blackhole_mergers_*.txt > blackhole_mergers.txt"

    cachefile = sim.derivPath + "blackhole_details.hdf5"

    smbhs = {}

    # check for existence of cache
    if isfile(cachefile) and not overwrite:
        with h5py.File(cachefile, "r") as f:
            for smbh_id in f:
                smbhs[smbh_id] = {}
                for k in f[smbh_id].keys():
                    smbhs[smbh_id][k] = f[smbh_id][k][()]
        print(f"Loaded [{cachefile}].")

        return smbhs

    # concatenate txt files if needed
    path = sim.simPath + "txt-files/"

    if isdir(sim.simPath + "blackhole_details/"):
        # txt-files/ is not present yet, run is likely still in progress
        path = sim.simPath

    path1 = path + "blackhole_details/"
    path2 = path + "blackhole_mergers/"

    filename = path1 + "blackhole_details.txt"
    filename2 = filename.replace("details", "mergers")

    decompressed = False

    if not isfile(filename):
        # .tar.gz files present and directories not? decompress
        if isfile(path + "blackhole_details.tar.gz") and not isdir(path1):
            assert "txt-files/" in path  # should not occur otherwise

            # have write permission?
            # if not os.access(path, os.W_OK | os.X_OK):
            print("Decompressing blackhole_details.tar.gz...")
            subprocess.run(["tar", "-xzf", "blackhole_details.tar.gz"], cwd=path)
            print("Decompressing blackhole_mergers.tar.gz...")
            subprocess.run(["tar", "-xzf", "blackhole_mergers.tar.gz"], cwd=path)
            decompressed = True

        # concat file is missing, check for individual files
        filename_indiv = filename.replace(".txt", "_0.txt")
        if isfile(filename_indiv):
            # run concat
            print("Concat blackhole_details*.txt...")
            subprocess.run(cmd1, cwd=path1, shell=True)

            # remove all lines containing "BH=" (stdout races/bad data)
            subprocess.run('sed -i "/BH=/d" blackhole_details.txt', cwd=path1, shell=True)

            print("Concat blackhole_mergers*.txt...")
            subprocess.run(cmd2, cwd=path2, shell=True)

    # determine number of columns in details
    with open(filename) as f:
        line = f.readline()
        ncols = len(line.split())

    # if ncols == 6: # v0: columns: ID time BH_Mass mdot rho csnd
    # if ncols == 15: # v2: columns: ID time BH_Mass mdot rho cs hsml ngbmaxdist spin x y z vx vy vz

    # load details,
    ids = np.loadtxt(filename, usecols=[0], dtype="int64")
    data = np.loadtxt(filename, usecols=np.arange(1, ncols), dtype="float32")

    # load mergers, columns: thistask time id0 mass0 id1 mass1
    # where id0 is the SMBH that remains, and id1 is the SMBH that is removed
    merger_ids = np.loadtxt(filename2, usecols=[2, 4], dtype="int64")
    merger_times = np.loadtxt(filename2, usecols=[1], dtype="float32")

    # calculate
    unique_ids = np.unique(ids)

    for smbh_id in unique_ids:
        # select
        w = np.where(ids == smbh_id)[0]
        time = data[w, 0]
        mass = data[w, 1]
        mdot = data[w, 2]

        # sort by time, remove duplicate entries
        _, inds = np.unique(time, return_index=True)

        print(f"SMBH [{smbh_id:12d}] found [{len(inds):6d} / {len(time):6d}] unique entries.")

        time = time[inds]
        mass = mass[inds]
        mdot = mdot[inds]

        smbhs[int(smbh_id)] = {"time": time, "mass": mass, "mdot": mdot}

        # additional columns
        if ncols > 6:
            w = w[inds]  # sorted, de-duplicated subset
            smbhs[int(smbh_id)]["rho"] = data[w, 3]
            smbhs[int(smbh_id)]["cs"] = data[w, 4]
            smbhs[int(smbh_id)]["hsml"] = data[w, 5]
            smbhs[int(smbh_id)]["ngbmaxdist"] = data[w, 6]
            smbhs[int(smbh_id)]["spin"] = data[w, 7]
            smbhs[int(smbh_id)]["x"] = data[w, 8]
            smbhs[int(smbh_id)]["y"] = data[w, 9]
            smbhs[int(smbh_id)]["z"] = data[w, 10]
            smbhs[int(smbh_id)]["vx"] = data[w, 11]
            smbhs[int(smbh_id)]["vy"] = data[w, 12]
            smbhs[int(smbh_id)]["vz"] = data[w, 13]

    # save
    with h5py.File(cachefile, "w") as f:
        for smbh_id in smbhs.keys():
            for key in smbhs[smbh_id].keys():
                f["%d/%s" % (smbh_id, key)] = smbhs[smbh_id][key]

        # also write mergers
        f["mergers/times"] = merger_times
        f["mergers/ids"] = merger_ids

    print(f"Saved [{cachefile}].")

    # delete decompressed files
    if decompressed:
        assert "txt-files/" in path
        print("Deleting decompressed files...")
        subprocess.run(["rm", "-r", "blackhole_details"], cwd=path)
        subprocess.run(["rm", "-r", "blackhole_mergers"], cwd=path)

    # add mergers to return
    smbhs["mergers"] = {"times": merger_times, "ids": merger_ids}

    return smbhs


def sf_sn_details(sim, overwrite=False):
    """Convert the sf_details/, sf_details_ids/, and sn_details/ files into HDF5."""
    cachefile = {"sf": sim.derivPath + "sf_details.hdf5", "sn": sim.derivPath + "sn_details.hdf5"}

    stars = {}
    supernovae = {}

    # check for existence of cache
    if isfile(cachefile["sf"]) and isfile(cachefile["sn"]) and not overwrite:
        with h5py.File(cachefile["sf"], "r") as f:
            for key in f:
                stars[key] = f[key][()]
        print(f"Loaded [{cachefile['sf']}].")
        with h5py.File(cachefile["sn"], "r") as f:
            for key in f:
                supernovae[key] = f[key][()]
        print(f"Loaded [{cachefile['sn']}].")

        w = np.where((stars["Density"] <= 0) | (stars["Time"] <= 0) | (stars["Temperature"] <= 0))[0]
        if len(w):
            print(f" WARNING: found [{len(w)}] stars with non-positive density, temp, or time, removing.")
            w = np.where((stars["Density"] > 0) & (stars["Time"] > 0) & (stars["Temperature"] > 0))[0]
            for key in stars:
                stars[key] = stars[key][w]

        return stars, supernovae

    # concatenate txt files if needed
    path = sim.simPath + "txt-files/"

    if not isdir(path):
        # txt-files/ is not present yet, run is likely still in progress
        path = sim.simPath

    file_sf = path + "sf_details/sf_details.txt"
    file_sf_ids = path + "sf_details_ids/sf_details_ids.txt"
    file_sn = path + "sn_details/sn_details.txt"

    decompressed = False

    if not isfile(file_sf):
        # .tar.gz files present and directories not? decompress
        if isfile(path + "sf_details.tar.gz") and not isdir(path + "sf_details/"):
            assert "txt-files/" in path  # should not occur otherwise

            print("Decompressing sf_details.tar.gz...")
            subprocess.run(["tar", "-xzf", "sf_details.tar.gz"], cwd=path)
            print("Decompressing sf_details_ids.tar.gz...")
            subprocess.run(["tar", "-xzf", "sf_details_ids.tar.gz"], cwd=path)
            print("Decompressing sn_details.tar.gz...")
            subprocess.run(["tar", "-xzf", "sn_details.tar.gz"], cwd=path)
            decompressed = True

    # concat file is missing, check for individual files, run concat
    if not isfile(file_sf):
        print("Concat sf_details*.txt...")
        subprocess.run("cat sf_details_*.txt > sf_details.txt", cwd=path + "sf_details/", shell=True)

    if not isfile(file_sf_ids):
        print("Concat sf_details_ids*.txt...")
        subprocess.run("cat sf_details_ids_*.txt > sf_details_ids.txt", cwd=path + "sf_details_ids/", shell=True)

    if not isfile(file_sn):
        print("Concat sn_details*.txt...")
        subprocess.run("cat sn_details_*.txt > sn_details.txt", cwd=path + "sn_details/", shell=True)

    # clean: only unique entries (this is star formation, so just one entry per star id)
    def _clean_sort_and_save(dict_in, unique_keys, name):
        """Clean dictionary to only unique entries based on unique_keys, sort by time, and save."""
        dict_out = {}
        unique_ids, unique_inds = np.unique(unique_keys, return_index=True)

        if unique_ids.size < len(unique_keys):
            print(f" {len(unique_keys) - unique_ids.size} duplicate [{name}] entries! Keep first occurrence only.")
            for key in dict_in.keys():
                dict_out[key] = dict_in[key][unique_inds]
        else:
            print(f" No duplicate [{name}] entries found.")
            dict_out = dict_in

        # sort by time (unless sort input explicitly)
        sort_inds = np.argsort(dict_out["Time"])
        for key in dict_out.keys():
            dict_out[key] = dict_out[key][sort_inds]

        dict_out["Coordinates"] = np.vstack((dict_out["pos0"], dict_out["pos1"], dict_out["pos2"])).T
        dict_out["Velocities"] = np.vstack((dict_out["vel0"], dict_out["vel1"], dict_out["vel2"])).T
        for key in ["pos0", "pos1", "pos2", "vel0", "vel1", "vel2"]:
            del dict_out[key]

        # save
        with h5py.File(cachefile[name], "w") as f:
            for key in dict_out:
                f[key] = dict_out[key]

        print(f"Saved [{cachefile[name]}].")

        return dict_out

    # load sf_details, columns:
    # thistask tistep num time pos0 pos1 pos2 vel0 vel1 vel2 dens temp metal mass initialsolomass 0 0
    sf_columns = [
        "Time",
        "pos0",
        "pos1",
        "pos2",
        "vel0",
        "vel1",
        "vel2",
        "Density",
        "Temperature",
        "Metallicity",
        "InitialSoloMass",
    ]

    sf_data = np.loadtxt(file_sf, usecols=np.arange(3, len(sf_columns) + 3), dtype="float32")
    sf = {col: sf_data[:, i] for i, col in enumerate(sf_columns)}
    for i, col in enumerate(["task", "tistep", "num"]):
        sf[col] = np.loadtxt(file_sf, usecols=[i], dtype="int32")

    # restrict to unique entries (must do this independently for sf_details and sf_details_ids as duplicates can differ)
    sf_keys = ["%d-%d-%d" % (sf["task"][i], sf["tistep"][i], sf["num"][i]) for i in range(sf["task"].size)]

    _, unique_inds = np.unique(sf_keys, return_index=True)
    for key in list(sf.keys()):
        sf[key] = sf[key][unique_inds]

    # load sf_details_ids, columns: task tistep index id
    sf_ids_columns = ["task", "tistep", "index", "id"]
    sf_ids_data = np.loadtxt(file_sf_ids, dtype="int64")
    sf_ids = {col: sf_ids_data[:, i] for i, col in enumerate(sf_ids_columns)}

    sf_ids_keys = [
        "%d-%d-%d" % (sf_ids["task"][i], sf_ids["tistep"][i], sf_ids["index"][i]) for i in range(sf_ids["task"].size)
    ]

    _, unique_inds = np.unique(sf_ids_keys, return_index=True)
    for key in list(sf_ids.keys()):
        sf_ids[key] = sf_ids[key][unique_inds]

    # combine sf_details and sf_details_ids
    if sf["task"].size == sf_ids["task"].size:
        for i in range(sf["task"].size):
            assert sf["task"][i] == sf_ids["task"][i]
            assert sf["tistep"][i] == sf_ids["tistep"][i]
            assert sf["num"][i] <= sf_ids["index"][i]

        sf["id"] = sf_ids["id"]

        stars = _clean_sort_and_save(sf, sf["id"], "sf")
    else:
        print("WARNING: sf_details and sf_details_ids have different lengths! Cannot combine. Skipping IDs.")
        sf["id"] = np.zeros(sf["task"].size, dtype="int32") - 1
        sf_keys = np.arange(sf["task"].size)
        stars = _clean_sort_and_save(sf, sf_keys, "sf")

    # load sn_details, columns:
    # id time pos0 pos1 pos2 vel0 vel1 vel2 dens temp metal mass_out energy_out age local_flag
    sn_columns = [
        "Time",
        "pos0",
        "pos1",
        "pos2",
        "vel0",
        "vel1",
        "vel2",
        "Density",
        "Temperature",
        "Metallicity",
        "Mass_SN",
        "Energy_SN",
        "Age",
    ]  # ,'local_flag']

    sn = np.loadtxt(file_sn, usecols=[0], dtype="int64")
    sn = dict.fromkeys(["id"], sn)
    sn_data = np.loadtxt(file_sn, usecols=np.arange(1, len(sn_columns) + 1), dtype="float32")
    for i, col in enumerate(sn_columns):
        sn[col] = sn_data[:, i]

    keys = ["%s-%s" % (id, time) for id, time in zip(sn["id"], sn["Time"])]

    supernovae = _clean_sort_and_save(sn, keys, "sn")

    # delete decompressed files
    if decompressed:
        assert "txt-files/" in path
        print("Deleting decompressed files...")
        subprocess.run(["rm", "-r", "sf_details"], cwd=path)
        subprocess.run(["rm", "-r", "sf_details_ids"], cwd=path)
        subprocess.run(["rm", "-r", "sn_details"], cwd=path)

    return stars, supernovae

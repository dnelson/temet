"""
Processing of ArepoVTK tiled output, to create image pyramids, and support Explorer functionality.
"""

import os

import h5py
import numpy as np

from .pyramid import genColorTable, getColorTableParams, getDataArrayHDF5, loadCPTColorTable, make_pyramid_all


def process():
    """Process ArepoVTK output files into image pyramids for Explorer visualization."""
    # config
    fileBase = "frame_1820"  # frame_455 #frame_1820 #frame_jobtest1
    outPath = "vtkout_szy/"
    totNumJobs = 256  # 16 #64 #256
    tileSize = 65536  # 256 for all normal renders
    reduceOrder = 1  # [1-5], 1=bilinear, 3=cubic
    fieldName = "SzY"  # Density, Temp, Entropy, Metal, VelMag, SzY, XRay
    levelMin = 9  # 0 for all normal renders, 9=max=only the full image level
    makeFullImage = False
    makePyramid = True

    # get initial (hdf5) file listing, size of each, and total image size
    fileList = []

    for curJobNum in range(0, totNumJobs):
        filename = fileBase + "_" + str(curJobNum) + "_" + str(totNumJobs) + ".hdf5"

        fileList.append(filename)

    # calculate pyramid levels (where in the pyramid do the hdf5 tiles sit)
    dummy = getDataArrayHDF5(fileList[0], "Density")
    mDims = np.array(dummy.shape, dtype="int64")

    jobsPerDim = np.int32(np.sqrt(totNumJobs))
    totImgSize = mDims * jobsPerDim
    totImgPx = totImgSize[0] * totImgSize[1]

    if totImgSize[0] != totImgSize[1]:
        print("Error: Expecting square image.")
        return

    print(
        "Final image size: "
        + str(totImgSize[0])
        + "x"
        + str(totImgSize[1])
        + " ("
        + str(totImgPx / 1e6)
        + " MP) (Total jobs: "
        + str(totNumJobs)
        + ", jobs per dim: "
        + str(jobsPerDim)
        + ")"
    )

    # level calculations
    levelMax = np.int32(np.log10(totImgSize[0]) / np.log10(2)) - 8  # totImgSize[0] / 256
    levelNat = np.int32(np.log10(mDims[0]) / np.log10(2)) - 8  # mDims[0] / 256

    print(" Level min: " + str(levelMin))
    print(" Level max: " + str(levelMax))
    print(" Level nat: " + str(levelNat))

    # make pyramid directories / meta file
    if os.path.exists(outPath):
        print("Error: Output location [" + outPath + "] already exists.")
        return

    os.makedirs(outPath)

    # load actual colortable
    ct = getColorTableParams(fieldName)
    ct = loadCPTColorTable(ct)

    config = {
        "ct": ct,
        "zoom": 0.5,
        "zoomGlobal": 0.5,
        "mDims": mDims,
        "jobsPerDim": jobsPerDim,
        "totImgSize": totImgSize,
        "totImgPx": totImgPx,
        "levelMin": levelMin,
        "levelMax": levelMax,
        "levelNat": levelNat,
        "fileList": fileList,
        "fieldName": fieldName,
        "tileSize": tileSize,
        "outPath": outPath,
        "reduceOrder": reduceOrder,
        "makeFullImage": makeFullImage,
        "makePyramid": makePyramid,
    }

    # OPTION (1): make full pyramid (global image allocation)
    make_pyramid_all(config)

    # OPTION (2): make upper and lower pyramids separately (memory efficient)
    # make_pyramid_upper(config)
    # make_pyramid_lower(config)

    # import pdb; pdb.set_trace() #idl stop


def processMark():
    """Process the visualization output of Mark (DM density) into image pyramids for Explorer visualization."""
    # config
    fileBase = "/n/hernquistfs1/mvogelsberger/Illustris-1/explorer/map"
    outPath = "vtkout_dmdens/"
    totNumJobs = 64
    tileSize = 32768  # 256
    reduceOrder = 1  # [1-5], 1=bilinear, 3=cubic
    fieldName = "map"
    levelMin = 9  # 0 for all normal renders, 9=max=only the full image level
    makeFullImage = False
    makePyramid = True

    # get initial (hdf5) file listing, size of each, and total image size
    jobsPerDim = np.int32(np.sqrt(totNumJobs))
    fileList = []

    for curJobNum in range(0, totNumJobs):
        rowNum = jobsPerDim - (curJobNum % jobsPerDim) - 1
        colNum = jobsPerDim - curJobNum / jobsPerDim - 1
        filename = fileBase + "_" + str(rowNum) + "_" + str(colNum) + ".hdf5"

        fileList.append(filename)

    # calculate pyramid levels (where in the pyramid do the hdf5 tiles sit)
    dummy = getDataArrayHDF5(fileList[0], fieldName)
    mDims = np.array(dummy.shape, dtype="int64")

    totImgSize = mDims * jobsPerDim
    totImgPx = totImgSize[0] * totImgSize[1]

    if totImgSize[0] != totImgSize[1]:
        print("Error: Expecting square image.")
        return

    print(
        "Final image size: "
        + str(totImgSize[0])
        + "x"
        + str(totImgSize[1])
        + " ("
        + str(totImgPx / 1e6)
        + " MP) (Total jobs: "
        + str(totNumJobs)
        + ", jobs per dim: "
        + str(jobsPerDim)
        + ")"
    )

    levelMax = np.int32(np.log10(totImgSize[0]) / np.log10(2)) - 8  # totImgSize[0] / 256
    levelNat = np.int32(np.log10(mDims[0]) / np.log10(2)) - 8  # mDims[0] / 256

    print(" Level min: " + str(levelMin))
    print(" Level max: " + str(levelMax))
    print(" Level nat: " + str(levelNat))

    # make pyramid directories / meta file
    if os.path.exists(outPath):
        print("Error: Output location [" + outPath + "] already exists.")
        return

    os.makedirs(outPath)

    # load actual colortable
    ct = genColorTable(fieldName)

    config = {
        "ct": ct,
        "zoom": 0.5,
        "zoomGlobal": 0.5,
        "mDims": mDims,
        "jobsPerDim": jobsPerDim,
        "totImgSize": totImgSize,
        "totImgPx": totImgPx,
        "levelMin": levelMin,
        "levelMax": levelMax,
        "levelNat": levelNat,
        "fileList": fileList,
        "fieldName": fieldName,
        "tileSize": tileSize,
        "outPath": outPath,
        "reduceOrder": reduceOrder,
        "makeFullImage": makeFullImage,
        "makePyramid": makePyramid,
    }

    # OPTION (2): make upper and lower pyramids separately (memory efficient)
    make_pyramid_all(config)
    # make_pyramid_upper(config)
    # make_pyramid_lower(config)


def processShy():
    """Process the stellar light visualization output of Shy into image pyramids for Explorer visualization."""
    # config 1820 128k
    # fileBase = "/n/ghernquist/sgenel/Illustris/stellar_images/L75n1820FP/L75n1820FP_s135_75000_75000_kpc_572_pc_tile"
    # fileEnding = '_RGBmat_scaled.hdf5'
    # fileBase = "/n/ghernquist/sgenel/Illustris/stellar_images/L75n1820FP/L75n1820FP_s135_75000_75000_kpc_286_pc_tile"
    # fileEnding = '_RGBmat.hdf5'

    # config TNG
    fileBase = "/home/extdylan/data/frames/shy/L205n2500TNG_s099_205000_205000_kpc_1564_pc_tile"
    fileEnding = "_RGBmat_scaled.hdf5"

    fieldName = "RGB"
    nThirdDims = 3  # 3=RGB, 4=RGBA
    outPath = "/home/extdylan/data/frames/tng_starlight/"
    totNumJobs = 64  # 16 #64 #256
    tileSize = 256
    reduceOrder = 1  # [1-5], 1=bilinear, 3=cubic
    levelMin = 0  # 0 for all normal renders, 9=max=only the full image level
    makeFullImage = False
    makePyramid = True

    # get initial (hdf5) file listing, size of each, and total image size
    fileList = []

    for curJobNum in range(0, totNumJobs):
        filename = fileBase + "_" + str(curJobNum) + fileEnding
        fileList.append(filename)

    # calculate pyramid levels (where in the pyramid do the hdf5 tiles sit)
    dummy = getDataArrayHDF5(fileList[4], fieldName)
    mDims = np.array(dummy.shape[1:3], dtype="int64")

    jobsPerDim = np.int32(np.sqrt(totNumJobs))
    totImgSize = np.zeros((3,), dtype="int64")
    totImgSize[0:2] = mDims * jobsPerDim
    totImgSize[2] = nThirdDims
    totImgPx = totImgSize[0] * totImgSize[1]

    if totImgSize[0] != totImgSize[1]:
        print("Error: Expecting square image.", totImgSize)
        return

    print(
        f"Final image size: {totImgSize} ({totImgPx / 1e6} MP) (Total jobs: {totNumJobs}, jobs per dim: {jobsPerDim})"
    )

    levelMax = np.int32(np.log10(totImgSize[0]) / np.log10(2)) - 8  # totImgSize[0] / 256
    levelNat = np.int32(np.log10(mDims[0]) / np.log10(2)) - 8  # mDims[0] / 256

    print(" Level min: " + str(levelMin))
    print(" Level max: " + str(levelMax))
    print(" Level nat: " + str(levelNat))

    # make pyramid directories / meta file
    if os.path.exists(outPath):
        print("Error: Output location [" + outPath + "] already exists.")
        return

    os.makedirs(outPath)

    config = {
        "ct": None,
        "zoom": (1.0, 0.5, 0.5),
        "zoomGlobal": (0.5, 0.5, 1.0),
        "nThirdDims": nThirdDims,
        "mDims": mDims,
        "jobsPerDim": jobsPerDim,
        "totImgSize": totImgSize,
        "totImgPx": totImgPx,
        "levelMin": levelMin,
        "levelMax": levelMax,
        "levelNat": levelNat,
        "fileList": fileList,
        "fieldName": fieldName,
        "tileSize": tileSize,
        "outPath": outPath,
        "reduceOrder": reduceOrder,
        "makeFullImage": makeFullImage,
        "makePyramid": makePyramid,
    }

    # OPTION (2): make upper and lower pyramids separately (memory efficient)
    make_pyramid_all(config)
    # make_pyramid_upper(config)
    # make_pyramid_lower(config)


def expandedJobNums(jobNum, totNumJobs=256, expansionFac=16):
    """Calculate the expanded job numbers corresponding to a parent ArepoVTK job."""
    # make expanded job matrix, then take square cutout
    jobMatrix = np.arange(totNumJobs * expansionFac)
    jobMatrix = np.reshape(jobMatrix, (np.sqrt(jobMatrix.shape), np.sqrt(jobMatrix.shape)))

    jobsPerDim = np.sqrt(totNumJobs)
    expPerDim = np.sqrt(expansionFac)

    xInd = np.int32(np.floor(jobNum / jobsPerDim))
    yInd = np.int32(np.floor(jobNum % jobsPerDim))

    jobCutout = jobMatrix[xInd * expPerDim : (xInd + 1) * expPerDim, yInd * expPerDim : (yInd + 1) * expPerDim]
    jobCutout = np.reshape(jobCutout, expansionFac)

    print("#SBATCH --array=" + ",".join([str(num) for num in jobCutout]) + " #j" + str(jobNum))
    print("-j " + str(jobNum) + " -e ${SLURM_ARRAY_TASK_ID}")

    return jobCutout


def combineExpandedJob(jobNum):
    """Combine the expanded ArepoVTK job outputs into a single file for a given parent job number."""
    # config
    totNumJobs = 256  # original number of jobs
    expansionFac = 16  # expanded jobs per job
    fileBase = "frame_1820"  # filenames of expanded job output
    outPath = "combined/"  # path to write combined file to
    fields = ["Density", "Entropy", "Metal", "SzY", "Temp", "VelMag", "XRay"]

    # get expanded job numbers
    expJobNums = expandedJobNums(jobNum, totNumJobs, expansionFac)
    expFacPerDim = np.sqrt(expansionFac)

    # get initial (hdf5) file listing, size of each, and total combined size
    fileList = []

    for jj in expJobNums:
        filename = fileBase + "_" + str(jj) + "_" + str(totNumJobs * expansionFac) + ".hdf5"
        fileList.append(filename)

    dummy = getDataArrayHDF5(fileList[1], fields[0])
    mDims = np.array(dummy.shape, dtype="int64")
    totImgSize = mDims * expFacPerDim

    # make pyramid directories / meta file
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    outFileName = outPath + fileBase + "_" + str(jobNum) + "_" + str(totNumJobs) + ".hdf5"

    f = h5py.File(outFileName, "w")

    # loop over each field
    for fieldName in fields:
        globalArray = np.zeros(totImgSize, dtype="float32")
        alloc_gb_str = str((float(totImgSize[0]) * totImgSize[1] * 4) / 1024 / 1024 / 1024)
        print(" -- " + fieldName + ": Allocating... [" + alloc_gb_str.format("%.2f") + " GB]")

        for i in range(len(fileList)):
            # load
            array = getDataArrayHDF5(fileList[i], fieldName)

            if len(array) == 0:
                continue

            # stamp
            jRow = (expFacPerDim - 1) - np.int32(np.floor(i % expFacPerDim))
            jCol = np.int32(np.floor(i / expFacPerDim))

            x0 = mDims[0] * jCol
            x1 = mDims[0] * (jCol + 1)
            y0 = mDims[1] * jRow
            y1 = mDims[1] * (jRow + 1)

            globalArray[x0:x1, y0:y1] = array

        # save field
        group = f.create_group(fieldName)
        group.create_dataset("Array", data=globalArray)

    f.close()

    print("Wrote: " + outFileName)


def makeColorBars():
    """Create a PDF with all colormaps used in ArepoVTK for reference."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fieldNames = ["Density", "Temp", "Entropy", "Metal", "VelMag", "SzY", "XRay"]
    cmap_dens = 512  # 1024

    outNames = [
        r"Gas Density [log 10$^{10}$ M$_\odot$ h$^2$ / kpc$^2$]",
        r"Gas Temperature [log K]",
        r"Gas Entropy [log K/cm$^2]$",
        r"Gas Metallicity [log (0.0127 * Z$_\odot$)]",
        r"Gas Velocity [km/s]",
        r"Sunyaev Zeldovich y-parameter [log relative units]",
        r"X-Ray Luminosity [log (relative units)]",
        r"Dark Matter Density [log 10$^{10}$ M$_\odot$ h$^2$ / kpc$^2$]",
        r"Stellar Luminosity [log L$_\odot$ / kpc$^2$]",
    ]

    fig = plt.figure(figsize=(8, 1.5 * (len(fieldNames) + 2)))

    # load actual colortable
    for i, fieldName in enumerate(fieldNames):
        fieldName = fieldNames[i]
        ct = getColorTableParams(fieldName)
        ct = loadCPTColorTable(ct)

        # raw cmap
        # rgb = np.vstack( (ct['table']['r'], ct['table']['g'], ct['table']['b']) )
        # rgb = np.transpose( rgb )
        # cmap = colors.ListedColormap( rgb )

        # use cmap as input for LinearSegmented to make it smooth
        cdict_r = [(xx, ct["table"]["r"][i], ct["table"]["r"][i]) for i, xx in enumerate(ct["table"]["x"])]
        cdict_g = [(xx, ct["table"]["g"][i], ct["table"]["g"][i]) for i, xx in enumerate(ct["table"]["x"])]
        cdict_b = [(xx, ct["table"]["b"][i], ct["table"]["b"][i]) for i, xx in enumerate(ct["table"]["x"])]
        cdict = {"red": cdict_r, "green": cdict_g, "blue": cdict_b}
        cmap_ls = colors.LinearSegmentedColormap("cmap_ls", cdict, cmap_dens)
        cmap_norm = colors.Normalize(vmin=ct["minmax"][0], vmax=ct["minmax"][1])

        # import pdb; pdb.set_trace()
        axis = fig.add_subplot(len(fieldNames) + 2, 1, i + 1)
        cb1 = mpl.colorbar.ColorbarBase(axis, cmap=cmap_ls, norm=cmap_norm, orientation="horizontal")
        cb1.set_label(outNames[i])

    # DM density
    cdict = {
        "red": ((0.0, 0.0, 0.0), (0.3, 0, 0), (0.6, 0.8, 0.8), (1.0, 1.0, 1.0)),
        "green": ((0.0, 0.0, 0.0), (0.3, 0.3, 0.3), (0.6, 0.4, 0.4), (1.0, 1.0, 1.0)),
        "blue": ((0.0, 0.05, 0.05), (0.3, 0.5, 0.5), (0.6, 0.6, 0.6), (1.0, 1.0, 1.0)),
    }

    ct = genColorTable("map")
    dmdens_cmap = colors.LinearSegmentedColormap("dmdens_cmap", cdict, cmap_dens)
    cmap_norm = colors.Normalize(vmin=ct["minmax"][0], vmax=ct["minmax"][1])

    axis = fig.add_subplot(len(fieldNames) + 2, 1, len(fieldNames) + 1)
    cb1 = mpl.colorbar.ColorbarBase(axis, cmap=dmdens_cmap, norm=cmap_norm, orientation="horizontal")
    cb1.set_label(outNames[-2])

    # stellar composite
    cdict = {
        "red": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        "green": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        "blue": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    }

    stellar_minmax = [6.0, 10.0]  # based on Shy's make_stars_image.m huge_image_mode==1
    stars_cmap = colors.LinearSegmentedColormap("stars_cmap", cdict, cmap_dens)
    cmap_norm = colors.Normalize(vmin=stellar_minmax[0], vmax=stellar_minmax[1])

    axis = fig.add_subplot(len(fieldNames) + 2, 1, len(fieldNames) + 2)
    cb1 = mpl.colorbar.ColorbarBase(axis, cmap=stars_cmap, norm=cmap_norm, orientation="horizontal")
    cb1.set_label(outNames[-1])

    # save
    plt.savefig("test.pdf")
    plt.close()


def _get_cmap_colors(ctName):
    """Handle LinearSegmentedColormap by sampling it."""
    from ..util.helper import loadColorTable

    rgb = loadColorTable(ctName, valMinMax=[12.0, 22.0])  # dummy range for custom tables

    if not hasattr(rgb, "colors"):
        rgb.colors = []
        for i in range(256):
            rgb.colors.append(rgb(i))

    return np.array(rgb.colors), len(rgb.colors)


def writeColorTable(ctName="inferno"):
    """Output a .tbl discrete color table for use in ArepoVTK, from one that loadColorTable() knows."""
    start = 0
    alpha = 1.0  # constant
    filename = "mpl_%s.tbl" % ctName

    # load
    rgb, nVals = _get_cmap_colors(ctName)

    # convert [0,1] to ArepoVTK input of [0,255] if necessary
    fac = 1.0
    if np.max(rgb) <= 1.0:
        fac = 255.0

    # write
    with open(filename, "w") as f:
        f.write("# comment\n")
        f.write("%d\n" % (nVals - start))

        for i in range(nVals):
            f.write("%5.1f %5.1f %5.1f %4.2f\n" % (rgb[i][0] * fac, rgb[i][1] * fac, rgb[i][2] * fac, alpha))

    print("Wrote: [%s]" % filename)


def writeColormapPNGs():
    """Output .png images of colormaps for loading as textures in three.js / Explorer3D volume rendering."""
    from png import Writer

    from ..plot.util import validColorTableNames

    ctNames = [ctName for ctName in validColorTableNames() if "_r" not in ctName and ctName[-1] != "0"]
    for ctName in ["HI_segmented", "H2_segmented"]:
        ctNames.remove(ctName)

    for ctName in ctNames:
        print(ctName)

        # load and write
        rgb, nVals = _get_cmap_colors(ctName)
        rgb = np.floor(rgb * 255)

        # save PNG
        if rgb.shape[1] == 4:
            rgb = rgb[:, 0:3]  # discard alpha channel (always unity)

        with open(ctName.lower() + ".png", "wb") as img:
            writer = Writer(rgb.shape[0], 1, alpha=False, bitdepth=8)  # 1 is height
            writer.write(img, np.reshape(rgb, (-1, rgb.shape[0] * 1 * rgb.shape[1])))

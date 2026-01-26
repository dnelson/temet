"""
ArepoVTK/Web: building image pyramids.
"""

import colorsys
import os.path

import h5py
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import scipy.ndimage


try:
    from png import Reader as pngReader
    from png import Writer as pngWriter
except ImportError:
    print("png module not found! Please install pypng (pip install git+https://gitlab.com/drj11/pypng).")


def rebin(a, shape):
    """See REBIN() IDL function."""
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def _reduceArray(arrayIn, numReductions, config):
    newShape = np.array(arrayIn.shape)
    tempArray = arrayIn

    for _j in range(numReductions):
        newShape[1] /= 2
        newShape[2] /= 2

        array = np.zeros(newShape)  # has different size, cannot write directly over

        for k in range(config["nThirdDims"]):
            array[k, :, :] = rebin(tempArray[k, :, :], [newShape[1], newShape[2]])

        tempArray = array  # override

    array = array.round().astype("uint8").clip(0, 255)
    return array


def _reduceArray2D(arrayIn, numReductions):
    newShape = np.array(arrayIn.shape)
    tempArray = arrayIn

    for _j in range(numReductions):
        newShape /= 2
        array = np.zeros(newShape)
        array = rebin(tempArray, newShape)

        tempArray = array
    return tempArray


def getDataArrayHDF5(filename, fieldName):
    """Load a single dataset from a HDF5 file with read_direct()."""
    if not os.path.isfile(filename):
        print(" " + filename + " -- MISSING")
        return []

    # open file, get data dimensions
    print(" " + filename)

    f = h5py.File(filename, "r")

    if fieldName == "RGB":
        # named datasets without a parent group
        dataset = f[fieldName]
        dtype = np.uint8
    elif fieldName == "map":
        # named datasets without a parent group
        dataset = f[fieldName]
        dtype = np.float32
    else:
        # generic named datasets within a named parent group
        group = f.get(fieldName)
        dataset = group.get("Array")
        dtype = np.float32

    dims = dataset.shape

    # allocate
    array = np.zeros(dims, dtype=dtype)

    # read and return
    dataset.read_direct(array)

    f.close()

    # array_mm = [ np.min(array), np.max(array) ]
    # print("  min: " + str(array_mm[0]) + " max: " + str(array_mm[1]));

    return array


def _convertDataArrayToRGB(array, colortable):
    if colortable is None:
        # fix to byte and immediate return
        return array

    if colortable["tableNum"] == 0:
        # using matplotlib cmap
        array_rgba = colortable["table"].to_rgba(array)
        array_rgb = (array_rgba[:, :, 0:3] * 255.0).round()
        # array_rgb = colortable['table'].to_rgb( array )
        # array_rgb = (array_rgb * 255.0).round()
        array_rgb = array_rgb.clip(0, 255).astype("uint8")
        # import pdb; pdb.set_trace()
        return array_rgb

    # take log
    dims = array.shape

    # convert float32 -> (R,G,B) tuples
    array_rgb = np.zeros((dims[0], dims[1], 3), dtype=np.uint8)

    # apply CT
    x = colortable["table"]["x"]
    r = colortable["table"]["r"]
    g = colortable["table"]["g"]
    b = colortable["table"]["b"]

    arr_f = (array - colortable["minmax"][0]) / (colortable["minmax"][1] - colortable["minmax"][0])

    array_rgb[:, :, 0] = np.clip(np.round(np.interp(arr_f, x, r) * 255.0), 0, 255)
    array_rgb[:, :, 1] = np.clip(np.round(np.interp(arr_f, x, g) * 255.0), 0, 255)
    array_rgb[:, :, 2] = np.clip(np.round(np.interp(arr_f, x, b) * 255.0), 0, 255)

    return array_rgb


def loadCPTColorTable(ct):
    """Load a color table in CPT format."""
    f = open(ct["file"])
    lines = f.readlines()
    f.close()

    x = []
    r = []
    g = []
    b = []
    colorModel = "RGB"

    # parse
    for l in lines:
        ls = l.split()

    for l in lines:
        ls = l.split()
        if l[0] == "#":
            if ls[-1] == "HSV":
                colorModel = "HSV"
                continue
            else:
                continue
        if ls[0] == "B" or ls[0] == "F" or ls[0] == "N":
            pass
        else:
            x.append(float(ls[0]))
            r.append(float(ls[1]))
            g.append(float(ls[2]))
            b.append(float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

    x.append(xtemp)
    r.append(rtemp)
    g.append(gtemp)
    b.append(btemp)

    nTable = len(r)
    x = np.array(x, np.float32)
    r = np.array(r, np.float32)
    g = np.array(g, np.float32)
    b = np.array(b, np.float32)

    if colorModel == "HSV":
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360.0, g[i], b[i])
            r[i] = rr
            g[i] = gg
            b[i] = bb
    if colorModel == "HSV":
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360.0, g[i], b[i])
            r[i] = rr
            g[i] = gg
            b[i] = bb
    if colorModel == "RGB":
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
    xNorm = (x - x[0]) / (x[-1] - x[0])

    # reverse?
    if ct["reverse"]:
        r = r[::-1]
        g = g[::-1]
        b = b[::-1]

    # gamma scaling?
    if ct["gamma"] != 1.0:
        xNorm = np.power(xNorm, ct["gamma"])

    red = []
    blue = []
    green = []

    for i in range(len(x)):
        red.append([xNorm[i], r[i], r[i]])
        green.append([xNorm[i], g[i], g[i]])
        blue.append([xNorm[i], b[i], b[i]])

    # colorDict = {"red":red, "green":green, "blue":blue}
    table = {"x": xNorm, "r": r, "g": g, "b": b}

    # add actual table to ct
    ct["table"] = table
    ct["tableNum"] = nTable

    return ct


def genColorTable(fieldName):
    """Generate a color table for a given field name."""
    if fieldName != "map":
        print("Error!")
        return

    ct = {}
    ct["log"] = True
    ct["gamma"] = 1.0
    ct["reverse"] = False
    ct["minmax"] = [4.75, 8.9]
    ct["tableNum"] = 0

    cdict = {
        "red": ((0.0, 0.0, 0.0), (0.3, 0, 0), (0.6, 0.8, 0.8), (1.0, 1.0, 1.0)),
        "green": ((0.0, 0.0, 0.0), (0.3, 0.3, 0.3), (0.6, 0.4, 0.4), (1.0, 1.0, 1.0)),
        "blue": ((0.0, 0.05, 0.05), (0.3, 0.5, 0.5), (0.6, 0.6, 0.6), (1.0, 1.0, 1.0)),
    }

    dmdens_cmap = colors.LinearSegmentedColormap("dmdens_cmap", cdict, 1024)
    cmap_norm = colors.Normalize(vmin=ct["minmax"][0], vmax=ct["minmax"][1])
    cmap = cmx.ScalarMappable(norm=cmap_norm, cmap=dmdens_cmap)

    ct["table"] = cmap

    return ct


def saveImageToPNG(filename, array_rgb):
    """Write a raw RGB array into a PNG image file."""
    dims = array_rgb.shape
    img = open(filename, "wb")

    # open image and write
    if dims[2] == 3:
        writer = pngWriter(dims[0], dims[1], alpha=False, bitdepth=8)
        writer.write(img, np.reshape(array_rgb, (-1, dims[0] * dims[2])))

    if dims[2] == 4:
        writer = pngWriter(dims[0], dims[1], alpha=True, bitdepth=8)
        writer.write(img, np.reshape(array_rgb, (-1, dims[0] * dims[2])))

    img.close()


def getColorTableParams(fieldName):
    """Set colortable and its parameters, for a given field name."""
    ctBase = os.path.expanduser("~") + "/idl/mglib/vis/cpt-city/"

    if fieldName == "Density":
        ct = {
            "file": ctBase + "ncl/WhiteBlueGreenYellowRed-dnA.cpt",
            "minmax": [-6.0, -1.4],  # [-5.5,-2.7] for 1820_16k_nni/cellgrad # original: [-4.7,-1.6]
            "log": True,
            "reverse": True,
            "gamma": 1.3,
        }

    if fieldName == "Temp":
        ct = {"file": ctBase + "kst/33_blue_red.cpt", "minmax": [3.0, 7.2], "log": True, "reverse": False, "gamma": 1.2}

    if fieldName == "Entropy":
        ct = {
            "file": ctBase + "pm/f-23-28-3-dnB.cpt",
            "minmax": [9.2, 12.5],  # originally [8.0,11.2]
            "log": True,
            "reverse": False,
            "gamma": 2.0,
        }

    if fieldName == "Metal":
        ct = {
            "file": ctBase + "wkp/tubs/nrwc.cpt",
            "minmax": [-5.0, -1.4],  # explorer_old: [-5.0,-1.4]
            "log": True,
            "reverse": False,
            "gamma": 1.0,
        }

    if fieldName == "VelMag":
        ct = {
            "file": ctBase + "pm/f-34-35-36.cpt",
            "minmax": [50.0, 960.0],
            "log": False,
            "reverse": False,
            "gamma": 1.0,
        }

    if fieldName == "SzY":
        ct = {
            "file": ctBase + "oc/zeu.cpt",
            "minmax": [-2.5, 4.7],  # originally [2.5,5.6]
            "log": True,
            "reverse": True,
            "gamma": 0.5,
        }

    if fieldName == "XRay":
        ct = {
            "file": ctBase + "kst/03_red_temperature.cpt",
            "minmax": [-12.0, -2.5],  # explorer_old: [-12.0,-2.5], originally [-7.6,-2.5]
            "log": True,
            "reverse": False,
            "gamma": 1.5,
        }

    return ct


def make_pyramid_all(config):
    """Create all sub-images for an image pyramid."""
    # allocate
    if config["totImgSize"].shape[0] == 2:
        alloc_gb_str = str((float(config["totImgSize"][0]) * config["totImgSize"][1] * 4) / 1024 / 1024 / 1024)
        globalArray = np.zeros(config["totImgSize"], dtype="float32")
    else:
        alloc_gb_str = str((float(config["totImgSize"][0]) * config["totImgSize"][1] * 3 * 1) / 1024 / 1024 / 1024)
        globalArray = np.zeros(config["totImgSize"], dtype="uint8")

    print("Allocating... [" + alloc_gb_str.format("%.2f") + " GB]")

    # load: loop over hdf5 files for global min/max
    print("Loading...")

    for i in range(len(config["fileList"])):
        # load
        array = getDataArrayHDF5(config["fileList"][i], config["fieldName"])

        if len(array) == 0:
            continue

        # stamp
        jRow = (config["jobsPerDim"] - 1) - np.int32(np.floor(i % config["jobsPerDim"]))
        jCol = np.int32(np.floor(i / config["jobsPerDim"]))

        x0 = config["mDims"][0] * jCol
        x1 = config["mDims"][0] * (jCol + 1)
        y0 = config["mDims"][1] * jRow
        y1 = config["mDims"][1] * (jRow + 1)

        if len(array.shape) == 3:
            for j in range(3):
                globalArray[x0:x1, y0:y1, j] = np.transpose(array[j, :, :])
        else:
            globalArray[x0:x1, y0:y1] = array

    # shy for TNG
    if 1:
        print("FLIPPING SECOND AXIS")
        globalArray = np.flip(globalArray, 0)

    # set all zeros to minimum non-zero value and take log (be careful about memory usage)
    if config["ct"] is not None:
        min_val = np.min(globalArray[globalArray > 0.0])
        globalArray[globalArray <= 0.0] = min_val

        if config["ct"]["log"]:
            globalArray = np.log10(globalArray)

    array_mm = [np.min(globalArray), np.max(globalArray)]
    print(" Global min: " + str(array_mm[0]) + " max: " + str(array_mm[1]))
    # loop over levels, starting at lowest (256px tiles)
    for level in range(config["levelMax"], config["levelMin"] - 1, -1):
        # if not at lowest (first iteration), downsize array by half its current value
        print("Level: " + str(level))

        if level != config["levelMax"]:
            print(" downsizing...")

            globalArray = scipy.ndimage.zoom(globalArray, config["zoomGlobal"], order=config["reduceOrder"])

        if config["makeFullImage"]:
            # save full image at this zoom level
            dens_rgb = _convertDataArrayToRGB(globalArray, config["ct"])
            saveImageToPNG(config["outPath"] + "full_" + str(level) + ".png", dens_rgb)

        # rasterize each to PNG, apply colortable, and save
        if config["makePyramid"]:
            print(" chunking...")
            os.makedirs(config["outPath"] + str(level))

            # slice array into 256x256 segments
            nSub = (globalArray.shape)[0] / config["tileSize"]

            for colIndex in range(nSub):
                os.makedirs(config["outPath"] + str(level) + "/" + str(colIndex))
                print("  col [" + str(colIndex + 1) + "] of [" + str(nSub) + "]")

                for rowIndex in range(nSub):
                    saveFilename = str(level) + "/" + str(colIndex) + "/" + str(rowIndex) + ".png"

                    # get chunk (TMS indexing convention)
                    x0 = ((nSub - 1) - rowIndex) * config["tileSize"]
                    x1 = ((nSub - 1) - rowIndex + 1) * config["tileSize"]
                    y0 = (colIndex) * config["tileSize"]
                    y1 = (colIndex + 1) * config["tileSize"]

                    if len(globalArray.shape) == 3:
                        array = globalArray[x0:x1, y0:y1, :]
                    else:
                        array = globalArray[x0:x1, y0:y1]

                    array_rgb = _convertDataArrayToRGB(array, config["ct"])
                    array = None
                    saveImageToPNG(config["outPath"] + saveFilename, array_rgb)


def make_pyramid_upper(config):
    """Make upper sections of an image pyramid (reducing size from tiled images)."""
    print("\nUpper pyramid:")

    sizePerDim = config["tileSize"] * config["jobsPerDim"]

    # allocate
    if config["totImgSize"].shape[0] == 2:
        globalArray = np.zeros((sizePerDim, sizePerDim), dtype="float32")
    else:
        globalArray = np.zeros((sizePerDim, sizePerDim, config["nThirdDims"]), dtype="uint8") + 255
        min_val = 0

    numReductions = np.log10(config["mDims"][0] / config["tileSize"]) / np.log10(2)
    numReductions = np.int32(np.round(numReductions))
    print("Number of reductions from natural tiles: [" + str(numReductions) + "]")

    # load: loop over hdf5 files, downsize each to tileSize and stamp in
    print("Loading...")

    for i in range(len(config["fileList"])):
        # load
        array = getDataArrayHDF5(config["fileList"][i], config["fieldName"])

        if len(array) == 0:
            continue

        # resize down to tileSize x tileSize
        if len(array.shape) == 3:  # Shy
            array = _reduceArray(array, numReductions, config)
        elif config["ct"]["tableNum"] == 0:  # Mark
            array = _reduceArray2D(array, numReductions)
            array = np.rot90(array)
        else:
            # OLD: scipy crap edge effects some bug WTF /Patrik style (probably only works for nat=5)
            for _j in range(numReductions):
                array = scipy.ndimage.zoom(array, config["zoom"], order=config["reduceOrder"])

        # stamp
        jRow = (config["jobsPerDim"] - 1) - np.int32(np.floor(i % config["jobsPerDim"]))
        jCol = np.int32(np.floor(i / config["jobsPerDim"]))

        x0 = config["tileSize"] * jCol
        x1 = config["tileSize"] * (jCol + 1)
        y0 = config["tileSize"] * jRow
        y1 = config["tileSize"] * (jRow + 1)

        if len(array.shape) == 3:
            for j in range(config["nThirdDims"]):
                globalArray[x0:x1, y0:y1, j] = np.transpose(array[j, :, :])
        else:
            globalArray[x0:x1, y0:y1] = array

    # set all zeros to minimum non-zero value and take log
    if config["ct"] is not None:
        if config["ct"]["tableNum"]:
            min_val = np.min(globalArray[globalArray > 0.0])
        else:
            min_val = 1.0  # Mark

        globalArray[globalArray <= 0.0] = min_val

        if config["ct"]["log"]:
            globalArray = np.log10(globalArray)

    array_mm = [min_val, np.max(globalArray)]

    print(" Global min: " + str(array_mm[0]) + " max: " + str(array_mm[1]))
    config["globalMin"] = array_mm[0]  # store for lower pyramid!
    # import pdb; pdb.set_trace()

    # render out native level up to level 0 by progressively downsizing
    startLevel = np.int32(np.log10(sizePerDim) / np.log10(2)) - 8

    for level in range(startLevel, config["levelMin"] - 1, -1):
        print("Level: " + str(level))

        if level != startLevel:
            print(" downsizing...")
            # probably 1px on right side / bottom is lost due to bug here
            globalArray = scipy.ndimage.zoom(globalArray, config["zoomGlobal"], order=config["reduceOrder"])

        if config["makeFullImage"]:
            # save full image at this zoom level
            dens_rgb = _convertDataArrayToRGB(globalArray, config["ct"])
            saveImageToPNG(config["outPath"] + "full_" + str(level) + ".png", dens_rgb)

        if not config["makePyramid"]:
            continue

        # rasterize each to PNG, apply colortable, and save
        print(" chunking...")
        os.makedirs(config["outPath"] + str(level))

        # slice array into 256x256 segments
        nSub = (globalArray.shape)[0] / config["tileSize"]

        for colIndex in range(nSub):
            os.makedirs(config["outPath"] + str(level) + "/" + str(colIndex))
            print("  col [" + str(colIndex + 1) + "] of [" + str(nSub) + "]")

            for rowIndex in range(nSub):
                saveFilename = str(level) + "/" + str(colIndex) + "/" + str(rowIndex) + ".png"

                # get chunk (TMS indexing convention)
                x0 = ((nSub - 1) - rowIndex) * config["tileSize"]
                x1 = ((nSub - 1) - rowIndex + 1) * config["tileSize"]
                y0 = (colIndex) * config["tileSize"]
                y1 = (colIndex + 1) * config["tileSize"]

                if len(globalArray.shape) == 3:
                    array = globalArray[x0:x1, y0:y1, :]
                else:
                    array = globalArray[x0:x1, y0:y1]

                array_rgb = _convertDataArrayToRGB(array, config["ct"])
                saveImageToPNG(config["outPath"] + saveFilename, array_rgb)


def make_pyramid_lower(config):
    """Make lower sections of an image pyramid."""
    print("\nLower pyramid:")

    sizePerDim = config["tileSize"] * config["jobsPerDim"]

    startLevel = np.int32(np.log10(sizePerDim) / np.log10(2)) - 8 + 1
    numReductions = config["levelMax"] - startLevel

    print("Number of reductions from natural tiles: [" + str(numReductions) + "]")

    for i in range(len(config["fileList"])):
        # load
        array = getDataArrayHDF5(config["fileList"][i], config["fieldName"])

        if len(array) == 0:
            continue

        if len(array.shape) == 3:  # processShy
            for j in range(config["nThirdDims"]):
                array[j, :, :] = np.transpose(array[j, :, :])

        # set all zeros to minimum non-zero value and take log
        if config["ct"] is not None:
            array[array <= 0.0] = config["globalMin"]

            if config["ct"]["log"]:
                array = np.log10(array)

            if config["ct"]["tableNum"] == 0:
                array = np.rot90(array)  # Mark

        for level in range(config["levelMax"], startLevel - 1, -1):
            print("  level [" + str(level) + "]")

            if level != config["levelMax"]:
                if len(array.shape) == 3:
                    array = _reduceArray(array, 1, config)
                elif config["ct"]["tableNum"] == 0:  # Mark
                    array = _reduceArray2D(array, 1)
                else:
                    # OLD: scipy crap edge effects some bug WTF /Patrik style (probably only works for nat=5)
                    array = scipy.ndimage.zoom(array, config["zoom"], order=config["reduceOrder"])

            # DEBUG:
            # if level == 8 or level == 9:
            #    continue
            # END DEBUG

            # global indices at this level
            levelDepth = level - startLevel + 1
            levelExpansionFac = 2**levelDepth

            jRow = (config["jobsPerDim"] - 1) - np.int32(np.floor(i / config["jobsPerDim"]))
            jCol = (config["jobsPerDim"] - 1) - np.int32(np.floor(i % config["jobsPerDim"]))

            if i == 0 and not os.path.exists(config["outPath"] + str(level)):
                os.makedirs(config["outPath"] + str(level))

            # slice array into 256x256 segments
            nSub = (array.shape)[-1] / config["tileSize"]

            for colIndex in range(nSub):
                globalCol = jCol * levelExpansionFac + colIndex
                outDirPath = config["outPath"] + str(level) + "/" + str(globalCol)
                if not os.path.exists(outDirPath):
                    os.makedirs(outDirPath)

                for rowIndex in range(nSub):
                    # need to transform local col,row indices into global indices
                    globalRow = jRow * levelExpansionFac + rowIndex

                    # get subaray chunk (TMS indexing convention)
                    x0 = ((nSub - 1) - rowIndex) * config["tileSize"]
                    x1 = ((nSub - 1) - rowIndex + 1) * config["tileSize"]
                    y0 = (colIndex) * config["tileSize"]
                    y1 = (colIndex + 1) * config["tileSize"]

                    if len(array.shape) == 3:
                        subarray = array[0 : config["nThirdDims"], x0:x1, y0:y1]
                        subarray = np.rollaxis(subarray, 0, 3)  # shift (3,N,N) to (N,N,3)
                    else:
                        subarray = array[x0:x1, y0:y1]

                    # save
                    saveFilename = str(level) + "/" + str(globalCol) + "/" + str(globalRow) + ".png"

                    array_rgb = _convertDataArrayToRGB(subarray, config["ct"])
                    saveImageToPNG(config["outPath"] + saveFilename, array_rgb)

        # all levels done for this hdf5 file, move on to next
    print("\nDone.")


def _shuffle4imgs(fname, fname_out):
    # setup
    r = pngReader(filename=fname)
    columnCount, rowCount, pngData, metaData = r.asDirect()

    dims = rowCount
    half = dims / 2

    # read
    image_2d = np.zeros((dims, 3 * dims), dtype=np.uint8)

    for row_index, one_boxed_row_flat_pixels in enumerate(pngData):
        image_2d[row_index, :] = one_boxed_row_flat_pixels

    image_3d = np.reshape(image_2d, (dims, dims, 3))

    print(" Read [" + fname + "] size = " + str(dims))

    # allocate
    image_3d_new = np.zeros((dims, dims, 3), dtype=np.uint8)

    # shuffle
    for i in range(3):
        image_3d_new[half:, half:, i] = image_3d[0:half, half:, i]  # UL -> UR
        image_3d_new[half:, 0:half:, i] = image_3d[half:, half:, i]  # UR -> LR
        image_3d_new[0:half, 0:half, i] = image_3d[half:, 0:half, i]  # LR -> LL
        image_3d_new[0:half, half:, i] = image_3d[0:half, 0:half, i]  # LL -> UL

    # write
    pngFile = open(fname_out, "wb")
    writer = pngWriter(dims, dims, alpha=False, bitdepth=8)
    writer.write(pngFile, np.reshape(image_3d_new, (-1, dims * 3)))
    pngFile.close()

    print(" Wrote [" + fname_out + "].")


def shuffleFixAll():
    """Fix images where the four quadrants are shuffled."""
    for j in range(16):
        xx = j / 4
        yy = j % 4
        fname = "../vtkout_dmdens/9/" + str(xx) + "/" + str(yy) + ".png"
        fname_out = "dmdens_" + str(j) + ".png"

        print(fname + "  " + fname_out)

        _shuffle4imgs(fname, fname_out)


def combine4to1(fnames_in, fname_out):
    """Combine four PNG images into one larger PNG image."""
    # setup
    r = pngReader(filename=fnames_in[0])
    columnCount, rowCount, pngData, metaData = r.asDirect()

    dims = rowCount

    # read [0]
    image_2d = np.zeros((dims, 3 * dims), dtype=np.uint8)

    for row_index, one_boxed_row_flat_pixels in enumerate(pngData):
        image_2d[row_index, :] = one_boxed_row_flat_pixels

    image_3d = np.reshape(image_2d, (dims, dims, 3))

    print(" Read [" + fnames_in[0] + "] size = " + str(dims))

    # allocate
    image_3d_new = np.zeros((dims * 2, dims * 2, 3), dtype=np.uint8)

    # put into new image
    for i in range(3):
        # image_3d_new[0:dims,dims:,i]  = image_3d[:,:,i]  # [0] -> UL (OLD)
        image_3d_new[0:dims, 0:dims, i] = image_3d[:, :, i]  # [2] -> LL

    # read [1]
    r = pngReader(filename=fnames_in[1])
    columnCount, rowCount, pngData, metaData = r.asDirect()
    image_2d = np.zeros((dims, 3 * dims), dtype=np.uint8)

    for row_index, one_boxed_row_flat_pixels in enumerate(pngData):
        image_2d[row_index, :] = one_boxed_row_flat_pixels

    image_3d = np.reshape(image_2d, (dims, dims, 3))

    print(" Read [" + fnames_in[1] + "] size = " + str(dims))

    # put into new image
    for i in range(3):
        # image_3d_new[dims:,dims:,i]   = image_3d[:,:,i]   # [1] -> UR (OLD)
        image_3d_new[0:dims, dims:, i] = image_3d[:, :, i]  # [0] -> UL

    # read [2]
    r = pngReader(filename=fnames_in[2])
    columnCount, rowCount, pngData, metaData = r.asDirect()
    image_2d = np.zeros((dims, 3 * dims), dtype=np.uint8)

    for row_index, one_boxed_row_flat_pixels in enumerate(pngData):
        image_2d[row_index, :] = one_boxed_row_flat_pixels

    image_3d = np.reshape(image_2d, (dims, dims, 3))

    print(" Read [" + fnames_in[2] + "] size = " + str(dims))

    # put into new image
    for i in range(3):
        # image_3d_new[0:dims,0:dims,i] = image_3d[:,:,i]  # [2] -> LL (OLD)
        image_3d_new[dims:, 0:dims:, i] = image_3d[:, :, i]  # [3] -> LR

    # read [3]
    r = pngReader(filename=fnames_in[3])
    columnCount, rowCount, pngData, metaData = r.asDirect()
    image_2d = np.zeros((dims, 3 * dims), dtype=np.uint8)

    for row_index, one_boxed_row_flat_pixels in enumerate(pngData):
        image_2d[row_index, :] = one_boxed_row_flat_pixels

    image_3d = np.reshape(image_2d, (dims, dims, 3))

    print(" Read [" + fnames_in[3] + "] size = " + str(dims))

    # put into new image
    for i in range(3):
        # image_3d_new[dims:,0:dims:,i] = image_3d[:,:,i]   # [3] -> LR (OLD)
        image_3d_new[dims:, dims:, i] = image_3d[:, :, i]  # [1] -> UR

    # write
    print(" Writing...")
    pngFile = open(fname_out, "wb")
    writer = pngWriter(dims * 2, dims * 2, alpha=False, bitdepth=8)
    writer.write(pngFile, np.reshape(image_3d_new, (-1, dims * 2 * 3)))
    pngFile.close()

    print(" Wrote [" + fname_out + "].")


def combine4All():
    """Combine sets of four PNG images into larger PNG images."""
    sets = ([0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15])

    for i, set in enumerate(sets):
        fnames_in = ["dmdens_" + str(index) + ".png" for index in set]

        fname_out = "out_" + str(i) + ".png"
        print(fnames_in)
        print("  " + fname_out)

        combine4to1(fnames_in, fname_out)


def pyramidTNG(fieldName="gassynch"):
    """Combine files for TNG Explorer2D and write out image pyramid files."""
    from ..plot.util import loadColorTable
    from ..vis.boxDrivers import TNG_explorerImageSegments
    from ..vis.render import gridBox

    fileBase = os.path.expanduser("~") + "/data/frames/"
    nPixels = 16384
    nPanels = 64  # 8x8
    confNums = {
        "gasdens": 0,
        "dmdens": 1,
        "dmannih": 23,
        "dmvel": 15,
        "stardens": 2,
        "gasvel": 14,
        "gassynch": 24,
        "gastemp": 7,
        "gasbmag": 5,
        "gasxray": 11,
        "gasmachnum": 12,
        "gasshockdedt": 13,
        "gashi": 16,
        "gaso6o8": 33,
    }

    # allocate global (64GB or 64GB*3 for pngs)
    nPanelsPerDim = int(np.sqrt(nPanels))
    nPxPerDim = nPixels * nPanelsPerDim

    data = np.zeros((nPxPerDim, nPxPerDim), dtype="float32")

    # load
    for i in range(nPanels):
        # get panel
        panels = TNG_explorerImageSegments(conf=confNums[fieldName], taskNum=i, retInfo=True)
        p = panels[0]

        # get grid
        grid, config = gridBox(**p)

        # stamp
        panelRow = int(np.floor(i / nPanelsPerDim))
        panelCol = int(i % nPanelsPerDim)

        i0 = panelRow * nPixels
        i1 = (panelRow + 1) * nPixels
        j0 = panelCol * nPixels
        j1 = (panelCol + 1) * nPixels

        data[i0:i1, j0:j1] = grid

    # colormap
    vMM = p["valMinMax"] if "valMinMax" in p else None
    plaw = p["plawScale"] if "plawScale" in p else None
    if "plawScale" in config:
        plaw = config["plawScale"]
    if "plawScale" in p:
        plaw = p["plawScale"]
    cenVal = p["cmapCenVal"] if "cmapCenVal" in p else None
    if "cmapCenVal" in config:
        cenVal = config["cmapCenVal"]
    cmap = loadColorTable(config["ctName"], valMinMax=vMM, plawScale=plaw, cmapCenterVal=cenVal)

    # loop over levels, starting at lowest (256px tiles)
    outPath = fileBase + "/%s/" % fieldName
    tileSize = 256
    levelMax = 9
    levelMin = 0

    for level in range(levelMax, levelMin - 1, -1):
        # if not at lowest (first iteration), downsize array by half its current value
        print("Level: " + str(level))
        os.makedirs(outPath + str(level))

        if level != levelMax:
            data = _reduceArray2D(data, 1)

        # slice array into 256x256 segments
        nSub = (data.shape)[0] / tileSize

        for colIndex in range(nSub):
            os.makedirs(outPath + str(level) + "/" + str(colIndex))
            print("  col [" + str(colIndex + 1) + "] of [" + str(nSub) + "]")

            for rowIndex in range(nSub):
                saveFilename = str(level) + "/" + str(colIndex) + "/" + str(rowIndex) + ".png"

                # get chunk (TMS indexing convention)
                x0 = ((nSub - 1) - rowIndex) * tileSize
                x1 = ((nSub - 1) - rowIndex + 1) * tileSize
                y0 = (colIndex) * tileSize
                y1 = (colIndex + 1) * tileSize

                array = data[x0:x1, y0:y1]

                # colormap and save
                array_normed = (array - vMM[0]) / (vMM[1] - vMM[0])
                array_rgb = cmap(array_normed)
                array_rgb = array_rgb[:, :, 0:3]  # skip alpha channel
                array_rgb = np.clip(np.round(array_rgb * 255.0), 0, 255).astype("uint8")  # [0,1] -> [0,255]

                saveImageToPNG(outPath + saveFilename, array_rgb)

    print("Done.")

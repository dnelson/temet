# pyramid.py - building image pyramid
# dnelson
# feb.2014
# requires: curl -O https://raw.github.com/drj11/pypng/master/code/png.py

import h5py
import numpy as np
import png
import os.path
import scipy.ndimage
import colorsys

def getDataArrayHDF5(filename, fieldName):
    if not os.path.isfile(filename):
        print( " " + filename + " -- MISSING")
        return []
        
    # open file, get data dimensions
    print( " " + filename )
    
    f = h5py.File(filename,'r')
    
    if fieldName != "RGB":
        group = f.get(fieldName)
        dataset = group.get("Array")
        dtype = np.float32
    else:
        dataset = f[fieldName] # RGB is a dataset without a parent group
        dtype = np.uint8
    
    dims = dataset.shape
    
    # allocate
    array = np.zeros( dims, dtype=dtype )

    # read and return
    dataset.read_direct( array )
    
    f.close()
    
    #array_mm = [ np.min(array), np.max(array) ]
    #print("  min: " + str(array_mm[0]) + " max: " + str(array_mm[1]));    
    
    return array
    
def convertDataArrayToRGB(array, colortable):
    if colortable is None:
        # fix to byte and immediate return
        return array
        
    # take log
    dims = array.shape
    
    # convert float32 -> (R,G,B) tuples
    array_rgb = np.zeros( (dims[0],dims[1],3), dtype=np.uint8 )
    
    # apply CT
    x = colortable['table']['x']
    r = colortable['table']['r']
    g = colortable['table']['g']
    b = colortable['table']['b']
    
    arr_f = (array - colortable['minmax'][0]) / (colortable['minmax'][1] - colortable['minmax'][0])

    arr_r = np.interp( arr_f, x, r ) * 255.0
    arr_g = np.interp( arr_f, x, g ) * 255.0
    arr_b = np.interp( arr_f, x, b ) * 255.0
    
    array_rgb[:,:,0] = np.clip( np.round(arr_r), 0, 255 )
    array_rgb[:,:,1] = np.clip( np.round(arr_g), 0, 255 )
    array_rgb[:,:,2] = np.clip( np.round(arr_b), 0, 255 )
    
    return array_rgb
    
def loadColorTable(ct):
    # load CPT format
    f = open(ct['file'])
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
    x = np.array( x, np.float32 )
    r = np.array( r, np.float32 )
    g = np.array( g, np.float32 )
    b = np.array( b, np.float32 )
    
    if colorModel == "HSV":
       for i in range(r.shape[0]):
           rr,gg,bb = colorsys.hsv_to_rgb(r[i]/360.,g[i],b[i])
           r[i] = rr ; g[i] = gg ; b[i] = bb
    if colorModel == "HSV":
       for i in range(r.shape[0]):
           rr,gg,bb = colorsys.hsv_to_rgb(r[i]/360.,g[i],b[i])
           r[i] = rr ; g[i] = gg ; b[i] = bb
    if colorModel == "RGB":
        r = r/255.
        g = g/255.
        b = b/255.
    xNorm = (x - x[0])/(x[-1] - x[0])
 
    # reverse?
    if ct['reverse']:
        r = r[::-1]
        g = g[::-1]
        b = b[::-1]
        
    # gamma scaling?
    if ct['gamma'] != 1.0:
        xNorm = np.power(xNorm, ct['gamma'])
 
    red   = []
    blue  = []
    green = []
    
    for i in range(len(x)):
        red.append([xNorm[i],r[i],r[i]])
        green.append([xNorm[i],g[i],g[i]])
        blue.append([xNorm[i],b[i],b[i]])
        
    #colorDict = {"red":red, "green":green, "blue":blue}
    table = {"x":xNorm, "r":r, "g":g, "b":b}
    
    # add actual table to ct
    ct['table']    = table
    ct['tableNum'] = nTable
    
    return ct
    
def saveImageToPNG(filename, array_rgb):
    dims = array_rgb.shape
    
    # open image and write
    img = open(filename, 'wb')
    pngWriter = png.Writer(dims[0],dims[1],alpha=False,bitdepth=8)
    
    pngWriter.write( img, np.reshape(array_rgb, (-1,dims[0]*dims[2])) )
    
    img.close()
    
def getColorTableParams(fieldName):
    ctBase = '/n/home07/dnelson/idl/mglib/vis/cpt-city/'
    
    if fieldName == "Density":
        ct = { 'file'    : ctBase + 'ncl/WhiteBlueGreenYellowRed-dnA.cpt',
               'minmax'  : [-6.0,-1.4], # [-5.5,-2.7] for 1820_16k_nni/cellgrad # original: [-4.7,-1.6]
               'log'     : True,
               'reverse' : True,
               'gamma'   : 1.3 }
               
    if fieldName == "Temp":
        ct = { 'file'    : ctBase + 'kst/33_blue_red.cpt',
               'minmax'  : [3.0,7.2],
               'log'     : True,
               'reverse' : False,
               'gamma'   : 1.2 } 
    
    if fieldName == "Entropy":
        ct = { 'file'    : ctBase + 'pm/f-23-28-3-dnB.cpt',
               'minmax'  : [9.2,12.5], # originally [8.0,11.2]
               'log'     : True,
               'reverse' : False,
               'gamma'   : 2.0 } 
               
    if fieldName == "Metal":
        ct = { 'file'    : ctBase + 'wkp/tubs/nrwc.cpt',
               'minmax'  : [-5.0,-1.4],
               'log'     : True,
               'reverse' : False,
               'gamma'   : 1.0 } 
               
    if fieldName == "VelMag":
        ct = { 'file'    : ctBase + 'pm/f-34-35-36.cpt',
               'minmax'  : [50.0,960.0],
               'log'     : False,
               'reverse' : False,
               'gamma'   : 1.0 } 
           
    if fieldName == "SzY":
        ct = { 'file'    : ctBase + 'oc/zeu.cpt',
               'minmax'  : [-2.5,4.7], # originally [2.5,5.6]
               'log'     : True,
               'reverse' : True,
               'gamma'   : 0.5 } 
           
    if fieldName == "XRay":
        ct = { 'file'    : ctBase + 'kst/03_red_temperature.cpt',
               'minmax'  : [-12.0,-2.5], # originally [-7.6,-2.5]
               'log'     : True,
               'reverse' : False,
               'gamma'   : 1.5 } 

    return ct
               
def make_pyramid_all(config):
    # allocate
    if config['totImgSize'].shape[0] == 2:
        alloc_gb_str = str((float(config['totImgSize'][0])*config['totImgSize'][1]*4) / 1024 / 1024 / 1024)
        globalArray = np.zeros( config['totImgSize'], dtype='float32' )
    else:
        alloc_gb_str = str((float(config['totImgSize'][0])*config['totImgSize'][1]*3*1) / 1024 / 1024 / 1024)
        globalArray = np.zeros( config['totImgSize'], dtype='uint8' )
    
    print("Allocating... [" + alloc_gb_str.format("%.2f") + " GB]")
    
    # load: loop over hdf5 files for global min/max
    print("Loading...")
    
    for i in range(len(config['fileList'])):
        # load
        array = getDataArrayHDF5(config['fileList'][i], config['fieldName'])

        if len(array) == 0:
            continue
            
        # stamp
        jRow = (config['jobsPerDim']-1) - np.int32(np.floor( i % config['jobsPerDim'] ))
        jCol = np.int32(np.floor( i / config['jobsPerDim'] ))
        
        x0 = config['mDims'][0] * jCol
        x1 = config['mDims'][0] * (jCol+1)
        y0 = config['mDims'][1] * jRow
        y1 = config['mDims'][1] * (jRow+1)
        
        if len(array.shape) == 3:
            for j in range(3):
                globalArray[ x0:x1, y0:y1, j ] = np.transpose( array[j,:,:] )
        else:
            globalArray[ x0:x1, y0:y1 ] = array
    
    # set all zeros to minimum non-zero value and take log (be careful about memory usage)
    if config['ct'] != None:
        min_val = np.min( globalArray[globalArray > 0.0] )
        globalArray[ globalArray <= 0.0 ] = min_val
    
        if config['ct']['log']:
            globalArray = np.log10( globalArray )
        
    array_mm = [ np.min(globalArray), np.max(globalArray) ]
    print(" Global min: " + str(array_mm[0]) + " max: " + str(array_mm[1]));
    
    # loop over levels, starting at lowest (256px tiles)
    for level in range(config['levelMax'],config['levelMin']-1,-1):
        # if not at lowest (first iteration), downsize array by half its current value
        print("Level: " + str(level))
        
        if level != config['levelMax']:
            print(" downsizing...")
            
            globalArray = scipy.ndimage.zoom( globalArray, config['zoom'], order=config['reduceOrder'] )
            
        if config['makeFullImage']:
            # save full image at this zoom level
            dens_rgb = convertDataArrayToRGB(globalArray, config['ct'])
            saveImageToPNG(config['outPath'] + "full_" + str(level) + ".png",dens_rgb)
                
        # rasterize each to PNG, apply colortable, and save
        if config['makePyramid']:
            print(" chunking...")
            os.makedirs(config['outPath'] + str(level))
            
            # slice array into 256x256 segments
            nSub = (globalArray.shape)[0] / config['tileSize']
            
            for colIndex in range(nSub):
                os.makedirs(config['outPath'] + str(level) + "/" + str(colIndex))
                print("  col [" + str(colIndex+1) + "] of [" + str(nSub) + "]")
                
                for rowIndex in range(nSub):
                    saveFilename = str(level) + "/" + str(colIndex) + "/" + str(rowIndex) + ".png"
                    
                    # get chunk (TMS indexing convention)
                    x0 = ((nSub-1)-rowIndex) * config['tileSize']
                    x1 = ((nSub-1)-rowIndex+1) * config['tileSize']
                    y0 = (colIndex) * config['tileSize']
                    y1 = (colIndex+1) * config['tileSize']
                    
                    if len(globalArray.shape) == 3:
                        array = globalArray[ x0:x1, y0:y1, : ]
                    else:
                        array = globalArray[ x0:x1, y0:y1 ]
                    
                    array_rgb = convertDataArrayToRGB(array, config['ct'])
                    saveImageToPNG(config['outPath'] + saveFilename,array_rgb)
                    #print("   " + saveFilename)

def render_tiles_natural_only(config):
    
    print("Loading and rendering natural level...")
    
    #DEBUG: manually combine several tiles (upper right corner) for slightly larger single image
    #DEBUG (1820_128k):
    #globalArray = np.zeros( (8192,8192), dtype='float32' )
    #imgSize = 2048
    #maxRow = 63
    #minCol = 60
        
    # DEBUG (jobtest_exp):
    #globalArray = np.zeros( (512,512), dtype='float32' )
    #imgSize = 256
    #maxRow = 3
    #minCol = 2
    
    for i in range(len(config['fileList'])):
        # load
        array = getDataArrayHDF5(config['fileList'][i], config['fieldName'])
        
        if len(array) == 0: # missing file
            continue
        #DEBUG (1820_128k)
        #if i not in (0,1,2,3,64,65,66,67,128,129,130,131,192,193,194,195):
        #    continue
        #DEBUG (jobtest_exp?):
        #if i != 0:
        #    continue
                
        # set all zeros to minimum non-zero value and take log
        if config['ct']['log']:
            array = np.log10( array )
        
        # calculate row,column position (TMS indexing convention)
        colIndex = (config['jobsPerDim']-1) - np.int32(np.floor( i % config['jobsPerDim'] ))
        rowIndex = (config['jobsPerDim']-1) - np.int32(np.floor( i / config['jobsPerDim'] ))
        
        # DEBUG:
        #x0 = (colIndex-minCol) * imgSize
        #x1 = x0 + imgSize
        #y0 = (maxRow-rowIndex) * imgSize
        #y1 = y0 + imgSize
        #print("  x0: " + str(x0) + " x1: " + str(x1) + "  y0: " + str(y0) + " y1: " + str(y1))
        #globalArray[ y0:y1, x0:x1 ] = array
        
        # save
        saveFilename = "col_" + str(colIndex) + ".row_" + str(rowIndex) + ".png"
        array_rgb = convertDataArrayToRGB(array, config['ct'])
        saveImageToPNG(config['outPath'] + saveFilename,array_rgb)
          
    # DEBUG:
    #saveFilename = "full.png"
    #array_rgb = convertDataArrayToRGB(globalArray, config['ct'])
    #saveImageToPNG(config['outPath'] + saveFilename,array_rgb)
                   
def make_pyramid_upper(config):
    
    print("\nUpper pyramid:")
    
    sizePerDim = config['tileSize'] * config['jobsPerDim']
    
    # allocate
    if config['totImgSize'].shape[0] == 2:
        globalArray = np.zeros( (sizePerDim,sizePerDim), dtype='float32' )
    else:
        globalArray = np.zeros( (sizePerDim,sizePerDim,3), dtype='uint8' )
    
    numReductions = np.log10( config['mDims'][0] / config['tileSize'] ) / np.log10(2)
    numReductions = np.int32( np.round(numReductions) )
    print("Number of reductions from natural tiles: [" + str(numReductions) + "]")
    
    # load: loop over hdf5 files, downsize ecah to tileSize and stamp in
    print("Loading...")
    
    for i in range(len(config['fileList'])):
        # load
        array = getDataArrayHDF5(config['fileList'][i], config['fieldName'])
        
        if len(array) == 0:
            continue
        
        # resize down to tileSize x tileSize
        for j in range(numReductions):
            array = scipy.ndimage.zoom( array, config['zoom'], order=config['reduceOrder'] )

        # stamp
        jRow = (config['jobsPerDim']-1) - np.int32(np.floor( i % config['jobsPerDim'] ))
        jCol = np.int32(np.floor( i / config['jobsPerDim'] ))
        
        x0 = config['tileSize'] * jCol
        x1 = config['tileSize'] * (jCol+1)
        y0 = config['tileSize'] * jRow
        y1 = config['tileSize'] * (jRow+1)
        
        if len(array.shape) == 3:
            for j in range(3):
                globalArray[ x0:x1, y0:y1, j ] = np.transpose( array[j,:,:] )
        else:
            globalArray[ x0:x1, y0:y1 ] = array
    
    # set all zeros to minimum non-zero value and take log
    if config['ct'] != None:
        min_val = np.min( globalArray[globalArray > 0.0] )
        globalArray[ globalArray <= 0.0 ] = min_val
    
        if config['ct']['log']:
            globalArray = np.log10( globalArray )
        
    array_mm = [ np.min(globalArray), np.max(globalArray) ]

    print(" Global min: " + str(array_mm[0]) + " max: " + str(array_mm[1]));
 
    config['globalMin'] = array_mm[0] # store for lower pyramid!
    #import pdb; pdb.set_trace() 
    #return
    
    # render out native level up to level 0 by progressively downsizing
    startLevel = np.int32(np.log10(sizePerDim) / np.log10(2)) - 8

    for level in range(startLevel,config['levelMin']-1,-1):
        print("Level: " + str(level))
        
        if level != startLevel:
            print(" downsizing...")
            globalArray = scipy.ndimage.zoom( globalArray, config['zoom'], order=config['reduceOrder'] )
            
        if config['makeFullImage']:
            # save full image at this zoom level
            dens_rgb = convertDataArrayToRGB(globalArray, config['ct'])
            saveImageToPNG(config['outPath'] + "full_" + str(level) + ".png",dens_rgb)
            
        if not config['makePyramid']:
            continue
            
        # rasterize each to PNG, apply colortable, and save
        print(" chunking...")
        os.makedirs(config['outPath'] + str(level))
        
        # slice array into 256x256 segments
        nSub = (globalArray.shape)[0] / config['tileSize']
        
        for colIndex in range(nSub):
            os.makedirs(config['outPath'] + str(level) + "/" + str(colIndex))
            print("  col [" + str(colIndex+1) + "] of [" + str(nSub) + "]")
            
            for rowIndex in range(nSub):
                saveFilename = str(level) + "/" + str(colIndex) + "/" + str(rowIndex) + ".png"
                
                # get chunk (TMS indexing convention)
                x0 = ((nSub-1)-rowIndex) * config['tileSize']
                x1 = ((nSub-1)-rowIndex+1) * config['tileSize']
                y0 = (colIndex) * config['tileSize']
                y1 = (colIndex+1) * config['tileSize']
                
                if len(globalArray.shape) == 3:
                    array = globalArray[ x0:x1, y0:y1, : ]
                else:
                    array = globalArray[ x0:x1, y0:y1 ]
                
                array_rgb = convertDataArrayToRGB(array, config['ct'])
                saveImageToPNG(config['outPath'] + saveFilename,array_rgb)
   
def make_pyramid_lower(config):

    print("\nLower pyramid:")
    
    sizePerDim = config['tileSize'] * config['jobsPerDim']
    
    startLevel = np.int32(np.log10(sizePerDim) / np.log10(2)) - 8 + 1
    numReductions = config['levelMax'] - startLevel
    
    print("Number of reductions from natural tiles: [" + str(numReductions) + "]")
    
    # load: loop over hdf5 files for global min/max
    print("Loading...")
    
    for i in range(len(config['fileList'])):
        # load
        array = getDataArrayHDF5(config['fileList'][i], config['fieldName'])
        
        if len(array) == 0:
            continue
            
        # set all zeros to minimum non-zero value and take log
        if config['ct'] != None:
            array[ array <= 0.0 ] = config['globalMin']
            
            if config['ct']['log']:
                array = np.log10( array )
            
        for level in range(config['levelMax'],startLevel-1,-1):
            print("  level [" + str(level) + "]")
            
            if level != config['levelMax']:
                array = scipy.ndimage.zoom( array, config['zoom'], order=config['reduceOrder'] )
                
            # global indices at this level
            levelDepth = level - startLevel + 1
            levelExpansionFac = 2 ** levelDepth
            
            jRow = (config['jobsPerDim']-1) - np.int32(np.floor( i / config['jobsPerDim'] ))
            jCol = (config['jobsPerDim']-1) - np.int32(np.floor( i % config['jobsPerDim'] ))
            
            if i == 0 and not os.path.exists(config['outPath'] + str(level)):
                os.makedirs(config['outPath'] + str(level))
                
            # slice array into 256x256 segments
            nSub = (array.shape)[0] / config['tileSize']
            
            for colIndex in range(nSub):
                globalCol = jCol*levelExpansionFac + colIndex
                outDirPath = config['outPath'] + str(level) + "/" + str(globalCol)
                if not os.path.exists(outDirPath):
                    os.makedirs(outDirPath)
                
                for rowIndex in range(nSub):
                    # need to transform local col,row indices into global indices
                    globalRow = jRow*levelExpansionFac + rowIndex
                    
                    # get subaray chunk (TMS indexing convention)
                    x0 = ((nSub-1)-rowIndex) * config['tileSize']
                    x1 = ((nSub-1)-rowIndex+1) * config['tileSize']
                    y0 = (colIndex) * config['tileSize']
                    y1 = (colIndex+1) * config['tileSize']
                    
                    if len(array.shape) == 3:
                        subarray = array[ x0:x1, y0:y1, : ]
                    else:
                        subarray = array[ x0:x1, y0:y1 ]
                    
                    # save
                    saveFilename = str(level) + "/" + str(globalCol) + "/" + str(globalRow) + ".png"

                    array_rgb = convertDataArrayToRGB(subarray, config['ct'])
                    saveImageToPNG(config['outPath'] + saveFilename,array_rgb)
            
        # all levels done for this hdf5 file, move on to next
    print("\nDone.")    

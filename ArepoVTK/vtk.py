# vtk.py
# dnelson
# requires: curl -O https://raw.github.com/drj11/pypng/master/code/png.py

import h5py
import numpy as np
import png
import os.path
import scipy.ndimage
import colorsys

def getDataArrayHDF5(filename, fieldName):
    # open file, get data dimensions
    f = h5py.File(filename,'r')
    
    group = f.get(fieldName)
    dataset = group.get("Array")
    
    dims = dataset.shape
    
    # allocate
    array = np.zeros( dims, dtype=np.float32 )

    # read and return
    dataset.read_direct( array )
    
    f.close()
    
    return array
    
def convertDataArrayToRGB(array, colortable):
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
    
def process():

    # config
    fileBase    = 'frame_455'
    outPath     = 'vtkout/'
    totNumJobs  = 16
    tileSize    = 256
    reduceOrder = 1 # [1-5], 1=bilinear, 3=cubic
    
    makeFullImage = True
    makePyramid   = True
    
    # gas density
    fieldName  = 'Density'
    ct = { 'file'    : '/n/home07/dnelson/idl/mglib/vis/cpt-city/ncl/WhiteBlueGreenYellowRed-dnA.cpt',
           'minmax'  : [-4.7, -1.6],
           'log'     : True,
           'reverse' : True,
           'gamma'   : 1.3 }
    
    fileList = []
    
    # get initial (hdf5) file listing, size of each, and total image size
    for curJobNum in range(0,totNumJobs):
        filename = fileBase + '_' + str(curJobNum) + '_' + str(totNumJobs) + '.hdf5'
        
        if not os.path.isfile(filename):
            print("Error: File not found [" + filename + "]")
            return
            
        fileList.append( filename )
        
    # calculate pyramid levels (where in the pyramid do the hdf5 tiles sit)
    dummy = getDataArrayHDF5( fileList[0], "Density" )
    mDims = np.array(dummy.shape,dtype='int32')

    jobsPerDim = np.int32(np.sqrt(totNumJobs))
    totImgSize = mDims * jobsPerDim
    totImgPx   = totImgSize[0] * totImgSize[1]
    
    if totImgSize[0] != totImgSize[1]:
        print("Error: Expecting square image.")
        return
    
    print("Final image size: " + str(totImgSize[0]) + "x" + str(totImgSize[1]) + 
        " (" + str(totImgPx/1e6) + " MP) (Total jobs: " + str(totNumJobs) + 
        ", jobs per dim: " + str(jobsPerDim) + ")")
        
    levelMin = 0
    levelMax = np.int32(np.log10(totImgSize[0]) / np.log10(2)) - 8   #totImgSize[0] / 256
    levelNat = np.int32(np.log10(mDims[0]) / np.log10(2)) - 8   #mDims[0] / 256
    
    print(" Level min: " + str(levelMin))
    print(" Level max: " + str(levelMax))
    print(" Level nat: " + str(levelNat))
    
    # make pyramid directories / meta file
    if os.path.exists(outPath):
        print("Error: Output location [" + outPath + "] already exists.")
        return
        
    os.makedirs(outPath)
    
    # load actual colortable
    ct = loadColorTable(ct)
    
    # load: loop over hdf5 files
    print("Loading...")
    
    globalArray = np.zeros( totImgSize, dtype='float32' )
    
    for i in range(len(fileList)):
        print( " " + fileList[i] )
        # load
        array = getDataArrayHDF5(fileList[i], fieldName)

        # stamp
        jRow = (jobsPerDim-1) - np.int32(np.floor( i % jobsPerDim ))
        jCol = np.int32(np.floor( i / jobsPerDim ))
        
        x0 = mDims[0] * jCol
        x1 = mDims[0] * (jCol+1)
        y0 = mDims[1] * jRow
        y1 = mDims[1] * (jRow+1)
        
        #print " row col: " + str(jRow) + " " + str(jCol)
        #print " x: " + str(x0) + " " + str(x1) + " y: " + str(y0) + " " + str(y1)
        globalArray[ x0:x1, y0:y1 ] = array
        
    if ct['log']:
        globalArray = np.log10( globalArray )
        
    array_mm = [ np.min(globalArray), np.max(globalArray) ]
    print(" Global min: " + str(array_mm[0]) + " max: " + str(array_mm[1]));
    
    # loop over levels, starting at lowest (256px tiles)
    for level in range(levelMax,levelMin-1,-1):
        # if not at lowest (first iteration), downsize array by half its current value
        print("Level: " + str(level))
        
        os.makedirs(outPath + str(level))
        
        if level != levelMax:
            print " downsizing..."
            
            #globalArray = half_bilinear( globalArray )
            globalArray = scipy.ndimage.zoom( globalArray, 0.5, order=reduceOrder )
            
        if makeFullImage:
            # save full image at this zoom level
            dens_rgb = convertDataArrayToRGB(globalArray, ct)
            saveImageToPNG(outPath + "full_" + str(level) + ".png",dens_rgb)
                
        # rasterize each to PNG, apply colortable, and save
        if makePyramid:
            print " chunking..."
            
            # slice array into 256x256 segments
            nSub = (globalArray.shape)[0] / tileSize
            
            for colIndex in range(nSub):
                os.makedirs(outPath + str(level) + "/" + str(colIndex))
                print "  col [" + str(colIndex+1) + "] of [" + str(nSub) + "]"
                
                for rowIndex in range(nSub):
                    saveFilename = str(level) + "/" + str(colIndex) + "/" + str(rowIndex) + ".png"
                    
                    # get chunk (TMS indexing convention)
                    x0 = ((nSub-1)-rowIndex) * tileSize
                    x1 = ((nSub-1)-rowIndex+1) * tileSize
                    y0 = (colIndex) * tileSize
                    y1 = (colIndex+1) * tileSize
                    
                    #print " row col: " + str(rowIndex) + " " + str(colIndex)
                    #print " x: " + str(x0) + " " + str(x1) + " y: " + str(y0) + " " + str(y1)
                    array = globalArray[ x0:x1, y0:y1 ]
                    
                    array_rgb = convertDataArrayToRGB(array, ct)
                    saveImageToPNG(outPath + saveFilename,array_rgb)
                    #print "   " + saveFilename
       
    #idl stop:
    #import pdb; pdb.set_trace()
    
# vtk.py - processing of ArepoVTK tiled output
# dnelson
# feb.2014

from pyramid import *

def process():

    # config
    fileBase      = 'test' #frame_455 #frame_1820 #frame_jobtest1
    outPath       = 'vtkout_test/'
    totNumJobs    = 4096 #16 #64 #256
    tileSize      = 256
    reduceOrder   = 1 # [1-5], 1=bilinear, 3=cubic
    fieldName     = 'Density' # Density, Temp, Entropy, Metal, VelMag, SzY, XRay
    makeFullImage = True
    makePyramid   = False
    
    # get initial (hdf5) file listing, size of each, and total image size
    fileList = []
    
    for curJobNum in range(0,totNumJobs):
        filename = fileBase + '_' + str(curJobNum) + '_' + str(totNumJobs) + '.hdf5'
        
        fileList.append( filename )
        
    # calculate pyramid levels (where in the pyramid do the hdf5 tiles sit)
    dummy = getDataArrayHDF5( fileList[0], "Density" )
    mDims = np.array(dummy.shape,dtype='int64')

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
    ct = getColorTableParams(fieldName)
    ct = loadColorTable(ct)
    
    print("Colortable for [" + fieldName + "] min = " + str(ct['minmax'][0]) + " max = " + 
          str(ct['minmax'][1]) + " log = " + str(ct['log']) + " gamma = " + str(ct['gamma']))
    
    config = { 'ct'            : ct,
               'zoom'          : 0.5,
               'zoomGlobal'    : 0.5,
               'mDims'         : mDims,
               'jobsPerDim'    : jobsPerDim,
               'totImgSize'    : totImgSize,
               'totImgPx'      : totImgPx,
               'levelMin'      : levelMin,
               'levelMax'      : levelMax,
               'levelNat'      : levelNat,
               'fileList'      : fileList,
               'fieldName'     : fieldName,
               'tileSize'      : tileSize,
               'outPath'       : outPath,
               'reduceOrder'   : reduceOrder,
               'makeFullImage' : makeFullImage,
               'makePyramid'   : makePyramid }
    
    # OPTION (1): make full pyramid (global image allocation)
    #make_pyramid_all(config)
    
    # OPTION (2): make upper and lower pyramids separately (memory efficient)
    #make_pyramid_upper(config)
    #make_pyramid_lower(config)
    
    # OPTION (3): render the tiles at their natural (existing) level only for debugging
    render_tiles_natural_only(config)
    
    #import pdb; pdb.set_trace() #idl stop
    
def processShy():

    # config
    # 455 lowres
    #fileBase   = 'L75n455FPs135_75000_75000_kpc_9155_pc_tile'
    #totNumJobs = 16
    # 455 128k
    #fileBase   = "/n/ghernquist/sgenel/Illustris/stellar_images/L75n455FP/L75n455FPholys135_75000_75000_kpc_572_pc_tile"
    # 1820 128k
    fileBase = "/n/ghernquist/sgenel/Illustris/stellar_images/L75n1820FP/L75n1820FPholys135_75000_75000_kpc_572_pc_tile"
    #fileBase = "L75n1820FPholys135_75000_75000_kpc_572_pc_tile"
    
    fieldName     = "RGB"
    nThirdDims    = 3 # 3=RGB, 4=RGBA
    outPath       = 'vtkout_stellar/'
    totNumJobs    = 64 #16 #64 #256
    tileSize      = 256
    reduceOrder   = 1 # [1-5], 1=bilinear, 3=cubic
    makeFullImage = True
    makePyramid   = True
    
    # get initial (hdf5) file listing, size of each, and total image size
    fileList = []
    
    for curJobNum in range(0,totNumJobs):
        filename = fileBase + '_' + str(curJobNum) + '_RGBmat_scaled.hdf5'
        fileList.append( filename )
        
    # calculate pyramid levels (where in the pyramid do the hdf5 tiles sit)
    dummy = getDataArrayHDF5( fileList[27], fieldName )
    mDims = np.array(dummy.shape[1:3],dtype='int64')

    jobsPerDim = np.int32(np.sqrt(totNumJobs))
    totImgSize      = np.zeros( (3,), dtype='int64' )
    totImgSize[0:2] = mDims * jobsPerDim
    totImgSize[2]   = nThirdDims
    totImgPx   = totImgSize[0] * totImgSize[1]
    
    if totImgSize[0] != totImgSize[1]:
        print("Error: Expecting square image.")
        print totImgSize
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
    
    config = { 'ct'            : None,
               'zoom'          : (1.0,0.5,0.5),
               'zoomGlobal'    : (0.5,0.5,1.0),
               'nThirdDims'    : nThirdDims,
               'mDims'         : mDims,
               'jobsPerDim'    : jobsPerDim,
               'totImgSize'    : totImgSize,
               'totImgPx'      : totImgPx,
               'levelMin'      : levelMin,
               'levelMax'      : levelMax,
               'levelNat'      : levelNat,
               'fileList'      : fileList,
               'fieldName'     : fieldName,
               'tileSize'      : tileSize,
               'outPath'       : outPath,
               'reduceOrder'   : reduceOrder,
               'makeFullImage' : makeFullImage,
               'makePyramid'   : makePyramid }
    
    # OPTION (1): make full pyramid (global image allocation)
    #make_pyramid_all(config)
    
    # OPTION (2): make upper and lower pyramids separately (memory efficient)
    make_pyramid_upper(config)
    make_pyramid_lower(config)
    
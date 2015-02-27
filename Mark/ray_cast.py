import numpy as np
import readsnapHDF5 as rs
import RayCastNew as rc
import sys
import conversions as co
import ArepoVTK.png as png

def makePngFromDat():
    fname_in  = "image_2000_0.dat"
    fname_out = "image_2000.png"
    dims = np.array([960,540,4])
    fac = 2.0

    fd=open(fname_in, "r")
    image = np.fromfile(fd,dtype='float64')
    fd.close()
    
    zz = image.reshape( dims )
    
    # convert RGBA quadlets (in [0,1]) to bytes (in [0,255])
    array_rgb = (zz[:,:,0:3] * 255.0 * fac).round()
    array_rgb = array_rgb.clip(0,255).astype('uint8')

    array_rgb = array_rgb.transpose( (1,0,2) )
    
    # open image and write
    img = open(fname_out, 'wb')
    dims[2] = 3
    pngWriter = png.Writer(dims[0],dims[1],alpha=False,bitdepth=8)
    pngWriter.write( img, np.reshape(array_rgb, (-1,dims[0]*dims[2])) )
    img.close()
    
def testRender():

    # render config
    hsmlfac   = 3.0       # boost (sphere volume)^(1/3) by this factor
    RenderFac = 15000.0   # multiply masses to boost intensity in mass-weighting
    boxsize   = 7500.0    # ckpc/h
    near      = 250.0     # near plane
    far       = 2*boxsize # far plane
    width     = 533.33    # frustrum width? 1000.0
    height    = 300.0     # frustrum height? 750.0
    
    # frame/field config
    startnum   = 2000      # snapshot/frame num
    endnum     = 2001      # snapshot/frame num
    RenderWhat = 0        # 0=temp, 1=dens, 2=metal

    base = "/n/hernquistfs1/Illustris/Runs/Illustris-1/output/subbox0/"

    # orbit config
    framenum  = 720                 # how many frames per orbit
    radius    = boxsize/2.0 - 100.0 # orbit radius 500.0
    
    # image config
    boost = 1
    xbins = boost*1920/2
    ybins = boost*1080/2
    
    # calculate render setup
    center = np.array([boxsize/2.0,boxsize/2.0,boxsize/2.0])
    frames = 2.0*np.pi*np.arange(0,framenum)/(framenum-1.0)
    
    x = center[0] + radius*np.cos(frames)
    y = np.repeat(center[1], framenum)
    z = center[2] + radius*np.sin(frames)
    
    # load transfer function
    if (RenderWhat==0):
        rc.LoadTF("transRGB_temp.txt")
        basename = "./image_"
    if (RenderWhat==1):
        rc.LoadTF("transRGB_rho.txt")   
        basename = "./image_"
    if (RenderWhat==2):
        rc.LoadTF("transRGB_met.txt")
        basename = "./image_"

    #specify frustum (4:3 aspect ratios!)
    n=near
    f=far
    r=width/2.0
    l=-width/2.0
    t=+height/2.0
    b=-height/2.0

    xcam=x.copy()
    ycam=y.copy()
    zcam=z.copy()
    upx=np.repeat(0, x.shape[0])
    upy=np.repeat(1, x.shape[0])
    upz=np.repeat(0, x.shape[0])
    xcamto=np.repeat(center[0], x.shape[0])
    ycamto=np.repeat(center[1], y.shape[0])
    zcamto=np.repeat(center[2], z.shape[0])

    import pdb; pdb.set_trace()
    
    for j in range(startnum,endnum):
        print j, startnum, endnum
        sys.stdout.flush()
        i=j % framenum

        fname = base+"snapdir_subbox0_"+str(j).zfill(3)+"/snap_subbox0_"+str(j).zfill(3)

        pos  = rs.read_block(fname, "POS ", parttype=0).astype("float32")
        hsml = hsmlfac * (3.0/4.0/np.pi*rs.read_block(fname, "VOL ", parttype=0).astype("float32"))**(1.0/3.0)
        mass = rs.read_block(fname, "MASS", parttype=0).astype("float32")

        if (RenderWhat==0):
            u     = rs.read_block(fname, "U   ", parttype=0).astype("float64")
            ne    = rs.read_block(fname, "NE  ", parttype=0).astype("float64")
            quant = co.GetTemp(u, ne, 5.0/3.0).astype("float32")

        if (RenderWhat==1):
            quant   = rs.read_block(fname, "RHO ", parttype=0).astype("float32")
        if (RenderWhat==2):
            quant   = rs.read_block(fname, "GZ  ", parttype=0).astype("float32")

        #translate box so that center 
        pos[:,0]=(pos[:,0]-5250)
        pos[:,1]=(pos[:,1]-13250)
        pos[:,2]=(pos[:,2]-59250)

        #for k in range(0,3):
        #   print "min/max coord: ",k,pos[:,k].min(), pos[:,k].max()

        #save original values
        x_orig=pos[:,0]
        y_orig=pos[:,1]
        z_orig=pos[:,2]
        hsml_orig=hsml
        mass_orig=mass
        quant_orig=quant

        #eye and eyeto vectors
        eye=np.array([xcam[i], ycam[i], zcam[i]])
        eyeto=np.array([xcamto[i], ycamto[i], zcamto[i]])   
        up=np.array([upx[i],upy[i],upz[i]])

        #construct homog. transformation matrix
        PS=np.matrix([[(2*n)/(r-l),0,(r+l)/(r-l),0],[0,(2*n)/(t-b),(t+b)/(t-b),0],[0,0,-(f+n)/(f-n),-(2*f*n)/(f-n)],[0,0,-1,0]])
        nvec=-(eye-eyeto)*(-1.0)/np.sqrt((eye[0]-eyeto[0])**2.0 + (eye[1]-eyeto[1])**2.0 + (eye[2]-eyeto[2])**2.0)  #-1 
        temp=np.cross(up,nvec)
        rvec=temp/np.sqrt(temp[0]**2.0 + temp[1]**2.0 + temp[2]**2.0)
        uvec=np.cross(nvec, rvec)
        R=np.matrix([[rvec[0],rvec[1],rvec[2],0],[uvec[0],uvec[1],uvec[2],0],[nvec[0],nvec[1],nvec[2],0],[0,0,0,1]])
        T=np.matrix([[1,0,0,-eye[0]],[0,1,0,-eye[1]],[0,0,1,-eye[2]],[0,0,0,1]])

        PSRT=PS*R*T

        #PSRT tranformation: world coordinates -> camera coordinates
        x=PSRT[0,0]*x_orig + PSRT[0,1]*y_orig + PSRT[0,2]*z_orig + PSRT[0,3]*1
        y=PSRT[1,0]*x_orig + PSRT[1,1]*y_orig + PSRT[1,2]*z_orig + PSRT[1,3]*1
        z=PSRT[2,0]*x_orig + PSRT[2,1]*y_orig + PSRT[2,2]*z_orig + PSRT[2,3]*1
        w=PSRT[3,0]*x_orig + PSRT[3,1]*y_orig + PSRT[3,2]*z_orig + PSRT[3,3]*1

        hsml_x=PS[0,0]*hsml_orig + PS[0,1]*hsml_orig + PS[0,2]*hsml_orig + PS[0,3]*1
        hsml_y=PS[1,0]*hsml_orig + PS[1,1]*hsml_orig + PS[1,2]*hsml_orig + PS[1,3]*1

        w+=1e-20

        #homog. scaling
        x/=w
        y/=w
        z/=w
        mass=mass_orig
        quant=quant_orig
        s=np.abs(w)
        hsml_x/=s
        hsml_y/=s
        hsml_o=hsml_orig

        #clipping in frustum (clip a bit larger for particle contributions outside of frustum)
        index=(np.abs(x) < 1.01)  & (np.abs(y) < 1.01) & (np.abs(z) < 1.01)
        x=x[index]
        y=y[index]
        z=z[index]
        hsml_x=hsml_x[index]
        hsml_y=hsml_y[index]
        hsml_o=hsml_o[index]
        mass=mass[index]
        quant=quant[index]

        #sort ascending according to pseudo-depth
        index=np.argsort(z)
        x=x[index]
        y=y[index]
        z=z[index]
        hsml_x=hsml_x[index]
        hsml_y=hsml_y[index]
        hsml_o=hsml_o[index]
        mass=mass[index]
        quant=quant[index]

        #avoid single pixel flickering
        pixfac=0.5
        hsml_x[hsml_x<pixfac*2.0/xbins]=0
        hsml_y[hsml_y<pixfac*2.0/ybins]=0

        mass*=RenderFac
        print "start render..."
        image=rc.Render(x, y, quant, hsml_x, hsml_y, hsml_o, mass, xbins, ybins, hsmlfac)
        print "done."

        #save file
        fd=open(basename+str(j).zfill(4)+"_"+str(RenderWhat)+".dat", "wb")
        image.astype("float64").tofile(fd)
        fd.close()
        
        #clean image
        image=image*0.0 

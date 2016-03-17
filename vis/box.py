"""
box.py
  Visualizations for whole (cosmological) boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

from cosmo import load
from vis.common import gridBox

def boxImgSpecs(sP, zoomFac, relCenPos, axes):
    """ Factor out some box/image related calculations common to all plot styles. """
    boxSizeImg = [sP.boxSize, sP.boxSize, sP.boxSize]
    boxCenter  = [relCenPos[0],relCenPos[1],0.5] * np.array(boxSizeImg)

    for k in axes:
        boxSizeImg[k] *= zoomFac

    extent = [ boxCenter[axes[0]] - 0.5*boxSizeImg[axes[0]], 
               boxCenter[axes[0]] + 0.5*boxSizeImg[axes[0]], 
               boxCenter[axes[1]] - 0.5*boxSizeImg[axes[1]], 
               boxCenter[axes[1]] + 0.5*boxSizeImg[axes[1]]]

    return boxSizeImg, boxCenter, extent

def renderBox():
    """ Render a full/fraction of a cosmological box, variable number of panels, comparing any 
        combination of res, run, redshift, partType, partField. """
    from util import simParams

    # selection config (each entry adds one panel, and must specify all 5 parameters)
    #   partType  : dm, gas, stars, tracerMC, ...
    #   partField : coldens, coldens_cgs, temp, velmag, entr, ...
    panels = []
    panels.append( {'run':'tracer', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'temp'} )
    panels.append( {'run':'tracer', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'coldens'} )
    #panels.append( {'run':'feedback', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'temp'} )
    #panels.append( {'run':'feedback', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'coldens'} )

    # plot config
    method    = 'sphMap'  # sphMap, voronoi_const, voronoi_grads, ...
    nPixels   = 300       # number of pixels per dimension of images when projecting
    zoomFac   = 1         # [0,1], only in axes, not along projection direction
    relCenPos = [0.5,0.5] # [0-1,0-1] relative coordinates, only in axes
    axes      = [2,1]     # e.g. [0,1] is x,y
    plotStyle = 'edged'   # open, edged
    rasterPx  = 1200      # each panel will have this number of pixels if making a raster (png) output
                          # but note also it controls the relative size balance of raster/vector (e.g. fonts)

    #fieldsStr = '_'.join( [str(p['partType'])+'-'+str(p['partField']) for p in panels] )
    saveFilename = 'renderBox_N%d_%s_px%d_axes%d%d_zf%d.pdf' % \
      (len(panels),method,nPixels,axes[0],axes[1],zoomFac)

    if plotStyle == 'open':
        # start plot
        nRows  = np.floor(np.sqrt(len(panels)))
        nCols  = len(panels) / nRows
        aspect = nRows/nCols

        sizeFac = rasterPx / mpl.rcParams['savefig.dpi']
        fig = plt.figure(figsize=(1.167*sizeFac*nRows/aspect,sizeFac*nRows))

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            sP = simParams(res=p['res'], run=p['run'], redshift=p['redshift'])

            boxSizeImg, boxCenter, extent = boxImgSpecs(sP, zoomFac, relCenPos, axes)
        
            # grid projection for image
            print(sP.run,sP.res,sP.redshift,p['partType'],p['partField'])
            grid, config = gridBox(sP, method, p['partType'], p['partField'], [nPixels,nPixels], 
                                   axes, boxCenter, boxSizeImg)

            # set axes and place image
            ax = fig.add_subplot(nRows,nCols,i+1)
            ax.set_title('%s %d z=%3.1f %s %s' % (sP.simName,sP.res,sP.redshift,p['partType'],p['partField']))
            ax.set_xlabel( ['x','y','z'][axes[0]] + ' [ ckpc/h ]')
            ax.set_ylabel( ['x','y','z'][axes[1]] + ' [ ckpc/h ]')

            plt.imshow(grid, extent=extent, aspect=1.0)

            if 1:
                # debug plotting 10 most massive halos
                gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupPos','Group_R_Crit200'], skipIDs=True)

                for j in range(20):
                    xPos = gc['halos']['GroupPos'][j,axes[0]]
                    yPos = gc['halos']['GroupPos'][j,axes[1]]
                    rad  = gc['halos']['Group_R_Crit200'][j] * 2

                    c = plt.Circle( (xPos,yPos), rad, color='#ffffff', linewidth=1.5, fill=False)
                    ax.add_artist(c)

            # colobar
            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
            cb = plt.colorbar(cax=cax)
            cb.ax.set_ylabel(config['label'])

        fig.tight_layout()
        fig.savefig(saveFilename)

    if plotStyle == 'edged':
        # start plot
        nRows  = np.floor(np.sqrt(len(panels)))
        nCols  = len(panels) / nRows
        aspect = nRows/nCols

        fig = plt.figure(frameon=False, tight_layout=False)
        barAreaHeight = 0.1
        
        sizeFac = rasterPx / mpl.rcParams['savefig.dpi']
        fig.set_size_inches(sizeFac*nCols, sizeFac*nRows*(1/(1.0-barAreaHeight)))

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            sP = simParams(res=p['res'], run=p['run'], redshift=p['redshift'])

            boxSizeImg, boxCenter, extent = boxImgSpecs(sP, zoomFac, relCenPos, axes)
        
            # grid projection for image
            print(sP.run,sP.res,sP.redshift,p['partType'],p['partField'])
            grid, config = gridBox(sP, method, p['partType'], p['partField'], [nPixels,nPixels], 
                                   axes, boxCenter, boxSizeImg)

            # set axes and place image
            curRow = np.floor(i / nCols)
            curCol = i % nCols

            rowHeight  = (1.0 - barAreaHeight) / nRows
            colWidth   = 1.0 / nCols
            bottomNorm  = 1.0 - rowHeight * (curRow+1)
            leftNorm = colWidth * curCol

            pos = [leftNorm, bottomNorm, colWidth, rowHeight]
            print(curRow,curCol,pos)
            ax = fig.add_axes(pos)
            ax.set_axis_off()

            plt.imshow(grid, extent=extent, aspect=1.0)

            if 1:
                # debug plotting 10 most massive halos
                gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupPos','Group_R_Crit200'], skipIDs=True)

                for j in range(20):
                    xPos = gc['halos']['GroupPos'][j,axes[0]]
                    yPos = gc['halos']['GroupPos'][j,axes[1]]
                    rad  = gc['halos']['Group_R_Crit200'][j] * 2

                    c = plt.Circle( (xPos,yPos), rad, color='#ffffff', linewidth=1.5, fill=False)
                    ax.add_artist(c)

            # colobar
            factor  = 0.95 # bar length, fraction of column width, 1.0=whole
            height  = 0.04 # colorbar height, fraction of entire figure
            hOffset = 0.4  # padding between image and start of bar (fraction of height)

            leftNormBar   = leftNorm + 0.5*colWidth*(1-factor)
            bottomNormBar = barAreaHeight - height*hOffset - height

            posBar = [leftNormBar, bottomNormBar, colWidth*factor, height]

            cax = fig.add_axes(posBar)
            cax.set_axis_off()
            cb = plt.colorbar(cax=cax, orientation='horizontal')
            cb.ax.set_ylabel(config['label'])

        fig.savefig(saveFilename)#, bbox_inches='tight', pad_inches=0)
        fig.savefig(saveFilename+'.png')

    plt.close(fig)

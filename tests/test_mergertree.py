""" Test merger tree functionality. """
import pytest
from temet.cosmo.mergertree import plot_tree

from temet.util import simParams as sim
from os.path import isfile
from os import remove

@pytest.mark.requires_data
def test_plot_tree():
    sP = sim(res=1820,run='tng',redshift=0.0)

    for haloID in [200]: #[20,200,210,600,2000,20000]:
        subhaloID = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']
        saveFilename = 'mergertree_%s_%d.png' % (sP.simName,haloID)

        plot_tree(sP, subhaloID, saveFilename)

        # delete file
        assert isfile(saveFilename)
        remove(saveFilename)

@pytest.mark.requires_data
def test_plot_tree_mem(haloID=190):
    from io import BytesIO

    from imageio import imread, imwrite
    from skimage.transform import resize

    # config
    sP = sim(res=1820,run='tng',snap=99)

    supersample = 4
    output_fmt = 'png'

    # start
    buf = None # return image array by default

    if output_fmt == 'pdf':
        # fill memory buffer instead
        buf = BytesIO()

    subhaloID = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

    # render
    im = plot_tree(sP, subhaloID, saveFilename=buf, dpi=100*supersample, output_fmt=output_fmt)

    # resize
    if output_fmt in ['png','jpg']:
        output_shape = (im.shape[0]//supersample, im.shape[1]//supersample)
        im = resize(im, output_shape, order=3)
        im = (im * 255).astype('uint8')

        # 'save'
        buf = BytesIO()
        imwrite(buf, im, format=output_fmt)

    # write to file and then remove
    saveFilename = 'mergertree_%s_snap-%d_%d.%s' % (sP.simName,sP.snap,haloID,output_fmt)
    with open(saveFilename,'wb') as f:
        f.write(buf.getvalue())

    assert isfile(saveFilename)
    remove(saveFilename)

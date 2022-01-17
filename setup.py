import atexit
import os
import sys

from subprocess import check_call
from setuptools import setup
from setuptools.command.install import install
from pathlib import Path

def post_install():
    """ Custom post-install actions. """
    def find_module_path():
        for p in sys.path:
            if os.path.isdir(p) and 'tenet' in os.listdir(p):
                return os.path.join(p, 'tenet')
    install_path = Path(find_module_path())

    home = Path.home()

    # matplotlibrc
    mpl = home / ".config/matplotlib"
    mpl.mkdir(parents=True, exist_ok=True)

    if not (mpl / "matplotlibrc").exists():
        (mpl / "matplotlibrc").symlink_to(install_path / "matplotlibrc")
   
    # Roboto font

    # download tables/

    # install FSPS

    # download AtomDB tables

    # install SKIRT

    #print('Install all done!')
    #print('Path = [%s]' % install_path)
    #print('Exiting.')

class new_install(install):
    def __init__(self, *args, **kwargs):
        super(new_install, self).__init__(*args, **kwargs)
        atexit.register(post_install)

setup(
    cmdclass={'install': new_install},
    name='tenet',
    version='0.0.1',
    url='https://www.github.com/dnelson86/tenet',
    author='Dylan Nelson',
    packages=["tenet"],
    python_requires='>=3.6',
    install_requires=["astropy","cmocean","corner","emcee","fsps","h5py",
                      "healpy","llvmlite","matplotlib","numba","numpy",
                      "psutil","requests","requests-oauthlib","scipy",
                      "astro-sedpy","astro-prospector","reproject",
                      "scikit-image",
                      "illustris_python @ git+https://github.com/illustristng/illustris_python@master",
                      ],
)

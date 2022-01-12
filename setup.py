from setuptools import setup

setup(
    name='tenet',
    version='0.0.1',
    packages=["tenet"],
    install_requires=["astropy","cmocean","corner","emcee","fsps","h5py",
                      "healpy","llvmlite","matplotlib","numba","numpy",
                      "psutil","requests","requests-oauthlib","scipy",
                      "astro-sedpy","astro-prospector","reproject",
                      "scikit-image","importlib_resources",
                      "illustris_python @ git+https://github.com/cbyrohl/illustris_python@master",
                      ],
)

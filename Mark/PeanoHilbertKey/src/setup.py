from distutils.core import setup, Extension

setup(name = "PeanoHilbertKey", version = "0.1",
  ext_modules = [
    Extension("PeanoHilbertKey", ["main.c","peano.c","allvars.c"])
    ]
)

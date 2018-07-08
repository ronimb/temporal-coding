from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize('return_subset.pyx', annotate=True, include_path=[numpy.get_include(),
                  '/home/ronimber/anaconda3/envs/temporal_coding/include/python3.6m',
                  '/home/ronimber/anaconda3/include/python3.6m']),
    include_dirs=[numpy.get_include(),
                  '/home/ronimber/anaconda3/envs/temporal_coding/include/python3.6m',
                  '/home/ronimber/anaconda3/include/python3.6m']
)
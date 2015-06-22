"""Setup Telauges"""

from setuptools import setup, find_packages;

setup(
      name="Telauges",
      version="0.1.0",
      author="Yuhuang Hu",
      author_email="duguyue100@gmail.com",
      license="MIT License",
      description="Telauges is a Deep Neural Network library for education.",
      keywords="Deep Learning, Machine Learning, Optimization",
      url="http://rt.dgyblog.com/",
      packages=find_packages(),
      install_requires=['theano',
                        'numpy',
                        'scipy'],
      classifiers=['Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Topic :: Scientific/Engineering :: Machine Learning',
                   'Programming Language :: Python :: 2.7',
                   'Operating System :: OS Independent']
)
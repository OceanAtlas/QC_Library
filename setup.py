# -*- mode: python; coding: utf-8 -*
# Copyright (c) John Osborne 
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

from setuptools import setup, Extension
import os
import io

with io.open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()

setup_args = {
    'name': 'QC_Library',
    'author': 'John Osborne/Bryna Hazelton',
    'url': 'https://github.com/OceanAtlas/QC_Library',
    'license': 'BSD',
    'description': 'A Library of routines to despike TS data',
    'long_description': readme,
    'long_description_content_type': 'text/markdown',
    'package_dir': {'QC_Library': 'QC_Library'},
    'packages': ['QC_Library'],
    'version': '0.0.1',
    'include_package_data': True,
#    'setup_requires': ['numpy', 'scipy'],
    'install_requires': ['numpy', 'scipy'],
    'classifiers': ['Development Status :: 2 - Pre-Alpha',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: BSD License',
                    'Programming Language :: Python :: 3.6',
                    'Topic :: Scientific/Engineering'],
    'keywords': 'oceanography seawater carbon, time series, quality control'
}

if __name__ == '__main__':
    setup(**setup_args)

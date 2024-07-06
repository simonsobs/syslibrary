#!/usr/bin/env python
# -*- coding: utf-8 -*-
# """syslibrary: GitHub <https://github.com/simonsobs/syslibrary>."""

from setuptools import setup

setup(name='syslibrary',
      version='0.2.1',
      description='Systematics library',
      author='Simons Observatory sys crew',
      author_email='',
      packages=['syslibrary'],
      python_requires='>3.9',
      install_requires=[
          "numpy",
          "PyYAML",
          "fgspectra"
      ],
      include_package_data=True
      )

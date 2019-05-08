#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='TorchLab',
      version='0.1',
      description='A package to help experimenting with Pytorch.',
      author='Marvin Teichmann',
      author_email='marvin.teichmann@googlemail.com',
      packages=find_packages(),
      package_data={'': ['*.lst', '*camvid_ids.json']}
      )

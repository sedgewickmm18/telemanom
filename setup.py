#!/usr/bin/env python

from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='telemanom',
    version='0.0.2',
    packages=find_packages(),
    install_requires=requirements
)

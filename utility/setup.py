#!/usr/bin/env python3

from setuptools import Extension, setup

setup(
    name="orca",
    version="1.0",
    ext_modules=[Extension("orcastr", ["bind.cpp", "liborca.cpp"])],
    py_modules=["orca"],
)

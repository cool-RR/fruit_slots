# Copyright 2022 Ram Rachum and collaborators.
# This program is distributed under the MIT license.
import setuptools
import re


def read_file(filename):
    with open(filename) as file:
        return file.read()

version = re.search("__version__ = '([0-9.]*)'",
                    read_file('fruit_slots/__init__.py')).group(1)

setuptools.setup(
    name='fruit_slots',
    version=version,
    author='Ram Rachum',
    author_email='ram@rachum.com',
    description='An experiment in implicit communication between learning agents',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/cool-RR/fruit_slots',
    packages=setuptools.find_packages(exclude=['tests*']),
    install_requires=read_file('requirements.txt'),
    extras_require={
        'tests': {
            'pytest',
        },
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: CPython',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installation script for `orion.wrappers`."""
from setuptools import setup

setup_args = dict(
    name='orion.wrappers',
    version=0.1,
    description="Implement a multiple wrappers for tests purposes.",
    license='BSD-3-Clause',
    author='François Corneau-Tremblay',
    author_email='corneau90@gmail.com',
    url='https://github.com/mila-udem/orion',
    packages=['orion.wrappers'],
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=['orion.core'],
    setup_requires=['setuptools'],
    # "Zipped eggs don't play nicely with namespace packaging"
    # from https://github.com/pypa/sample-namespace-packages
    zip_safe=False
    )

setup_args['keywords'] = [
    'Machine Learning',
    'Deep Learning',
    'Distributed',
    'Optimization',
    ]

setup_args['platforms'] = ['Linux']

setup_args['classifiers'] = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
] + [('Programming Language :: Python :: %s' % x)
     for x in '3 3.4 3.5 3.6'.split()]

if __name__ == '__main__':
    setup(**setup_args)

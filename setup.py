#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installation script for Oríon."""
from glob import iglob
import os
import sys

from setuptools import setup

import versioneer

isfile = os.path.isfile
pjoin = os.path.join
repo_root = os.path.dirname(os.path.abspath(__file__))
mpath = pjoin(repo_root, 'src')
sys.path.insert(0, mpath)

import orion.core as orion  # noqa

print(sys.version)


def find_data_files():
    """Find Oríon's configuration and metadata files."""
    install_config_path = pjoin(orion.DIRS.site_data_dir, 'config')
    config_path = pjoin('config', '*')
    configs = [cfg for cfg in iglob(config_path) if isfile(cfg)]

    data_files = [
        (install_config_path, configs),
        (orion.DIRS.site_data_dir, ['LICENSE', 'README.rst']),
    ]

    return data_files


tests_require = [
    'pytest>=3.0.0'
    ]


packages = [
    'orion.core',
    'orion.client',
    'orion.algo',
    ]

setup_args = dict(
    name='orion.core',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=orion.__descr__,
    long_description=open(os.path.join(repo_root, "README.rst"), 'rt', encoding='utf8').read(),
    license=orion.__license__,
    author=orion.__author__,
    author_email=orion.__author_email__,
    url=orion.__url__,
    packages=packages,
    package_dir={'': 'src'},
    include_package_data=True,
    data_files=find_data_files(),
    entry_points={
        'console_scripts': [
            'orion = orion.core.cli:main',
            ],
        'BaseAlgorithm': [
            'random = orion.algo.random:Random',
            ],
        },
    install_requires=['PyYAML', 'pymongo>=3', 'numpy', 'scipy', 'gitpython'],
    tests_require=tests_require,
    setup_requires=['setuptools', 'pytest-runner'],
    extras_require=dict(test=tests_require),
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

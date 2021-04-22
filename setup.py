#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installation script for Oríon."""
import os

from setuptools import setup

import versioneer


repo_root = os.path.dirname(os.path.abspath(__file__))


tests_require = ["pytest>=3.0.0", "scikit-learn"]


packages = [  # Packages must be sorted alphabetically to ease maintenance and merges.
    "orion.algo",
    "orion.analysis",
    "orion.benchmark",
    "orion.client",
    "orion.core",
    "orion.plotting",
    "orion.serving",
    "orion.storage",
    "orion.testing",
]

setup_args = dict(
    name="orion",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Asynchronous [black-box] Optimization",
    long_description=open(
        os.path.join(repo_root, "README.rst"), "rt", encoding="utf8"
    ).read(),
    license="BSD-3-Clause",
    author=u"Epistímio",
    author_email="xavier.bouthillier@umontreal.ca",
    url="https://github.com/epistimio/orion",
    packages=packages,
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "orion = orion.core.cli:main",
        ],
        "OptimizationAlgorithm": [
            "random = orion.algo.random:Random",
            "gridsearch = orion.algo.gridsearch:GridSearch",
            "asha = orion.algo.asha:ASHA",
            "hyperband = orion.algo.hyperband:Hyperband",
            "tpe = orion.algo.tpe:TPE",
            "EvolutionES = orion.algo.evolution_es:EvolutionES",
        ],
        "Storage": [
            "track = orion.storage.track:Track",
            "legacy = orion.storage.legacy:Legacy",
        ],
    },
    install_requires=[
        "PyYAML",
        "pymongo>=3",
        "numpy",
        "scipy",
        "gitpython",
        "filelock",
        "tabulate",
        "AppDirs",
        "plotly",
        "kaleido",
        "requests",
        "pandas",
        "gunicorn",
        "falcon",
        "scikit-learn",
        "psutil",
        "pytest>=3.0.0",
    ],
    tests_require=tests_require,
    setup_requires=["setuptools", "appdirs", "pytest-runner"],
    extras_require=dict(test=tests_require),
    # "Zipped eggs don't play nicely with namespace packaging"
    # from https://github.com/pypa/sample-namespace-packages
    zip_safe=False,
)

setup_args["keywords"] = [
    "Machine Learning",
    "Deep Learning",
    "Distributed",
    "Optimization",
]

setup_args["platforms"] = ["Linux"]

setup_args["classifiers"] = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
] + [("Programming Language :: Python :: %s" % x) for x in "3 3.6 3.7 3.8 3.9".split()]

if __name__ == "__main__":
    setup(**setup_args)

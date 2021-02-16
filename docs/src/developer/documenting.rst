***********
Documenting
***********
The documentation is built using Sphinx_ with the `Read The Docs`_ theme.

We try to write the documentation at only one place and reuse it as much as possible. For instance,
the home page of this documentation (https://orion.readthedocs.io/) is actually pulled
from the README.rst and appended with a table of content of the documentation generated
automatically. The advantage of having a single source of truth is that it's vastly easier to find
information and keep it up to date.

Updating README.rst
===================

When you need to reference a page from the documentation on the README.rst, make sure to always
point to the **stable** channel in readthedocs (https://orion.readthedocs.io/en/stable/).

If you need to add a link to a specific page in the documentation that is not yet on the stable
channel, make the link to the **latest** channel (https://orion.readthedocs.io/en/latest/). During
the :doc:`release process </developer/release>` the link will be updated to the stable channel.

Writing examples
================

There is two types of tutorials in Oríon, the
:ref:`visualization tutorials <visualizations>` and
the
:ref:`code tutorials <sphx_glr_auto_tutorials>`.
The visualization tutorials are meant to be light in computations and
should run on readthedocs. If there is a need for heavy computations to
generate data for the visualizations, the code used for the heavy computations
should be turned into a code tutorial. As an example, see
:ref:`sphx_glr_auto_tutorials_code_2_hyperband_checkpoint.py` which served for
the generation of data in
:ref:`sphx_glr_auto_examples_plot_2_parallel_coordinates.py`.

To create a new visualization example, create a python script in ``examples/plotting``
with the name ``plot_{i}_{name}.py`` where ``i`` is the index of the new
visualization tutorial and ``name`` is its name.
Likewise, for code examples, create a python script in ``examples/tutorials``
with the name ``code_{i}_{name}.py``.

The database used in visualization examples should always be the pickleddb
found at ``examples/db.pkl``
(which is ``../db.pkl`` relative to example scripts file locations).
This is because these tutorials are executed sequentially by sphinx-gallery
and the storage does not get reset between the examples. We could reset the
database at beginning of each examples but that would clutter the tutorial and
likely confuse the users. This is why we fix the database throughout all
visualization tutorials.

The execution of the code examples must be done manually with the following command::

   tox -e build-doc-data code_{i}_{name}

This will:

1. Flush the full database ``db.pkl``.
2. Create a database for the given example ``code_{i}_{name}``.
3. Execute the script.
4. Merge back all example databases ``db.pkl``.
5. Generate plots for each types of plot and in
   ``docs/src/_static/{experiment.name}_{plot_type}.html``.

You can then ``git commit`` the new ``code_{i}_{name}_db.pkl``,
the new version of ``db.pkl``, and all new plots in ``docs/src/_static/``.

If the new example have additional requirements, add them to
``docs/scripts/requirements.txt``.

Building documentation
======================

To generate the *html* and *man* pages of the documentation, run:

.. code-block:: sh

   tox -e docs

When writing, you can run ``$ tox -e serve-docs`` to host the content of
``/docs/build/html`` on http://localhost:8000.

.. _Read The Docs: https://sphinx-rtd-theme.readthedocs.io/en/latest/
.. _Sphinx: http://www.sphinx-doc.org/en/master/

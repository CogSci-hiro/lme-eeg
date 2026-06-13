Install
=======

LmeEEG currently targets Python 3.10 or newer.

Base install
------------

Install the published package from PyPI:

.. code-block:: bash

   pip install lmeeeg

Install from source
-------------------

For local development from the repository root:

.. code-block:: bash

   pip install -e .

Optional MNE backends
---------------------

Cluster-based and TFCE permutation correction require the optional MNE dependency:

.. code-block:: bash

   pip install -e ".[mne]"

Documentation dependencies
--------------------------

To build this documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html

Developer install
-----------------

For tests plus documentation tooling:

.. code-block:: bash

   pip install -e ".[dev,docs,mne]"

Sanity check
------------

.. code-block:: bash

   python -c "import lmeeeg; print(lmeeeg.__all__)"

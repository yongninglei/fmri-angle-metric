Installation
============

uv (recommended)
----------------

.. code-block:: bash

   uv venv && source .venv/bin/activate
   uv pip install -e .

pip
---

.. code-block:: bash

   pip install -e .

conda / micromamba
------------------

.. code-block:: bash

   micromamba env create -f environment.yml
   micromamba activate fmri-angle-metric

Development hooks
-----------------

.. code-block:: bash

   pip install pre-commit
   pre-commit install

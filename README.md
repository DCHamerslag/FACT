# FACT

## Voor het installeren (dit werkte voor mij maar idk)
In de folder met environment.yml:
conda env create
conda activate FACT-AI
conda install -c cvxgrp cvxcanon==0.1.1
pip uninstall cvxpy
pip install cvxpy==0.4.10
conda install -c conda-forge scipy==1.1.0

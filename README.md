# CyS Complentary Material: Source Code and Data

This repository contains complementary material to a paper published in Computación y Sistemas (CYS):

A computational approach to find SEIR model parameters that best explain infected and recovered time series for SARS-CoV 2.

## Repository structure

The main entry point for the experimental results and source code is the Jupyter Notebook: `COVID-SCENARIOS.iphnb`. From this, we can find the following sub-structure:

- Folder `data` contains a `.json`file with COVID timeseries of several countries downloaded from [https://covid19.who.int](https://covid19.who.int).
- Folder `dnn_opt_seir` which is a modified clone of the library [dnn_opt](https://github.com/ml-opt/dnn_opt)


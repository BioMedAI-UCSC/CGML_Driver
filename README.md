# Driver Repository

This is the Driver repository for our entire molecular dynamics and machine learning pipeline, providing a coordinated and seamless workflow for simulation, training, and analysis.

## Overview

This repository contains scripts and tools that integrate multiple components of our computational pipeline:

* **Regular OpenMM All Atom MD Programs and Scripts** (`openmm_generate`)  
  Standard molecular dynamics simulations using OpenMM for all-atom representations

* **Training CGSchNet-based Models** (`base_model`)  
  Machine learning model training infrastructure for coarse-grained molecular representations using CGSchNet

* **WESTPA for Data Generation** (`westpa_prop`)  
  Weighted Ensemble Simulation Toolkit with Parallelization and Analysis (WESTPA) integration supporting:
  - ML-driven simulations
  - Regular MD simulations
  - Custom progress coordinates
  - Enhanced sampling workflows

* **Benchmark Suite** (`benchmark`)  
  Qualitative and quantitative comparison tools for evaluating and validating models

* **Shared Modules** (`module`)  
  Repository containing all code and functions shared between base model, benchmark, openmm_generate, and westpa_prop.

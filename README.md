# Vehicle Stability Early Warning

This project contains a vehicle stability early-warning pipeline using simulation, physics-informed modeling, preprocessing, and evaluation workflows.

## Project structure

- `data/raw/` - source dataset files
- `data/processed/` - processed and feature-engineered datasets
- `simulator/` - vehicle dynamics simulation and synthetic data generation
- `physics/` - stability boundary and margin calculations
- `preprocessing/` - data cleaning, noise injection, and sequence creation
- `models/` - baseline and trained models
- `evaluation/` - metrics and reporting
- `experiments/` - configuration and experiment runs
- `results/` - figures, tables, and saved models
- `notebooks/` - exploratory and analysis notebooks

## Setup

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run simulations or experiments from the `experiments/` package.

## Notes

This repository is intended as a starting scaffold for an early-warning vehicle stability system.

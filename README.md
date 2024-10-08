# Project KRKO

## Overview
This project is designed to train a machine learning model using the `uv` tool.

## Project Structure
```
krko/
.
├── data                    # Data storage directory
│   ├── external            # External data sources
│   ├── interim             # Intermediate data
│   ├── processed           # Processed data ready for modeling
│   └── raw                 # Raw, unprocessed data
│       └── krkopt.data     # Original dataset file
├── Makefile                # Makefile for automating tasks
├── models                  # Trained models storage
│   └── model_0_0.h5        # Example trained model file
├── notebooks               # Jupyter notebooks for experiments
├── pyproject.toml          # Project dependencies and configuration
├── README.md               # Project overview and instructions
├── reports                 # Generated reports and figures
│   └── figures             # Figures for reports
├── src                     # Source code for the project
│   ├── config.py           # Configuration settings
│   ├── dataset.py          # Data loading and preprocessing
│   ├── features.py         # Feature engineering scripts
│   ├── __init__.py         # Package initialization
│   ├── modeling            # Modeling scripts
│   │   ├── __init__.py     # Package initialization for modeling
│   │   ├── predict.py      # Prediction script
│   │   └── train.py        # Training script
│   ├── models              # Model architectures
│   │   ├── ffnn.py         # Feedforward neural network model
│   │   └── __init__.py     # Package initialization for models
│   └── plots.py            # Plotting utilities
└── uv.loc                  # UV tool configuration file
```

## How to Use `uv`
To train the model, use the following command:

```sh
uv run -m src.modeling.train
```
To add dependecies

```sh
uv add numpy
```

Make sure you have all the necessary dependencies installed before running the command.

## Dependencies
- `uv` tool from https://docs.astral.sh/uv/
- Python 3.x
- Other dependencies as listed in `pyproject.toml`


## License
This project is licensed under the MIT License.

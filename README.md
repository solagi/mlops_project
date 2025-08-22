# mlops_project
DTU PhD course on Advanced Machine Leaning: MLOps topics (August 2025)

## Codebase Overview

This repo contains code for practical exercises covered in the course. Mainly focusing on using Weights & Biases (wandb)for model training and experiments. The example setup is building a small CNNs on the MNIST dataset using PyTorch.


### Main Components
- `app.py`: Streamlit web app for showcase demo (light sensors).
- `adurino.c`: Example C code for Arduino integration and reading sensor data real-time (color rgb scale).
- `cnn_model.py`: Defines the ConvNet architecture.
- `utils.py`: Utility functions for data loading, preprocessing, model building, training, evaluation, and experiment logging.
- `main.py`: Entry point for running experiments.
- `project_setup.ipynb`: Jupyter notebook for interactive setup and experimentation.

The codebase demonstrates best practices in experiment tracking, artifact management, and reproducible machine learning workflows. All files are logged on the *wandb* project.  
# mlops_project
DTU PhD course on Advanced Machine Leaning: MLOps topics (August 2025)

## Codebase Overview

- Practical exercises using Weights & Biases (wandb) for model training and experiment tracking. Building small CNNs on MNIST with PyTorch. Data and artifacts are logged directly on the *wandb* project.  
- Example project code for Arduino® Nano 33 BLE Sense board.

### Main Components
Weights & Biases
- `cnn_model.py`: Defines the ConvNet architecture.
- `utils.py`: Utility functions for data loading, preprocessing, model building, training, evaluation, and experiment logging.
- `main.py`: Entry point for running experiments.
- `project_setup.ipynb`: Jupyter notebook for interactive setup and experimentation.

Arduino setup
- `app.py`: Streamlit web app for showcase demo (light sensors).
- `adurino.c`: Example C code for Arduino integration and reading sensor data real-time (color rgb scale).



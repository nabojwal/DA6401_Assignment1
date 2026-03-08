# A Multi-Layer Perceptron for Image Classification using MNIST/Fashion-MNIST Datasets

## Overview

The aim of this project is to implement a neural network **from scratch using only NumPy**.  
All core components are built manually, including layers, activation functions, optimizers, and loss functions.  
The model is trained and evaluated on the **MNIST** and **Fashion-MNIST** datasets.
## Repository
Link to GitHub Repository:  
[https://github.com/nabojwal/DA6401_Assignment1](https://github.com/nabojwal/DA6401_Assignment1)

## W&B Report
Link to W&B Report:  
[https://wandb.ai/nabojwal_dl1/da6401_assignment_1/reports/DA6401-Assignment-1-Report-Part-1--VmlldzoxNjExMDM0MA?accessToken=o8762b8isv6u9lvc4w4oostt1tcwu7whj1yksuaqf3ffic6dniz0flt7stq6ezoz](https://wandb.ai/nabojwal_dl1/da6401_assignment_1/reports/DA6401-Assignment-1-Report-Part-1--VmlldzoxNjExMDM0MA?accessToken=o8762b8isv6u9lvc4w4oostt1tcwu7whj1yksuaqf3ffic6dniz0flt7stq6ezoz)
## Project Structure
```DA6401_Assignment1/
├── src/
│   ├── ann/                        # Core Neural Network Package
│   │   ├── __init__.py             
│   │   ├── activations.py          # ReLU, Sigmoid, Tanh, Softmax
│   │   ├── neural_layer.py         # Linear layer (Z = WA + b)
│   │   ├── neural_network.py       # Orchestrator (forward/backward/train)
│   │   ├── objective_functions.py  # Loss functions (CrossEntropy, MSE)
│   │   └── optimizers.py           # Adam, SGD, Momentum, Nadam, NAG, and RMSProp
│   │
│   ├── utils/                      # Helper modules
│   │   ├── __init__.py
│   │   └── data_loader.py          # Data fetching & normalization
│   │
│   ├── best_model.npy              # Trained model weights
│   ├── train.py                    # Training CLI 
│   └── inference.py                # Prediction script
│
├── models/                         # Experiment logs & backups
    ├── model.npy
│   └── config.json                 
│
├── requirements.txt                # Dependencies
└── README.md                       # Project Documentation
```

## Train on MNIST 
Run the following command for training with the best hyperparameters.
```markdown
```bash
python -m src.train -d mnist -e 10 -b 64 -lr 0.001 -o adam -nhl 3 -sz 128 128 128 -a relu -l cross_entropy -wi xavier
```

## Thank You!















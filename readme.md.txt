```markdown
# PyTorch Deep Learning Project

This repository contains a PyTorch project for deep learning tasks, focusing on computer vision using the MNIST dataset.

## Overview

The project includes code for:

- Loading and preprocessing the MNIST dataset.
- Defining a neural network architecture for image classification.
- Training the model on the MNIST dataset.
- Evaluating model performance.
- Visualizing training progress and results.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

You can install the required packages using pip:

## Including Required Lines

#### Ensure to include the following lines at the beginning of your Python scripts or Jupyter Notebooks:

```python
import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
%matplotlib inline
```

**Note:** Make sure to include `%matplotlib inline` in your Jupyter Notebook or JupyterLab environment to display matplotlib plots inline without calling `plt.show()` explicitly.

## Description

- 1 contain pytorch basic
- 2 contain liner regression implementing using pytorch
- 3 contain use one layer neural network(linear regression) to identify hand written digits
- 4
  - contain how to create two layer nural network
  - how to use GPU for training
````

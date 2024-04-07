I'd be glad to provide information about `torch-summary` (formerly `torchsummary`):

**What is torch-summary?**

`torch-summary` is a third-party library for PyTorch that offers a concise and informative way to visualize the architecture of your neural network models. It's not part of the core PyTorch library but is a valuable tool for understanding the structure and parameters of your models.

**How to Use torch-summary**

1. **Installation:**

   If you haven't already, install `torch-summary` using pip:

   ```bash
   pip install torch-summary
   ```

2. **Import:**

   In your Python code, import the `summary` function from `torch-summary`:

   ```python
   from torchsummary import summary
   ```

3. **Usage:**

   Pass your PyTorch model to the `summary` function along with the input size (a tuple representing the shape of the input tensor):

   ```python
   import torch

   # Example model (replace with your actual model)
   model = torch.nn.Sequential(
       torch.nn.Conv2d(3, 6, 5),
       torch.nn.ReLU(),
       torch.nn.MaxPool2d(2, stride=2),
       torch.nn.Flatten(),
       torch.nn.Linear(16, 10)
   )

   # Example input size
   input_size = (3, 28, 28)  # 3 color channels, 28x28 image

   summary(model, input_size=input_size)
   ```

**Output:**

The `summary` function will print a table summarizing the model architecture. Here's an example output:

```
------------------------------------------------------------------------------------------
        Layer (type)               Input               Output                      Param         # Learnable Params  # Non-Learnable Params
------------------------------------------------------------------------------------------
            Conv2d-1 (Conv2d)       3 x 28 x 28          6 x 24 x 24             144           144                   0
             ReLU-2 (ReLU)            6 x 24 x 24          6 x 24 x 24               0             0                   0
         MaxPool2d-3 (MaxPool2d)     6 x 24 x 24          3 x 12 x 12               0             0                   0
           Flatten-4 (Flatten)       3 x 12 x 12         144                            0             0                   0
            Linear-5 (Linear)        144                 10                            1450          1450                   0
==========================================================================================
Total params: 1,594
Trainable params: 1,594
Non-trainable params: 0
------------------------------------------------------------------------------------------
```

The table provides details for each layer, including:

- **Layer (type):** Name and type of the layer (e.g., Conv2d, ReLU)
- **Input:** Shape of the input tensor to the layer
- **Output:** Shape of the output tensor from the layer
- **Param:** Total number of parameters in the layer (including trainable and non-trainable)
- **# Learnable Params:** Number of parameters that are updated during training
- **# Non-Learnable Params:** Number of parameters that are not updated during training (e.g., batch normalization layers)

**Benefits of Using torch-summary:**

- **Debugging:** Helps identify potential issues in your model architecture, such as incorrect layer connections or unexpected input/output shapes.
- **Performance Analysis:** Provides a rough estimate of the computational cost (number of parameters) associated with running the model.
- **Model Comparison:** Compares the complexity and size of different models for decision-making.
- **Conciseness:** Offers a clear and concise summary compared to manually inspecting the model structure.

**Alternative:**

The `torchinfo` library, which has replaced `torch-summary` in development, provides similar functionality with some potential enhancements. It's worth considering if you're starting a new project. However, `torch-summary` remains widely used and is still a viable option for summarizing PyTorch models.
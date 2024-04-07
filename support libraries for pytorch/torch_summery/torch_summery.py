from torchsummary import summary

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

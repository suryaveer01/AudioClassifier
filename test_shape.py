import torch

# Create a tensor of shape [1, 128, 431] with random values
tensor = torch.randn(1, 128, 431)

# Reshape the tensor to [128, 431]
tensor = tensor.squeeze(0)

print(tensor.shape)  # should print [128, 431]

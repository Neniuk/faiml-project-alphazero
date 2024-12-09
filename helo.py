import torch
import numpy as np

# Example tensor
tensor = torch.rand(3, 3)

# Convert to numpy
numpy_array = tensor.cpu().detach().numpy()
print(numpy_array)

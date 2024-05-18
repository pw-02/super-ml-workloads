import torch

tensor = torch.tensor([2, 2, 7, 6, 3, 4, 8, 1, 6, 8, 4, 9, 3, 1, 6, 9, 9, 7, 4, 6, 1, 0, 3, 0])
unique_values = torch.unique(tensor)
num_unique_values = unique_values.numel()

print(f'The number of unique values is: {num_unique_values}')

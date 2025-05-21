import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[7, 8, 9], [10, 11, 12]])
print(f"Tensor A: \n{a}")
print(f"Tensor B: \n{b}")

print(f"Addition: \n{a + b}")
print(f"Subtraction: \n{a - b}")
print(f"Multiplication: \n{a * b}")
print(f"Division: \n{a / b}")
print(f"Matrix Multiplication: \n{torch.matmul(a, b.T)}")

# In-place operations
# Note: In-place operations modify the original tensor
a.add_(b)
print(f"Tensor A after in-place addition: \n{a}")

# dimensions
print(f"\nDIMS EXAMPLE")
c = torch.ones((2, 3, 2))
c += torch.eye(2).view(2, 1, 2)
c[1] += 2

print(f"Tensor C: \n{c}")

print(f"Shape of Tensor A: {c.shape}")
print(f"Number of dimensions of Tensor A: {c.ndim}")
print(f"Tensor A after unsqueeze: \n{c.unsqueeze(0)}")
print(f"Shape of Tensor A after unsqueeze: {c.unsqueeze(0).shape}")
print(f"Tensor A after sum dim=0 so rows are collapsed: \n{c.sum(dim=0)}")
print(f"Tensor A after sum dim=1 so cols are collapsed: \n{c.sum(dim=1)}")
print(f"Tensor A after sum dim=2 so drawers are collapsed: \n{c.sum(dim=2)}")
print(f"Tensor size on dim = 0 so rows \n{c.size(0)}")
print(f"Tensor size on dim = 1 so cols \n{c.size(1)}")
print(f"Tensor size on dim = 2 so drawers \n{c.size(2)}")
print(f"Tensor size on dim = 2 so drawers \n{c.size(-1)}")
print(f"Tensor mean on dim = 0 so rows \n{c.mean(dim=0, keepdim=False)}")

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")          
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")

# Setting number of batches, inputs, hidden layers and outputs
dtype = torch.float
N = 1
D_in = 4
H1 = 4
H2 = 4
D_out = 4

# Setting input and output tensors
y = torch.randn(N, D_in, device=device, dtype=dtype)

print(y)
y.zero_()
print(y)
y.data(2) = 1
print(y)
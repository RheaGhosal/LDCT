# Get cpu or gpu device for training
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

import torch
import os
batch_size = 20
sequence_length = 5
hidden_size = 5
n_layers = (1, 1)
use_bias = (True, True)
dropout = (0, 0)
gpu = None
train_gaussian_percentage = 0.25
epsilon = 1e-4
alpha = 1e-2
iter_eps_sensor = 1e-2
iter_eps_pump = 1
iter_eps_mv = 0.5

# set device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda" if torch.cuda.is_available() else "cpu"

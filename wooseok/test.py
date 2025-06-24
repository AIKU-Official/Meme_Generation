import os
import torch
import torch.nn as nn

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)  # 출력결과: cuda 
print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 1 (GPU #2 한개 사용하므로)
print('Current cuda device:', torch.cuda.current_device())  # 출력결과: 2 (GPU #2 의미)

#net = ResNet50().to(device)
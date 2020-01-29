import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils import data
from PIL import Image
from dataset import Dataset
from model import Demosaic

BATCH_SIZE = 32
train_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 4}
train_dataset = Dataset(train_ids)
train_generator = data.DataLoader(train_dataset, **params)
test_params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 4}
test_dataset = Dataset(test_ids)
test_generator = data.DataLoader(test_dataset, **test_params)

"""
model parameters
"""
k = 5
temp = 0.5
model = Demosaic(k, temp)

"""
training parameters
"""
learning_rate = 1e-4
epochs = 30

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
    total_loss = 0
    for X in train_generator:
        out = model(X["bayer"], X["r"], X["g"], X["b"])
        loss = criterion(out, X["image"])
        total_loss += (loss / (len(train_dataset)/32))
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print(total_loss)

# test model
errors = 0
count = 0
for X, y in test_generator:
    out = model(X)
    loss = criterion(out, X["image"])
    total_loss += (loss / (len(train_dataset)/32))

    for i in range(out.shape[0]):
        count += 1
        if torch.argmax(out[i]).item() != y[i].item():
            errors += 1

print("count {} wrong {}".format(count, errors))
print(len(test_dataset))
print("error rate: {:.3f}".format(errors/len(dataset)))

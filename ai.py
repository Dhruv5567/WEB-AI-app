import numpy as np
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[0, 0],[0, 1],[1, 0],[1, 1]],dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [1]],dtype = torch.float32)

data = []
for i in range(len(x)):
    data.append((torch.tensor(x[i],dtype = torch.float32),torch.tensor(y[i],dtype = torch.float32)))



model = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    total_loss = 0.0
    for inputs,outputs in data:
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = criterion(prediction,outputs.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

def test_model(x):
    x = torch.tensor(x,dtype = torch.float32)
    output = model(x).round()
    return output.item()

x=int(input("Enter binary digit 1: "))
y=int(input("Enter binary digit 2 :"))

print(test_model([x,y]))

torch.save(model.state_dict(), 'model.pth')



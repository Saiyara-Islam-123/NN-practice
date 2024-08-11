#this isn't CNN
import torch
from torch import nn
from torch import optim

class SupervisedNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = SupervisedNeuralNetwork()
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs = torch.rand(10,10)

labels = torch.rand(torch.Size([10, 2]))

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    print(outputs.shape)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    

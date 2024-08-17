#this isn't CNN
import torch
from torch import nn
from torch import optim

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(50 ,30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU()
        )
        
        self.decoder = torch.nn.Sequential(
            
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30 ,50),
            nn.ReLU(),
            
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
            


class SupervisedNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 10), #input of 20. Output of 10
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
'''    
    
model = SupervisedNeuralNetwork()
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs = torch.rand(32, 1, 20) #batch of size 32. 1 X 20 features

labels = torch.rand(torch.Size([32, 2]))

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    print(outputs.shape)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
  
'''

model = AutoEncoder()
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs = torch.rand(32, 1, 50) #batch of size 32. 1X50 features

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    print(outputs.shape)
    loss = loss_fn(outputs, inputs)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


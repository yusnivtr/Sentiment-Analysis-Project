import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_size,hidden_units=256,output_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_units)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units,hidden_units//2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_units//2,output_size)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    

        
        

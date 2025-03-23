import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    a class for representing a SiameseNetwork
    """

    def __init__(self, inputs=100,fc1=64, fc2=32, fc3=16, output=8):
        super(SiameseNetwork, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(inputs, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, fc3),
            nn.ReLU(),
            nn.Linear(fc3, output),
        )
    
    def forward_once(self, x):
        output = self.fc_layers(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

class ContrastiveLossFunction(torch.nn.Module):
    """
    a class for calculating the loss
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLossFunction, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = (torch.mean((1-label) * torch.pow(distance, 2)) + (label) * 
        torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss
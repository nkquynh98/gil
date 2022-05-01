import torch
import torch.nn as nn
import torch.nn.functional as F


class SetupTableMoveNet(nn.Module):

    def __init__(self, observation_dim = 28, action_dim=2):
        super(SetupTableMoveNet, self).__init__()
        # an affine operation: y = Wx + b
        self.bn1 = nn.BatchNorm1d(observation_dim, affine=True)
        self.fc1 = nn.Linear(observation_dim, 64)  
        self.bn2 = nn.BatchNorm1d(64, affine=True)
        self.fc2 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64, affine=True)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        is_one_dim = False
        if len(x.size())==1:
            is_one_dim = True
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(self.bn1(x)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = self.fc3(self.bn3(x))
        if is_one_dim:
            return x[0]
        else:
            return x



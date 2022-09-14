import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, inputDim, outputDim, hiddenDim=128, layerNum = 2, nonlin=F.relu):
        """
        Inputs:
            inputDim (int): Number of dimensions in input
            outputDim (int): Number of dimensions in output
            hiddenDim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()
        self.fc_first = nn.Linear(inputDim, hiddenDim)
        self.fc_last = nn.Linear(hiddenDim, outputDim)
        self.fcs_middle = [nn.Linear(hiddenDim, hiddenDim) for _ in range(layerNum-1)]
        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h = self.nonlin(self.fc_first(X))
        for fc in self.fcs_middle:
            h = self.nonlin(fc(h))
        out = self.fc_last(h)
        return out
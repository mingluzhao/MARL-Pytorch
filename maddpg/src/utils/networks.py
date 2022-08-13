import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, inputDim, outputDim, hiddenDim=128, layerNum = 2, nonlin=F.relu,
                 constrainOutput=False, normInput=False, isDiscreteAction=True): # TODO changed normInput to False
        """
        Inputs:
            inputDim (int): Number of dimensions in input
            outputDim (int): Number of dimensions in output
            hiddenDim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if normInput:  # normalize inputs
            self.processInput = nn.BatchNorm1d(inputDim)
            self.processInput.weight.data.fill_(1)
            self.processInput.bias.data.fill_(0)
        else:
            self.processInput = lambda x: x
        
        self.fc_first = nn.Linear(inputDim, hiddenDim)
        self.fc_last = nn.Linear(hiddenDim, outputDim)
        self.fcs_middle = [nn.Linear(hiddenDim, hiddenDim) for _ in range(layerNum-1)]
        self.nonlin = nonlin
        
        if constrainOutput and not isDiscreteAction: # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.processOutput = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.processOutput = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h = self.nonlin(self.fc_first(self.processInput(X)))
        for fc in self.fcs_middle:
            h = self.nonlin(fc(h))
        out = self.processOutput(self.fc_last(h))
        return out
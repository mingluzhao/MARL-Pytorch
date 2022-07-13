from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, dimPolicyInput, dimPolicyOutput, dimCriticInput, layerNum = 3, hiddenDim=64,
                 lr=0.01, isDiscreteAction=True):
        """
        Inputs:
            dimPolicyInput (int): number of dimensions for policy input
            dimPolicyOutput (int): number of dimensions for policy output
            dimCriticInput (int): number of dimensions for critic input
        """
        self.policyTrain = MLPNetwork(dimPolicyInput, dimPolicyOutput, layerNum = layerNum,
                                      hiddenDim=hiddenDim, constrainOutput=True, isDiscreteAction=isDiscreteAction)
        self.criticTrain = MLPNetwork(dimCriticInput, 1, layerNum = layerNum, hiddenDim=hiddenDim, constrainOutput=False)
        self.policyTarget = MLPNetwork(dimPolicyInput, dimPolicyOutput, layerNum = layerNum,
                                       hiddenDim=hiddenDim, constrainOutput=True, isDiscreteAction=isDiscreteAction)
        self.criticTarget = MLPNetwork(dimCriticInput, 1, layerNum = layerNum, hiddenDim=hiddenDim, constrainOutput=False)
        hard_update(self.policyTarget, self.policyTrain)
        hard_update(self.criticTarget, self.criticTrain)

        self.policy_optimizer = Adam(self.policyTrain.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.criticTrain.parameters(), lr=lr)
        if not isDiscreteAction:
            self.exploration = OUNoise(dimPolicyOutput)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.isDiscreteAction = isDiscreteAction

    def resetNoise(self):
        if not self.isDiscreteAction:
            self.exploration.reset()

    def scaleNoise(self, scale):
        if self.isDiscreteAction:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def act(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policyTrain(obs)
        if self.isDiscreteAction:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policyTrain': self.policyTrain.state_dict(),
                'criticTrain': self.criticTrain.state_dict(),
                'policyTarget': self.policyTarget.state_dict(),
                'criticTarget': self.criticTarget.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policyTrain.load_state_dict(params['policyTrain'])
        self.criticTrain.load_state_dict(params['criticTrain'])
        self.policyTarget.load_state_dict(params['policyTarget'])
        self.criticTarget.load_state_dict(params['criticTarget'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

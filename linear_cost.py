import torch
import torch.nn as nn

import numpy as np
import pdb

UPDATE_TYPES = ['exact', 'geometric', 'decay', 'decay_sqrt', 'ficticious', 'gd', 'polynomial', 'exponential']

class RBFLinearCost:
    """
    MMD cost implementation with rff feature representations
    TODO: Currently hardcoded to cpu....ok for now
    :param expert_data: (torch Tensor) expert data used for feature matching
    :param feature_dim: (int) feature dimension for rff
    :param input_type: (str) state (s), state-action (sa), state-next state (ss),
                       state-action-next state (sas)
    :param update_type: (str) exact, geometric, decay, decay_sqrt, ficticious
    :param cost_range: (list) inclusive range of costs
    :param bw_quantile: (float) quantile used to fit bandwidth for rff kernel
    :param bw_samples: (int) number of samples used to fit bandwidth
    :param lambda: (float) weight parameter for bonus and cost
    :param lr: (float) learning rate for discriminator/cost update. 0.0 = closed form update
    :param seed: (int) random seed to set cost function
    """
    def __init__(self,
                 expert_data,
                 device,
                 feature_dim=1024,
                 input_type='sa',
                 update_type='exact',
                 cost_range=[-1.,0.],
                 bw_quantile=0.1,
                 bw_samples=100000,
                 lambda_b=1.0,
                 lr=0.0,
                 T=400,
                 seed=100):

        # Set Random Seed 
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.expert_data = expert_data
        input_dim = expert_data.size(1)
        self.input_type = input_type
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.cost_range = cost_range
        if cost_range is not None:
            self.c_min, self.c_max = cost_range
        self.lambda_b = lambda_b
        self.lr = lr
        self.device = device

        # Fit Bandwidth
        self.quantile = bw_quantile
        self.bw_samples = bw_samples
        #self.bw = self.fit_bandwidth(expert_data)

        # Define Phi and Cost weights
        self.rff = nn.Linear(input_dim, feature_dim)
        self.rff.to(device)
        self.rff.bias.data = (torch.rand_like(self.rff.bias.data)-0.5)*2.0*torch.pi
        self.init_rand_weight = torch.rand_like(self.rff.weight.data)
        self.init_rand_weight.to(device)
        self.update_bandwidth(self.expert_data)
        #self.rff.weight.data = torch.rand_like(self.rff.weight.data)/(self.bw+1e-8)

        # W Update Init
        self.T = T
        self.w_bar = None
        self.w = None
        self.w_ctr, self.w_sum = 0, 0
        self.update_type = update_type
        if self.update_type not in UPDATE_TYPES:
            raise NotImplementedError("This update type is not available")

        # Compute Expert Phi Mean
        self.expert_rep = self.get_rep(expert_data)
        self.phi_e = self.expert_rep.mean(dim=0)

    def get_rep(self, x):
        with torch.no_grad():
            out = self.rff(x)
            out = torch.cos(out)*torch.sqrt(2/self.feature_dim)
        return out

    def update_expert_data(self,data):
        self.expert_data = data

    def update_bandwidth(self, data):
        num_data = data.shape[0]
        idxs_0 = torch.randint(low=0, high=num_data, size=(self.bw_samples,))
        idxs_1 = torch.randint(low=0, high=num_data, size=(self.bw_samples,))
        norm = torch.norm(data[idxs_0, :]-data[idxs_1, :], dim=1)
        bw = np.quantile(norm.numpy(), q=self.quantile)
        self.rff.weight.data = self.init_rand_weight/(bw+1e-8)
        #return bw

    def update(self, policy_buffer):
        policy_data = torch.cat([policy_buffer.states[:-1], policy_buffer.actions], dim=-1).view(-1, self.input_dim)
        #TODO: Should we update bandwidth??

        phi = self.get_rep(policy_data).mean(0)
        feat_diff = phi - self.phi_e

        self.w = feat_diff

        return {'mmd': torch.dot(self.w, feat_diff).item()}

    def get_costs(self, x):
        data = self.get_rep(x)
        if self.cost_range is not None:
            return torch.clamp(torch.matmul(data, self.w).unsqueeze(-1), self.c_min, self.c_max)
        return torch.matmul(data, self.w).unsqueeze(-1)

    def get_expert_cost(self):
        return torch.clamp(torch.mm(self.expert_rep, self.w.unsqueeze(1)), self.c_min, self.c_max).mean()

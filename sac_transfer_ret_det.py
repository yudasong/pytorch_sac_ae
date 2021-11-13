import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder
from decoder import make_decoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)



class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0])
        )

        self.apply(weight_init)

    def forward(
        self, obs, std, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu = self.trunk(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

    def log(self, L, step, log_freq=LOG_FREQ):

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)

class Ag_Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)

        self.outputs['q1'] = q1

        return q1

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class SacTransferReturnDetAgent(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='identity',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        no_entropy=True,
        stddev_schedule='linear(1.0,0.1,100000)',
        stddev_clip=0.3
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.encoder_type = encoder_type

        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.ag_critic = Ag_Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.actor.encoder.copy_conv_weights_from(self.ag_critic.encoder)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.ag_critic_optimizer = torch.optim.Adam(
            self.ag_critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.ag_critic.train(training)


    def warm_start_from(self,expert):
        self.critic.encoder.duplicate_conv_weights_from(expert.critic.encoder)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def act(self, obs, step, eval_mode=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(obs, stddev)
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
            
        return action.cpu().numpy().flatten()


    def update_ag_critic(self, expert, obs, action, reward, next_obs, not_done, values, L, step):
        with torch.no_grad():
           target_Q = reward + (not_done * self.discount * values)

        
        # with torch.no_grad():

        #     #target_V1 = torch.zeros(len(obs),1).to(self.device)
        #     #target_V2 = torch.zeros(len(obs),1).to(self.device)
        #     #for i in range(10):
        #     policy_action,_, log_pi, _ = expert.actor(next_obs)
        #     target_Q1, target_Q2 = expert.critic(next_obs, policy_action)
        #     #target_V1 = target_V1 + target_Q1 - expert.alpha.detach() * log_pi
        #     #target_V2 = target_V2 + target_Q2 - expert.alpha.detach() * log_pi
        #     target_V = torch.min(target_Q1, target_Q2) 

        #     #target_V = bc_agent.value_net(next_obs)
        #     target_Q = reward + (not_done * self.discount * target_V)

        
        # with torch.no_grad():

        #     target_V1 = torch.zeros(len(obs),1).to(self.device)
        #     target_V2 = torch.zeros(len(obs),1).to(self.device)
        #     for i in range(10):
        #         _, policy_action, log_pi, _ = expert.actor(next_obs)
        #         target_Q1, target_Q2 = expert.critic(next_obs, policy_action)
        #         target_V1 = target_V1 + target_Q1 
        #         target_V2 = target_V2 + target_Q2 
        #     target_V = torch.min(target_V1, target_V2) / 10
        # target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        #current_Q1, current_Q2 = self.critic(obs, action)
        current_Q1 = self.ag_critic(obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,target_Q) 
        L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.ag_critic_optimizer.zero_grad()
        critic_loss.backward()

        total_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.ag_critic.parameters())):
            #print(p)
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5

        L.log('train_grad/norm', total_norm, step)

        torch.nn.utils.clip_grad_norm(self.ag_critic.parameters(), 20)

        self.ag_critic_optimizer.step()

        self.ag_critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        #_, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)

        ag_Q = self.ag_critic(obs, action, detach_encoder=True)

        actor_loss = -ag_Q.mean()
        

        L.log('train_actor/loss', actor_loss, step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor.log(L, step)

    def update(self, replay_buffer, expert, L, step, total_steps=0):
        #for _ in range(5):
        #obs, state, action, reward, next_obs, next_state, not_done, values = replay_buffer.sample()
        obs, state, action, reward, next_obs, next_state, not_done, values = replay_buffer.sample()


        L.log('train/batch_reward', reward.mean(), step)

        if self.encoder_type == 'identity':
            self.update_ag_critic(expert, state, action, reward, next_state, not_done, values, L, step)
        else:
            self.update_ag_critic(expert, obs, action, reward, next_obs, not_done, values, L, step)

        if step % self.actor_update_freq == 0:
            
            if self.encoder_type == 'identity':
                self.update_actor_and_alpha(state, L, step)
            else:
                self.update_actor_and_alpha(obs, L, step)


    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.ag_critic.state_dict(), '%s/ag_critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step, no_entropy=False, post_step=500000):
        #self.actor.load_state_dict(
        #    torch.load('%s/actor_%s.pt' % (model_dir, step), map_location=self.device)
        #)

        self.ag_critic.load_state_dict(
                 torch.load('%s/det_critic_%s.pt' % (model_dir, post_step), map_location=self.device)
             )
        
        #pass

        # else:
        #     self.ag_critic.load_state_dict(
        #         torch.load('%s/critic_%s.pt' % (model_dir, step), map_location=self.device)
        #    )
        #self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        #pass

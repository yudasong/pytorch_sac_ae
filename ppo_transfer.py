import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from torch.utils.data import DataLoader

import utils
from encoder import make_encoder
from decoder import make_decoder

LOG_FREQ = 15000

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

def get_flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_shape[0])
        )

        self.sigma = nn.Parameter(torch.zeros(action_shape[0], 1))

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)
        mu = self.trunk(obs)
        shape = [1]*len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp() # using exp instead of softplus...
        return mu, sigma

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)

class PPOTransferAgent(object):
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 gamma=0.99,
                 gae_lam=0.95,
                 clip_eps=0.2,
                 max_norm=0.5,
                 lr=3e-4,
                 weight_ent=0.0,
                 num_updates=10,
                 batch_size=64,
                 hidden_dim=64,
                 grad_norm=True,
                 encoder_type='identity',
                 encoder_feature_dim=50,
                 num_layers=4,
                 num_filters=32):


        self.device = device
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.clip_eps = clip_eps
        self.batch_size = batch_size
        self.num_updates = num_updates
        self.grad_norm = grad_norm
        self.max_norm = max_norm
        self.weight_ent = weight_ent

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        # TODO: learning rate scheduler....
        self.optim = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.ag_critic.train(training)

    def dist(self, *logits):
        return Independent(Normal(*logits), 1)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            logits = self.actor(obs, compute_log_pi=False)
            dist = self.dist(*logits)
            pi = dist.sample()
            return pi.cpu().data.numpy().flatten()

    def compute_returns_and_advantages(self, obs, next_obs, rewards, not_dones, expert):
        """
        From the expert V_sim, uses GAE to compute advantages and episodic discounted returns for on-policy samples
        """
        # Get v_s and v_s+1 from v_sim
        with torch.no_grad():
            v_s1, v_s2  = torch.zeros(len(obs), 1).to(self.device), torch.zeros(len(obs), 1).to(self.device)
            v_s1_, v_s2_  = torch.zeros(len(obs), 1).to(self.device), torch.zeros(len(obs), 1).to(self.device)
            for _ in range(10):
                _, policy_action, log_pi, _ = expert.actor(obs)
                _, policy_action_, log_pi_, _ = expert.actor(next_obs)
                target_q1, target_q2 = expert.critic(obs, policy_action)
                target_q1_, target_q2_ = expert.critic(next_obs, policy_action_)
                v_s1 = v_s1 + target_q1 - expert.alpha.detach() * log_pi
                v_s2 = v_s2 + target_q2 - expert.alpha.detach() * log_pi
                v_s1_ = v_s1_ + target_q1_ - expert.alpha.detach() * log_pi_
                v_s2_ = v_s2_ + target_q2_ - expert.alpha.detach() * log_pi_
            v_s = torch.min(v_s1, v_s2) / 10
            v_s_ = torch.min(v_s1_, v_s2_) / 10

        advantages = torch.zeros_like(rewards)
        delta = rewards + v_s_ * self.gamma - v_s
        m = not_dones * (self.gamma * self.gae_lam)
        for i in range(len(rewards) - 1, -1, -1):
            gae = delta[i] + m[i] * gae
            advantages[i] = gae
        returns = advantages + v_s
        return returns, advantages

    def compute_logp_old(self, obs, act):
        with torch.no_grad():
            logits = self.actor.forward(obs)
            return self.dist(*logits).log_prob(act)

    def update(self, replay_buffer, bc_agent, expert, L, step, total_steps=0):
        # NOTE: currently not doing dual clipping trick, reward normalization, or learning rate scheduler
        obs, state, action, reward, next_obs, next_state, not_done = replay_buffer.sample()
        L.log('train/batch_reward', reward.mean(), step)
        _, advantages = self.compute_returns_and_advantages(obs, next_obs, reward, not_done, expert)
        logp_old = self.compute_logp_old(obs, action)

        losses, clip_losses, ent_losses = [], [], []

        loader = DataLoader(list(zip(obs, action, logp_old, advantages)), shuffle=True, batch_size=self.batch_size)

        for _ in range(self.num_updates):
            for o, a, old_logp, adv in loader:
                logits = self.actor.forward(o)
                dist = self.dist(*logits)

                # Normalize Batch Advantages
                adv_mean, adv_std = adv.mean(), adv.std()
                adv = (adv - adv_mean)/adv_std

                ratio = (dist.log_prob(a) - old_logp).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * adv
                surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv

                pg_loss = -torch.min(surr1, surr2).mean()

                ent_loss = dist.entropy().mean()

                loss = pg_loss - self.weight_ent * ent_loss

                self.optim.zero_grad()
                loss.backward()

                if self.grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), max_norm=self.max_norm
                    )

                self.optim.step()
                clip_losses.append(pg_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())


class TRPOTransferAgent(object):
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 gamma=0.99,
                 max_kld=1e-2,
                 weight_ent=0.0,
                 hidden_dim=64,
                 damping=0.1,
                 cg_steps=10,
                 encoder_type='identity',
                 encoder_feature_dim=50,
                 num_layers=4,
                 num_filters=32):


        self.device = device
        self.gamma = gamma
        self.weight_ent = weight_ent
        self.damping = damping
        self.cg_steps = cg_steps
        self.max_kld = max_kld

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.ag_critic.train(training)

    def dist(self, *logits):
        return Independent(Normal(*logits), 1)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            logits = self.actor(obs, compute_log_pi=False)
            dist = self.dist(*logits)
            pi = dist.sample()
            return pi.cpu().data.numpy().flatten()

    def compute_returns_and_advantages(self, obs, next_obs, rewards, not_dones, expert):
        #TODO: This is where you would put in the critic!
        returns, advantages = None, None
        return returns, advantages

    @staticmethod
    def get_conjugate_gradient(Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x
    
    def linesearch(self, f, init_params, fullstep, expected_improve_rate, max_backtracks=10):
        with torch.no_grad():
            fval = f()
            for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
                new_params = init_params + stepfrac * fullstep
                set_flat_params(self.actor, new_params)
                newfval = f()
                actual_improve = fval - newfval
                expected_improve = expected_improve_rate * stepfrac
                ratio = actual_improve / expected_improve
                if self.verbose > 0:
                    logger.log("a/e/r ", actual_improve.item(), expected_improve.item(), ratio.item())
                if ratio.item() > self.linesearch_accepted_ratio and actual_improve.item() > 0:
                    return True, new_params
            return False, init_params
    
    def compute_logp_old(self, obs, act):
        with torch.no_grad():
            logits = self.actor.forward(obs)
            return self.dist(*logits).log_prob(act)

    def update(self, replay_buffer, bc_agent, expert, L, step, total_steps=0):
        # NOTE: currently not doing dual clipping trick, reward normalization, or learning rate scheduler
        obs, state, action, reward, next_obs, next_state, not_done = replay_buffer.sample()
        L.log('train/batch_reward', reward.mean(), step)
        _, advantages = self.compute_returns_and_advantages(obs, next_obs, reward, not_done, expert)

        losses = []

        # Start of TRPO update
        logp_old = self.compute_logp_old(obs, action)

        def get_action_loss():
            logits = self.actor.forward(obs)
            dist = self.dist(*logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            action_loss_ = - advantages * torch.exp(log_prob - logp_old) - self.weight_ent * entropy
            return action_loss_.mean()

        def get_kl():
            action_means, action_stds = self.actor.forward(obs)
            action_logstds = torch.log(action_stds)

            fixed_action_means = action_means.detach()
            fixed_action_logstds = action_logstds.detach()
            fixed_action_stds = action_stds.detach()
            kl = action_logstds - fixed_action_logstds + \
                 (fixed_action_stds.pow(2) + (fixed_action_means - action_means).pow(2)) / \
                 (2.0 * action_stds.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        action_loss = get_action_loss()
        action_loss_grad = torch.autograd.grad(action_loss, self.actor.parameters())
        flat_action_loss_grad = torch.cat([grad.view(-1) for grad in action_loss_grad]).data
        
        def Fvp(v):
            kl = get_kl()
            kl = kl.mean()

            kld_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
            flat_kld_grad = torch.cat([grad.view(-1) for grad in kld_grad])

            kl_v = (flat_kld_grad * v).sum()
            kld_grad_grad = torch.autograd.grad(kl_v, self.actor.parameters())
            flat_kld_grad_grad = torch.cat([grad.contiguous().view(-1) for grad in kld_grad_grad]).data

            return flat_kld_grad_grad + v * self.damping

        stepdir = self.get_conjugate_gradient(Fvp, -flat_action_loss_grad, self.cg_steps)

        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0)

        lm = torch.sqrt(shs / self.max_kld)
        fullstep = stepdir / lm

        neggdotstepdir = (-flat_action_loss_grad * stepdir).sum(0, keepdim=True)
        prev_params = get_flat_params(self.actor)
        success, new_params = self.linesearch(get_action_loss, prev_params, fullstep, neggdotstepdir / lm)
        set_flat_params(self.actor, new_params)



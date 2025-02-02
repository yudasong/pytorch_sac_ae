import torch
import numpy as np
import torch.nn as nn
import gym
import os
import re
from collections import deque
import random
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from subproc_vec_env import SubprocVecEnv

import dmc2gym
                                             
def make_env(args):
    def _thunk():
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed,
            visualize_reward=False,
            spec=args.env_spec,
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )
        env.seed(args.seed)
        #env.physics.model.opt.gravity[2] = args.gravity
        return env
    return _thunk

def make_vec_envs(args):
    envs = [
        make_env(args)
        for i in range(args.num_vec_envs)
    ]

    envs = SubprocVecEnv(envs)
    return envs


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        print("dir not created!")
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, state_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.uint8

        #self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        #self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, state, action, reward, next_obs, next_state, done):
        #np.copyto(self.obses[self.idx], obs)
        np.copyto(self.states[self.idx],state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        #np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_states[self.idx],next_state)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def get_recent_next_states(self, discard_random=True):
        if discard_random:
            if self.full:
                return self.next_states[1000:]
            else:
                return self.next_states[1000:self.idx]
        else:
            if self.full:
                return self.next_states
            else:
                return self.next_states[:self.idx]

    def get_recent_states(self, discard_random=False):
        if discard_random:
            if self.full:
                return self.states[1000:]
            else:
                return self.states[1000:self.idx]
        else:
            if self.full:
                return self.states
            else:
                return self.states[:self.idx]

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        #obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        states = torch.as_tensor(self.states[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        #next_obses = torch.as_tensor(
        #    self.next_obses[idxs], device=self.device
        #).float()
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        #return obses,states, actions, rewards, next_obses, next_states, not_dones
        return states,states, actions, rewards, next_states, next_states, not_dones

    def get_episode(self):
        states = torch.as_tensor(self.states, device=self.device).float()
        actions = torch.as_tensor(self.actions, device=self.device)

        return states, actions, self.rewards

    def update_reward(self, cost_function):
        states = torch.as_tensor(self.states[:self.idx], device=self.device).float()
        costs = -1.0 * cost_function.get_costs(states).cpu().numpy()
        self.rewards[:self.idx] = costs
        
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            #self.obses[self.last_save:self.idx],
            self.states[self.last_save:self.idx],
            #self.next_obses[self.last_save:self.idx],
            self.next_states[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            #self.obses[start:end] = payload[0]
            self.states[start:end] = payload[0]
            #self.next_obses[start:end] = payload[2]
            self.next_states[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end


class ValueReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, state_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.uint8

        #self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        #self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.values = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, state, action, reward, next_obs, next_state, done, value):
        #np.copyto(self.obses[self.idx], obs)
        np.copyto(self.states[self.idx],state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        #np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_states[self.idx],next_state)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.values[self.idx], value)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def get_recent_next_states(self, discard_random=True):
        if discard_random:
            if self.full:
                return self.next_states[1000:]
            else:
                return self.next_states[1000:self.idx]
        else:
            if self.full:
                return self.next_states
            else:
                return self.next_states[:self.idx]

    def get_recent_states(self, discard_random=False):
        if discard_random:
            if self.full:
                return self.states[1000:]
            else:
                return self.states[1000:self.idx]
        else:
            if self.full:
                return self.states
            else:
                return self.states[:self.idx]

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        #obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        states = torch.as_tensor(self.states[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        #next_obses = torch.as_tensor(
        #    self.next_obses[idxs], device=self.device
        #).float()
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        values = torch.as_tensor(self.values[idxs], device=self.device)

        #return obses,states, actions, rewards, next_obses, next_states, not_dones
        return states,states, actions, rewards, next_states, next_states, not_dones, values

    def get_episode(self):
        states = torch.as_tensor(self.states, device=self.device).float()
        actions = torch.as_tensor(self.actions, device=self.device)

        return states, actions, self.rewards

    def update_reward(self, cost_function):
        states = torch.as_tensor(self.states[:self.idx], device=self.device).float()
        costs = -1.0 * cost_function.get_costs(states).cpu().numpy()
        self.rewards[:self.idx] = costs
        
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            #self.obses[self.last_save:self.idx],
            self.states[self.last_save:self.idx],
            #self.next_obses[self.last_save:self.idx],
            self.next_states[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            #self.obses[start:end] = payload[0]
            self.states[start:end] = payload[0]
            #self.next_obses[start:end] = payload[2]
            self.next_states[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end


class MultiStepReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, state_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.uint8

        #self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        #self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.targets = np.empty((capacity, 1), dtype=np.float32)                    
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, state, action, reward, next_obs, next_state, done):
        #np.copyto(self.obses[self.idx], obs)
        np.copyto(self.states[self.idx],state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        #np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_states[self.idx],next_state)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample_single(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        #obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        states = torch.as_tensor(self.states[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        #next_obses = torch.as_tensor(
        #    self.next_obses[idxs], device=self.device
        #).float()
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        #return obses,states, actions, rewards, next_obses, next_states, not_dones
        return states,states, actions, rewards, next_states, next_states, not_dones

    def sample(self, h=5):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx - h, size=self.batch_size
        )

        #obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        states = torch.as_tensor(self.states[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = []
        next_states = []
        not_dones = []
        for i in range(h):
            rewards.append(torch.as_tensor(self.rewards[idxs+i], device=self.device))
            next_states.append(torch.as_tensor(self.next_states[idxs+i], device=self.device).float())
            not_dones.append(torch.as_tensor(self.not_dones[idxs+i], device=self.device))

        #return obses,states, actions, rewards, next_obses, next_states, not_dones
        return states,states, actions, rewards, next_states, next_states, not_dones

    def get_episode(self):
        states = torch.as_tensor(self.states, device=self.device).float()
        actions = torch.as_tensor(self.actions, device=self.device)

        return states, actions, self.rewards

    def update_reward(self, cost_function):
        states = torch.as_tensor(self.states[:self.idx], device=self.device).float()
        costs = -1.0 * cost_function.get_costs(states).cpu().numpy()
        self.rewards[:self.idx] = costs
        



class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs, state = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), state

    def step(self, action):
        obs, state, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), state, reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

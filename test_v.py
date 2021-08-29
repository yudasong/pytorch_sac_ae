import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy

import utils
from logger import Logger
from video import VideoRecorder

from sac_ae import SacAeAgent, BCAgent
from sac_transfer import SacTransferAgent

import scipy.signal

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--env_spec', default='full')
    
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000, type=int)

    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--num_train_steps', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--expert_encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--expert_decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_type', default='identity', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--expert_dir', default='.', type=str)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')

    parser.add_argument('--gravity', default=-9.8, type=float)

    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step):
    for i in range(num_episodes):
        obs, state = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                if agent.encoder_type == 'identity':
                    action = agent.select_action(state)
                else:
                    action = agent.select_action(obs)
            obs, state, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
    L.dump(step)



def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'sac_ae':
        return SacAeAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.expert_encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.expert_decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_lambda=args.decoder_weight_lambda,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

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

    # stack several consecutive frames together
    #if args.encoder_type == 'pixel':
    env = utils.FrameStack(env, k=args.frame_stack)

    from soco_device import DeviceCheck

    dc = DeviceCheck()
    # will return a device name ('cpu'/'cuda') and a list of gpu ids, if any
    device_name, device_ids = dc.get_device(n_gpu=1)

    if len(device_ids) == 1:
        device_name = '{}:{}'.format(device_name, device_ids[0])
        device = torch.device(device_name)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        state_shape=env.state_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )
    
    if args.expert_encoder_type == 'pixel':

        agent = make_agent(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=device
        )


    else:
        agent = make_agent(
            obs_shape=env.state_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=device)


    agent.load(os.path.join(args.expert_dir, 'model'),990000)
    

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:

            obs, state = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            

            if episode >= 1:
                states, actions, rewards = replay_buffer.get_episode()
                
                with torch.no_grad():
                    current_Q1, current_Q2 = agent.critic(states, actions)


                returns = discount_cumsum(np.squeeze(rewards), agent.discount)
                target_Q  = returns + np.squeeze(np.array(log_pis))

                print("current:")
                print(current_Q1.cpu().data.numpy()[:100])
                print("target:")
                print(target_Q[:100])

                print((current_Q1.cpu().data.numpy().squeeze() - target_Q)[:100])
                #print(current_Q2.cpu().data.numpy() - target_Q)

            episode += 1
            log_pis = []
        
        else:
            with utils.eval_mode(agent):

                if step < 20:
                    action = env.action_space.sample()  
                    log_pi = [0]  
                else:
                    if agent.encoder_type == 'identity':
                        action, log_pi = agent.sample_action_with_logpi(state)
                    else:
                        action, log_pi = agent.sample_action_with_logpi(obs)
        # run training update

            next_obs, next_state, reward, done, _ = env.step(action)
            reward = reward - agent.alpha.cpu().detach() * log_pi[0]
            log_pis.append(log_pi)

            # allow infinit bootstrap
            done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
                done
            )

            #episode_reward += reward
            replay_buffer.add(obs, state, action, reward, next_obs, next_state, done_bool)

            obs = next_obs
            state = next_state
            episode_step += 1
            step += 1
    

if __name__ == '__main__':
    main()

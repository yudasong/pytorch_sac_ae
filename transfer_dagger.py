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

from sac_imitation import SacAeImitationAgent
from sac_ae import SacAeAgent, BCAgent
from sac_transfer import SacTransferAgent

from linear_cost import RBFLinearCost

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
    parser.add_argument('--replay_buffer_capacity', default=150000, type=int)
    parser.add_argument('--expert_replay_buffer_capacity', default=1000000, type=int)
    parser.add_argument('--imitation_replay_buffer_capacity', default=500000, type=int)
    parser.add_argument('--imitation_rollout_buffer_capacity', default=100000, type=int)

    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--init_random_steps', default=1000, type=int)
    parser.add_argument('--init_expert_steps', default=2000, type=int)

    parser.add_argument('--num_train_steps', default=150000, type=int)

    parser.add_argument('--num_post_q_updates', default=5000, type=int)
    parser.add_argument('--initial_imitation_episode', default=10,type=int)
    parser.add_argument('--num_imitation_train_steps', default=40000, type=int)
    parser.add_argument('--num_imitation_rollout_steps', default=20000, type=int)
    parser.add_argument('--imitation_freq', default=2, type=int)

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
    parser.add_argument('--init_temperature', default=0.1, type=float)
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

def make_bcagent(obs_shape, action_shape, args, device):
    return BCAgent(
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
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters
    )

def make_transfer_agent(obs_shape, action_shape, args, device):
    return SacTransferAgent(
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
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters
    )

def make_imitation_agent(obs_shape, action_shape, args, device):
    return SacAeImitationAgent(
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

def dac(agent, env, cost_function, imitation_replay_buffer, L, episode, args):
    if episode == args.initial_imitation_episode:
        num_imitation_train_steps = args.num_imitation_train_steps * 10
    else:
        num_imitation_train_steps = args.num_imitation_train_steps
    episode, episode_reward, done = 0, 0, True
    
    start_time = time.time()
    for step in range(num_imitation_train_steps):
        if done:
            #if step > 0:
                # L.log('train/duration', time.time() - start_time, step)
                # start_time = time.time()
                # L.dump(step)

            obs, state = env.reset()
            done = False
            episode_step = 0
            episode += 1

            if imitation_replay_buffer.idx > 0:
                recent_states = imitation_replay_buffer.get_recent_states()
                mmd_loss = cost_function.update(recent_states)

                L.log('imitation/mmd_loss', mmd_loss, step)

                imitation_replay_buffer.update_reward(cost_function)
                

                L.log('imitation/episode', episode, step)
                L.log('imitation/episode_reward', episode_reward, step)
                L.dump(step)

            episode_reward = 0

        # sample action for data collection
        if imitation_replay_buffer.idx < args.init_random_steps and episode == args.initial_imitation_episode:
        #if False:
            action = env.action_space.sample()                
        
        else:
            with utils.eval_mode(agent):

                if agent.encoder_type == 'identity':
                    action = agent.sample_action(state)
                else:
                    action = agent.sample_action(obs)
        
        if imitation_replay_buffer.idx > args.init_random_steps or episode > args.initial_imitation_episode:
            num_updates = args.init_random_steps if (imitation_replay_buffer.idx == args.init_random_steps and episode == args.initial_imitation_episode) else 1
        
            for _ in range(num_updates):
                agent.update(imitation_replay_buffer, L, step)
                #expert_agent.update(replay_buffer, L, step)
        next_obs, next_state, raw_reward, done, _ = env.step(action)
        reward = -1.0 * cost_function.get_single_cost(state)
        #print(reward)
        #print(reward)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += raw_reward
        imitation_replay_buffer.add(obs, state, action, reward, next_obs, next_state, done_bool)

        obs = next_obs
        state = next_state
        episode_step += 1

def rollout(agent, replay_buffer, env, args):
    episode, episode_reward, done = 0, 0, True
    for step in range(args.num_imitation_rollout_steps):
        if done:
            obs, state = env.reset()
            done = False
            print("rollout reward: {}".format(episode_reward))
            episode_reward = 0
            episode_step = 0
            episode += 1

        with utils.eval_mode(agent):
            if np.random.rand() > 0.05:
            #if True:
                if agent.encoder_type == 'identity':
                    action = agent.sample_action(state)
                else:
                    action = agent.sample_action(obs)
            else:
                action = env.action_space.sample()
        
        next_obs, next_state, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, state, action, reward, next_obs, next_state, done_bool)

        obs = next_obs
        state = next_state
        episode_step += 1


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
    env.physics.model.opt.gravity[2] = args.gravity

    source_env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        spec=args.env_spec,
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat
    )
    source_env.seed(args.seed)

    # stack several consecutive frames together
    #if args.encoder_type == 'pixel':
    env = utils.FrameStack(env, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

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

    imitation_replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        state_shape=env.state_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.imitation_replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    imitation_rollout_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        state_shape=env.state_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.imitation_rollout_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    expert_replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        state_shape=env.state_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.expert_replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )


    if args.expert_encoder_type == 'pixel':

        expert_agent = make_agent(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=device
        )

        imitation_agent = make_imitation_agent(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=device
        )


    else:
        expert_agent = make_agent(
            obs_shape=env.state_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=device)

        imitation_agent = make_imitation_agent(
            obs_shape=env.state_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=device
        )


    if args.encoder_type == 'pixel':
        bc_agent = make_bcagent(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=device
        )
        #bc_agent.load(os.path.join(args.work_dir, 'bc_model'),90)
        #print("bc loaded.")

        agent = make_transfer_agent(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=device
        )
    else:

        bc_agent = make_bcagent(
            obs_shape=env.state_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=device
        )
        #bc_agent.load(os.path.join(args.work_dir, 'bc_model_state'),90)
        #print("bc loaded.")

        agent = make_transfer_agent(
            obs_shape=env.state_space.shape,
            action_shape=env.action_space.shape,
            args=args,
            device=device
        )



    expert_agent.load(os.path.join(args.expert_dir, 'model'),990000)
    expert_replay_buffer.load(os.path.join(args.expert_dir, 'buffer'))
    expert.set_zero_alpha()

    #if args.expert_encoder_type == 'pixel':
    #    agent.warm_start_from(expert_agent)
    agent.load(os.path.join(args.expert_dir, 'model'),990000)
    agent.set_zero_alpha()


    print("expert loaded.")


    L = Logger(args.work_dir, use_tb=args.save_tb)

    #L.log('eval/episode', 0, 0)
    #evaluate(env, bc_agent, video, args.num_eval_episodes, L, 0)


    cost_function = None

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if step > 0:

                L.log('train/duration', time.time() - start_time, step)
                L.log('train/episode_reward', episode_reward, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if step % args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                #evaluate(env, expert_agent, video, args.num_eval_episodes, L, step)
                evaluate(env, agent, video, args.num_eval_episodes, L, step)
                if args.save_model:
                    #expert_agent.save(model_dir, step)
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            

            obs, state = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            

            if not cost_function and episode >= args.initial_imitation_episode:
                recent_states = replay_buffer.get_recent_states()
                cost_function = RBFLinearCost(recent_states, device, seed=args.seed)

            elif episode >= args.initial_imitation_episode:
                recent_states = replay_buffer.get_recent_states()
                cost_function.update_bandwidth(recent_states)
                cost_function.update_expert_data(recent_states)
                

            if cost_function and episode % args.imitation_freq == 0:
                dac(imitation_agent, source_env, cost_function, imitation_replay_buffer, L, episode, args)
                #rollout(imitation_agent, imitation_rollout_buffer, source_env, args)
                
                recent_states = imitation_rollout_buffer.get_recent_states()
                mmd = cost_function.get_mmd(recent_states)
                print("mmd: {}".format(mmd))

                for s in range(args.num_post_q_updates):
                    expert_agent.post_update_critic(expert_replay_buffer, imitation_replay_buffer,s)
                
                expert_agent.save_post_critics(args.work_dir, step)


            L.log('train/episode', episode, step)




        # sample action for data collection
        if step < args.init_random_steps:
        #if False:
            action = env.action_space.sample()        
        
        elif step < args.init_expert_steps:
            
            if expert_agent.encoder_type == 'identity':
                action = expert_agent.sample_action(state)
            else:
                action = expert_agent.sample_action(obs)
        
        
        else:
            with utils.eval_mode(agent):
                '''
                if expert_agent.encoder_type == 'identity':
                    action = expert_agent.sample_action(state)
                else:
                    action = expert_agent.sample_action(obs)
                '''
                if np.random.rand() > 0.05:
                #if True:
                    if agent.encoder_type == 'identity':
                        action = agent.sample_action(state)
                    else:
                        action = agent.sample_action(obs)
                else:
                    action = env.action_space.sample()

        # run training update

        
        if step >= args.init_expert_steps:
            num_updates = args.init_expert_steps if step == args.init_expert_steps else 1
        

        #if step >= args.init_random_steps:
        #    num_updates = args.init_random_steps if step == args.init_random_steps else 1
        
            for _ in range(num_updates):
                agent.update(replay_buffer, bc_agent, expert_agent, L, step, total_steps = args.num_train_steps)
                #expert_agent.update(replay_buffer, L, step)
        next_obs, next_state, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, state, action, reward, next_obs, next_state, done_bool)

        obs = next_obs
        state = next_state
        episode_step += 1
    

    step += 1
    L.log('eval/episode', episode, step)
    #evaluate(env, expert_agent, video, args.num_eval_episodes, L, step)
    evaluate(env, agent, video, args.num_eval_episodes, L, step)
    if args.save_model:
        #expert_agent.save(model_dir, step)
        agent.save(model_dir, step)
        imitation_agent.save(model_dir, step)
    if args.save_buffer:
        replay_buffer.save(buffer_dir)

if __name__ == '__main__':
    main()

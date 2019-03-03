import argparse
import numpy as np
import mujoco_py
import gym
import tensorflow as tf
from Actor import *
from Critic import *
from OU_Noise import *
from tqdm import tqdm
from collections import deque
import random
import os
import shutil


# function to organize the selected elements from the batch
def batch_selector(batch):
    states = np.array([i[0] for i in batch])
    actions = np.array([i[1] for i in batch])
    rews = np.array([i[2] for i in batch])
    obss = np.array([i[3] for i in batch])
    dones = np.array([i[4] for i in batch])
    return states, actions, rews, obss, dones

# create the folder to store the saved networks
if not os.path.exists("networks"):
    os.mkdir("networks")
    print('Directory created!')
else:
    print('Directory already exists')

# to reset the tensorboard stuff 
if not os.path.exists("stats/"):
    os.mkdir('stats/')
else:
    shutil.rmtree('stats/')
    os.mkdir('stats/')

env = gym.make('HalfCheetah-v2')
# env = gym.make('Hopper-v2')
# env = gym.make('Pendulum-v0')


# parameters initialization
num_episodes = 50000
max_steps = 1000
buffer_size = 100000
batch_size = 32
gamma = 0.99

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
buffer_replay = deque()

noise = OU_Noise(action_size)
# tensorflow session
sess = tf.Session()
actor = Actor(action_size, state_size, batch_size, sess)
critic = Critic(action_size, state_size, actor.num_trainable_vars, sess)
saver = tf.train.Saver()
# tensorboard stuff 
writer = tf.summary.FileWriter('stats/', sess.graph)
episode_reward = tf.Variable(0.)
tf.summary.scalar("Reward", episode_reward)
summary = tf.summary.merge_all()
# net initialization
sess.run(tf.global_variables_initializer())
checkpoint = tf.train.get_checkpoint_state("networks")
steps_vec = []
rew_vec = []

# restore networks after a training, if possible
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")

def train():
    for episode in (range(num_episodes)):
        state = env.reset()
        steps = 0
        tot_reward = 0.0
        for step in (range(max_steps)):
            # action with noise. The noise decrease with the number of the episodes
            action = actor.predict([state]) + 1.0/(1.0 + 0.005*episode)*noise.sample()
            obs, rew, done, _ = env.step(action[0])
            # fill the buffer
            buffer_replay.append((state, action[0], rew, obs, done))
            state = obs
            # if the buffer is full, forget the oldest experience
            if len(buffer_replay) >= buffer_size:
                buffer_replay.popleft()
            # if the buffer is full enough, start the training!
            if len(buffer_replay) >= batch_size:
                # sample a random batch of experiences from the buffer
                batch = random.sample(buffer_replay, batch_size)
                b_states, b_actions, b_rews, b_next_states, b_dones = batch_selector(batch)
                # produce action and q-val from the target networks
                action_target = actor.target_predict(b_next_states)
                q_target = critic.target_predict(b_next_states, action_target)
                y = np.zeros(batch_size)
                # use the Bellman equation
                for i in range(batch_size):
                    if b_dones[i]:
                        y[i] = b_rews[i]
                    else:
                        y[i] = b_rews[i] + gamma*q_target[i]
                # critic train, minimize the TD error
                y = np.reshape(y, (batch_size,1))
                critic.train(b_states, b_actions, y)
                # actor train. Update the policy using the gradient
                pred_actions = actor.predict(b_states)
                gradient = critic.gradient(b_states, pred_actions)
                actor.train(b_states, gradient[0])
                # update softly the target networks
                actor.target_train()
                critic.target_train()
            steps += 1
            tot_reward += rew
            if done:
                break
        # collect stats
        steps_vec.append(steps)
        rew_vec.append(tot_reward)
        mean_steps = np.mean(steps_vec[-100:])
        mean_rew = np.mean(rew_vec[-100:])
        print("Train_ep: {} | Ep_reward: {} | Last_mean_reward {} ".format(episode, round(tot_reward, 2), round(mean_rew, 2)), end="\r")
        # write on the tensorboard
        summary_ = sess.run(summary, feed_dict={episode_reward: tot_reward})
        writer.add_summary(summary_, episode)
        writer.flush()
        # save networks
        if episode % 100 == 0:
            saver.save(sess, "networks/"+"ddpg")
        
def test():
    # just run the policy on the environment
    state = env.reset()
    total_rew = 0
    for episode in range(num_episodes):
        for step in range(max_steps):
            env.render()
            action = actor.predict([state])
            obs, rew, done, _ = env.step(action[0])
            total_rew += rew
            state = obs
            if done:
                state = env.reset()
                print("Episode: {} | Reward: {} \r".format(episode, round(total_rew, 3)))
                total_rew = 0
                break
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='train or test', default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    else:
        test()

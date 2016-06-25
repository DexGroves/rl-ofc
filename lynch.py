#!/usr/bin/env python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"


from skimage.transform import resize
from skimage.color import rgb2gray
import threading
import tensorflow as tf
import sys
import random
import numpy as np
import time
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model
from collections import deque, Counter
from keras import backend as K

from rlofc.ofc_environment import OFCEnv
from rlofc.gamestate_encoder import SelfRankBinaryEncoder


# RLOFC stuff
encoder = SelfRankBinaryEncoder()

# Path params
EXPERIMENT_NAME = "rlofc"
SUMMARY_SAVE_PATH = "summaries/"+EXPERIMENT_NAME
CHECKPOINT_SAVE_PATH = "/tmp/"+EXPERIMENT_NAME+".ckpt"
CHECKPOINT_NAME = "/tmp/" + EXPERIMENT_NAME + ".ckpt-5"
CHECKPOINT_INTERVAL=50
SUMMARY_INTERVAL=10

TRAINING = True
SHOW_TRAINING = False

# Experiment params
ACTIONS = 3
NUM_CONCURRENT = 1
NUM_EPISODES = 200

AGENT_HISTORY_LENGTH = encoder.dim

# DQN Params
GAMMA = 0.99

# Optimization Params
LEARNING_RATE = 0.0001

# Shared global parameters
T = 0
TMAX = 80000000
t_max = 32


def build_policy_and_value_networks(num_actions, agent_history_length):
    with tf.device("/cpu:0"):
        state = tf.placeholder("float", [None,
                                         agent_history_length])

        inputs = Input(shape=[encoder.dim])

        shared = Dense(100, name="h1", activation='relu')(inputs)
        shared = Dense(100, name="h2", activation='relu')(shared)

        action_probs = Dense(name="p",
                             output_dim=num_actions,
                             activation='softmax')(shared)

        state_value = Dense(name="v", output_dim=1, activation='linear')(shared)

        policy_network = Model(input=inputs, output=action_probs)
        value_network = Model(input=inputs, output=state_value)

        p_params = policy_network.trainable_weights
        v_params = value_network.trainable_weights

        p_out = policy_network(state)
        v_out = value_network(state)

    return state, p_out, v_out, p_params, v_params


def sample_policy_action(num_actions, probs):
    """
    Sample an action from an action probability distribution output by
    the policy network.
    """
    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    probs = probs - np.finfo(np.float32).epsneg

    histogram = np.random.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index


def actor_learner_thread(num, env, session, graph_ops, summary_ops, saver):
    # We use global shared counter T, and TMAX constant
    global TMAX, T

    # Unpack graph ops
    s, a, R, minimize, p_network, v_network = graph_ops

    # Unpack tensorboard summary stuff
    r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op = summary_ops

    # Wrap env with AtariEnvironment helper class
    # env = AtariEnvironment(gym_env=env,
    #                        resized_width=RESIZED_WIDTH,
    #                        resized_height=RESIZED_HEIGHT,
    #                        agent_history_length=AGENT_HISTORY_LENGTH)

    time.sleep(5 * num)

    # Set up per-episode counters
    ep_reward = 0
    ep_rewards = []
    ep_avg_v = 0
    v_steps = 0
    ep_t = 0
    running_reward = None

    probs_summary_t = 0

    # Observe and encode the game state
    env.reset()
    observation = env.observe()
    plyr_board, oppo_board, cur_card, cards, terminal, r_t = observation
    s_t = encoder.encode(*observation)

    terminal = False

    while T < TMAX:
        s_batch = []
        past_rewards = []
        a_batch = []

        t = 0
        t_start = t

        while not (terminal or ((t - t_start)  == t_max)):
            # Perform action a_t according to policy pi(a_t | s_t)
            probs = session.run(p_network, feed_dict={s: [s_t]})[0]
            action_index = sample_policy_action(ACTIONS, probs)
            a_t = np.zeros([ACTIONS])
            a_t[action_index] = 1

            if probs_summary_t % 100 == 0 and running_reward is not None:
                # print "P, ", np.max(probs), "V ", session.run(v_network, feed_dict={s: [s_t]})[0][0], "R ", running_reward
                print str(num) + '\t' + str(running_reward) + '\t' + Counter(ep_rewards).__repr__()

            s_batch.append(s_t)
            a_batch.append(a_t)

            env.step(action_index)
            observation = env.observe()
            plyr_board, oppo_board, cur_card, cards, terminal, r_t = observation

            s_t1 = encoder.encode(*observation)

            ep_reward += r_t

            # r_t = np.clip(r_t, -1, 1)
            past_rewards.append(r_t)

            t += 1
            T += 1
            ep_t += 1
            probs_summary_t += 1

            s_t = s_t1

        if terminal:
            R_t = 0
        else:
            R_t = session.run(v_network, feed_dict={s: [s_t]})[0][0] # Bootstrap from last state

        R_batch = np.zeros(t)
        for i in reversed(range(t_start, t)):
            R_t = past_rewards[i] + GAMMA * R_t
            R_batch[i] = R_t

        session.run(minimize, feed_dict={R : R_batch,
                                         a : a_batch,
                                         s : s_batch})
        if R_t > 0:
            print R_batch
            print a_batch
            print s_batch

        # Save progress every 5000 iterations
        if T % CHECKPOINT_INTERVAL == 0:
            saver.save(session, CHECKPOINT_SAVE_PATH, global_step = T)

        if terminal:
            # Episode ended, collect stats and reset game
            # session.run(update_ep_reward, feed_dict={r_summary_placeholder: ep_reward})
            # print "THREAD:", num, "/ TIME", T, "/ REWARD", ep_reward

            env.reset()
            observation = env.observe()
            plyr_board, oppo_board, cur_card, cards, terminal, r_t = observation
            s_t = encoder.encode(*observation)

            running_reward = ep_reward if running_reward is None \
                else running_reward * 0.99 + ep_reward * 0.01

            terminal = False
            # Reset per-episode counters
            ep_rewards.append(ep_reward)
            ep_reward = 0
            ep_t = 0



def build_graph():
    # Create shared global policy and value networks
    s, p_network, v_network, p_params, v_params = \
        build_policy_and_value_networks(num_actions=ACTIONS,
                                        agent_history_length=AGENT_HISTORY_LENGTH)

    # Shared global optimizer
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    # Op for applying remote gradients
    R_t = tf.placeholder("float", [None])
    a_t = tf.placeholder("float", [None, ACTIONS])
    log_prob = tf.log(tf.reduce_sum(tf.mul(p_network, a_t), reduction_indices=1))
    p_loss = -log_prob * (R_t - v_network)
    v_loss = tf.reduce_mean(tf.square(R_t - v_network))

    total_loss = p_loss + (0.5 * v_loss)

    minimize = optimizer.minimize(total_loss)
    return s, a_t, R_t, minimize, p_network, v_network


# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Episode Reward", episode_reward)
    r_summary_placeholder = tf.placeholder("float")
    update_ep_reward = episode_reward.assign(r_summary_placeholder)
    ep_avg_v = tf.Variable(0.)
    tf.scalar_summary("Episode Value", ep_avg_v)
    val_summary_placeholder = tf.placeholder("float")
    update_ep_val = ep_avg_v.assign(val_summary_placeholder)
    summary_op = tf.merge_all_summaries()
    return r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op


def train(session, graph_ops, saver):
    # Set up game environments (one per thread)
    envs = [OFCEnv([]) for i in range(NUM_CONCURRENT)]

    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(SUMMARY_SAVE_PATH, session.graph)

    # Start NUM_CONCURRENT training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread,
                                              args=(thread_id,
                                                    envs[thread_id],
                                                    session, graph_ops,
                                                    summary_ops, saver))
                                for thread_id in range(NUM_CONCURRENT)]

    for t in actor_learner_threads:
        t.start()

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        if SHOW_TRAINING:
            for env in envs:
                env.render()
        now = time.time()
        if now - last_summary_time > SUMMARY_INTERVAL:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now
    for t in actor_learner_threads:
        t.join()


def evaluation(session, graph_ops, saver):
    saver.restore(session, CHECKPOINT_NAME)
    print "Restored model weights from ", CHECKPOINT_NAME
    monitor_env = gym.make(GAME)
    monitor_env.monitor.start('/tmp/'+EXPERIMENT_NAME+"/eval")

    # Unpack graph ops
    s, a_t, R_t, learning_rate, minimize, p_network, v_network = graph_ops

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=monitor_env, resized_width=RESIZED_WIDTH, resized_height=RESIZED_HEIGHT, agent_history_length=AGENT_HISTORY_LENGTH)

    for i_episode in xrange(100):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            # Forward the deep q network, get Q(s,a) values
            probs = p_network.eval(session = session, feed_dict = {s : [s_t]})[0]
            action_index = sample_policy_action(ACTIONS, probs)
            s_t1, r_t, terminal, info = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        print ep_reward
    monitor_env.monitor.close()


def main(_):
    g = tf.Graph()
    with g.as_default(), tf.Session() as session:
        K.set_session(session)
        graph_ops = build_graph()
        saver = tf.train.Saver()

        if TRAINING:
            train(session, graph_ops, saver)
        else:
            evaluation(session, graph_ops, saver)

if __name__ == "__main__":
    tf.app.run()

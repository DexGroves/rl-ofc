#!/usr/bin/env python

import threading
import time
import tensorflow as tf
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model

from rlofc.ofc_environment import OFCEnv
from rlofc.gamestate_encoder import SelfRankBinaryEncoder

# Experiment params
ACTIONS = 3
NUM_CONCURRENT = 8
NUM_EPISODES = 200

LEARNING_RATE = 3e-5
t_max = 32
GAMMA = 0.95

# Path params
EXPERIMENT_NAME = "rlofc"
SUMMARY_SAVE_PATH = "summaries/" + EXPERIMENT_NAME
CHECKPOINT_SAVE_PATH = "/tmp/" + EXPERIMENT_NAME + ".ckpt"
CHECKPOINT_NAME = "/tmp/" + EXPERIMENT_NAME + ".ckpt-5"
CHECKPOINT_INTERVAL = 50
SUMMARY_INTERVAL = 10

encoder = SelfRankBinaryEncoder()
INPUT_DIM = encoder.dim

# Shared global parameters
T = 0
TMAX = 80000000
t_max = 32


def get_networks():
    """Get policy and value networks."""
    with tf.device("/cpu:0"):
        # Placeholder for a tensor that will always be fed
        state = tf.placeholder("float", [None, INPUT_DIM])

        inputs = Input(shape=[INPUT_DIM])

        shared = Dense(100, activation="relu")(inputs)
        shared = Dense(100, activation="relu")(shared)

        action_probs = Dense(name="p",
                             output_dim=ACTIONS,
                             activation="softmax")(shared)

        state_value = Dense(name="v",
                            output_dim=1,
                            activation="linear")(shared)

        policy_model = Model(input=inputs, output=action_probs)
        value_model = Model(input=inputs, output=state_value)

        p_params = policy_model.trainable_weights
        v_params = value_model.trainable_weights

        policy_network = policy_model(state)
        value_network = value_model(state)

    return state, policy_network, value_network, p_params, v_params


def build_tf_graph():
    """Create global shared policy and value networks. Define loss function."""
    state, policy_network, value_network, p_params, v_params = \
        get_networks()

    # Shared global optimizer
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    # tf magic. "Op for applying remote gradients" (???)
    R_t = tf.placeholder("float", [None])
    a_t = tf.placeholder("float", [None, ACTIONS])

    log_prob = tf.log(tf.reduce_sum(tf.mul(policy_network, a_t),
                                    reduction_indices=1))
    p_loss = -log_prob * (R_t - value_network)
    v_loss = tf.reduce_mean(tf.square(R_t - value_network))
    total_loss = p_loss + (0.5 * v_loss)

    minimize = optimizer.minimize(total_loss)
    return state, a_t, R_t, minimize, policy_network, value_network


def build_summary_ops():
    """Tensorflow magic episode summary operations.
    I have no idea what this does or how this works."""
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


def a3c_thread(session, thread_index, tf_graph, summary_ops, env, saver):

    global TMAX, T

    # Don't all start asynchronously criticising at once...
    time.sleep(2 * thread_index)

    # Unpack input objects
    s, a, R, minimize, policy_network, value_network = tf_graph

    r_summary_placeholder, update_ep_reward, val_summary_placeholder, \
        update_ep_val, summary_op = summary_ops

    running_reward = None

    # Observe and encode game state
    env.reset()
    observation = env.observe()
    _, _, _, _, terminal, r_t = observation
    s_t = env.encoder.encode(*observation)

    elapsed_games = 0

    while T < TMAX:
        # Per-batch counters
        s_batch = []
        a_batch = []

        t = 0
        t_start = 0

        while not terminal:
            # Forward the policy network and sample an action
            probs = session.run(policy_network, feed_dict={s: [s_t]})[0]
            action_idx = np.random.choice(ACTIONS, 1, p=probs)
            a_t = np.zeros([ACTIONS])
            a_t[action_idx] = 1

            # Append state and action to batch
            s_batch.append(s_t)
            a_batch.append(a_t)

            # Take the action and observe
            env.step(action_idx)
            observation = env.observe()
            _, _, _, _, terminal, r_t1 = observation
            s_t1 = env.encoder.encode(*observation)

            # Increment everything
            t += 1
            T += 1
            s_t = s_t1

        R_t = r_t1
        R_d = discount_rewards(R_t, (t - t_start))

        running_reward = R_t if running_reward is None \
            else running_reward * 0.99 + R_t * 0.01

        elapsed_games += 1

        if elapsed_games % 100 == 0:
            print "P, ", np.max(probs), "V ", session.run(value_network, feed_dict={s: [s_t]})[0][0], "R ", running_reward
        # Minimize globally!
        session.run(minimize, feed_dict={R: R_d,
                                         a: a_batch,
                                         s: s_batch})

        # Tensorboard stuff
        session.run(update_ep_reward, feed_dict={r_summary_placeholder: R_t})

        # Reset and reobserve!
        env.reset()
        observation = env.observe()
        _, _, _, _, terminal, r_t = observation
        s_t = env.encoder.encode(*observation)

        if T % CHECKPOINT_INTERVAL == 0:
            saver.save(session, CHECKPOINT_SAVE_PATH, global_step=T)


def discount_rewards(R, t):
    """Decay rewards back in time."""
    R_d = np.zeros(t)
    for i in reversed(range(t)):
        R_d[i] = R * (GAMMA ** (i - 1))
    return R_d


def train(session, tf_graph, saver):
    """Set up threaded environments."""
    envs = [OFCEnv([], SelfRankBinaryEncoder) for i in range(NUM_CONCURRENT)]

    summary_ops = build_summary_ops()
    summary_op = summary_ops[-1]

    session.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(SUMMARY_SAVE_PATH, session.graph)

    a3c_threads = [threading.Thread(target=a3c_thread,
                                    args=(session,
                                          thread_id,
                                          tf_graph,
                                          summary_ops,
                                          envs[thread_id],
                                          saver))
                    for thread_id in range(NUM_CONCURRENT)]

    for t in a3c_threads:
        t.start()

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        now = time.time()
        if now - last_summary_time > SUMMARY_INTERVAL:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now

    for t in a3c_threads:
        t.join()


def main(_):
    g = tf.Graph()
    with g.as_default(), tf.Session() as session:
        K.set_session(session)
        graph_ops = build_tf_graph()
        saver = tf.train.Saver()

        train(session, graph_ops, saver)


if __name__ == "__main__":
    tf.app.run()

from time import time

import numpy as np
import torch
from torch import save as torch_save
import torch.nn.functional as torch_func
import torch.optim as torch_optim
from pickle import dump as pkl_dump, load as pkl_load
from dotmap import DotMap

from python.unityagents import UnityEnvironment

from p1_navigation.DQN import DQN
from p1_navigation.EpsilonGreedyExploration import EpsilonGreedyExplorationStrategy
from p1_navigation.plots import plot_rewards
from p1_navigation.QNetwork import QNetwork

### Experiment Setup

TRAIN_AGENT = False
TEST_AGENT = True
TEST_MODE = False

RESULTS_CONFIG = DotMap({
    'SAVE_REWARDS_DATA': True,
    'SAVE_REWARDS_PLOT': True,
    'SAVE_MODEL': True,
})

## Environment Config

RENDER_ENV = False
SEED = 51
ENV_PLATFORM = 'unity'
ENV_NAME = 'Banana'

## Training/Testing Config

NUM_TRAIN_EPISODES = 1000
NUM_TEST_EPISODES = 1000
MAX_NUM_STEPS = 10000
TRAIN_MODEL = True
PROGRESS_LOG_STEP_FREQUENCY = 1

MIN_PASSING_ACC_REWARD = 13.0
MIN_PASSING_NUM_EPISODES = 100

### Learning Algorithm's Hyper-params Config

ALPHA = 0.0001           # learning rate (used by optimizer)
GAMMA = 0.99             # discount rate
EPSILON_START = 0.5
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.98
EXPLORATION_STRATEGY_FN = EpsilonGreedyExplorationStrategy

## Experience Replay

BUFFER_SIZE = int(1e5)   # replay buffer size
BATCH_SIZE = 128         # minibatch size
MIN_NUM_BATCHES = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Neural Network

HIDDEN_LAYERS_UNITS = (64, 64)
ACTIVATION_FNS = (torch_func.relu, torch_func.relu)
NETWORK_UPDATE_INTERVAL = 2     # time-step frequency for conducting Q-Network updates
TAU = 1e-3                      # time for soft update of target parameters
LOSS_FN = torch_func.mse_loss
OPTIMIZER_FN = torch_optim.Adam

## Agents

if ENV_PLATFORM == 'unity':
    env = UnityEnvironment(file_name=ENV_NAME + '.app', no_graphics=not RENDER_ENV)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment state

## MDP

state_space = env_info.vector_observations[0]
state_size = len(state_space)
action_size = brain.vector_action_space_size


### Run Experiments

agent = DQN(state_size=state_size,
            action_size=action_size,
            q_network_hidden_layers_dims=HIDDEN_LAYERS_UNITS,
            q_network_activation_fns=ACTIVATION_FNS,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon_start = EPSILON_START,
            epsilon_min = EPSILON_MIN,
            epsilon_decay=EPSILON_DECAY,
            tau=TAU,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            min_num_batches=MIN_NUM_BATCHES,
            optimizer_fn=OPTIMIZER_FN,
            loss_fn=LOSS_FN,
            q_model_update_step_interval=NETWORK_UPDATE_INTERVAL,
            exploration_strategy_fn=EXPLORATION_STRATEGY_FN,
            device=DEVICE,
            seed=SEED)


def train_agent(env_name: str,
                env_platform: str,
                num_episodes: int,
                max_num_steps: int,
                min_passing_acc_reward=MIN_PASSING_ACC_REWARD,
                min_passing_num_episodes=MIN_PASSING_NUM_EPISODES,
                progress_log_step_frequency=PROGRESS_LOG_STEP_FREQUENCY,
                render=False):

    epoch_rewards = []  # list containing scores from each episode_idx
    first_time_solved = False

    start_time = time()

    for episode_idx in range(num_episodes):
        episode_acc_reward = 0
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]  # reset the environment

        for step in range(max_num_steps):
            action = agent.choose_action(state=state)

            step_feedback_info = env.step(vector_action=action)[brain_name]

            next_state = step_feedback_info.vector_observations[0]
            reward = step_feedback_info.rewards[0]
            done = step_feedback_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            state = next_state

            episode_acc_reward += reward

            if done:
                break

        epoch_rewards.append(episode_acc_reward)  # save most recent score
        test_window_mean_acc_reward = np.mean(epoch_rewards[-min_passing_num_episodes:])

        if progress_log_step_frequency and episode_idx % progress_log_step_frequency == 0:
            print('\rEpisode {}\tMean Acc. Reward: {:.2f}\tEps: {:.3f}'.format(episode_idx,
                                                                               test_window_mean_acc_reward,
                                                                               agent.epsilon))

        if not first_time_solved and test_window_mean_acc_reward >= min_passing_acc_reward:
            first_time_solved = True
            time_to_solve = '{:.3f}'.format(time() - start_time)
            env_solved_log_msg = '\nEnvironment solved in {} episodes!\tAverage Score: {:.2f}'
            print(env_solved_log_msg.format(episode_idx - test_window_mean_acc_reward,
                                            test_window_mean_acc_reward))

    stats = DotMap({
        'time_to_solve': time_to_solve,
        'epoch_train_time': '{:.3f}'.format(time() - start_time)
    })
    return epoch_rewards, stats


if TRAIN_AGENT:
    acc_rewards, stats = train_agent(env_name='Banana',
                                     num_episodes=NUM_TRAIN_EPISODES,
                                     env_platform='unity',
                                     max_num_steps=MAX_NUM_STEPS)

    experiment_filename = '{epoch_train_time}-a_{alpha}-g_{gamma}-e_{epsilon}-edecay_{epsilon_decay}-emin_{epsilon_min}'\
        .format(epoch_train_time=stats.epoch_train_time,
                alpha=ALPHA,
                gamma=GAMMA,
                epsilon=EPSILON_START, epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY)

    print("\n\nScore: {}".format(acc_rewards))

    if RESULTS_CONFIG.SAVE_MODEL:
        torch_save(agent.online_q_network.state_dict(), experiment_filename + 'pth')

    if RESULTS_CONFIG.SAVE_REWARDS_DATA:
        pkl_dump(acc_rewards, open('./results/' + experiment_filename + ".p", 'wb'))

if TEST_AGENT:
    model = torch.load('./models/checkpoint' + '.pth')
    model = agent.target_q_network
    model.eval()

    test_epoch_rewards = []  # list containing scores from each episode_idx
    first_time_solved = False

    test_start_time = time()

    for episode_idx in range(NUM_TEST_EPISODES):
        episode_acc_reward = 0
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]  # reset the environment

        for step in range(MAX_NUM_STEPS):

            state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action_values = model(state)
                q_values = action_values.cpu().data.numpy().squeeze()

            action = np.argmax(q_values)

            step_feedback_info = env.step(vector_action=action)[brain_name]

            next_state = step_feedback_info.vector_observations[0]
            reward = step_feedback_info.rewards[0]
            done = step_feedback_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            state = next_state

            episode_acc_reward += reward

            if done:
                break

        test_epoch_rewards.append(episode_acc_reward)  # save most recent score

        if PROGRESS_LOG_STEP_FREQUENCY and episode_idx % PROGRESS_LOG_STEP_FREQUENCY == 0:
            print('\rEpisode {}\tAcc. Reward: {:.2f}\tEps: {:.3f}'.format(episode_idx,
                                                                               reward,
                                                                               agent.epsilon))

    test_stats = DotMap({
        'test_epoch_time': '{:.3f}'.format(time() - test_start_time)
    })

    results_savetofilename = 'results/test'
    plot_rewards(saveto_filename=results_savetofilename, data=test_epoch_rewards, ylim=(-5, 25), dpi=320)



if RESULTS_CONFIG.SAVE_REWARDS_PLOT:
    results_savetofilename = 'acc_rewards_01'
    acc_rewards = pkl_load(open('./results/' + results_savetofilename + '.p', 'rb'))

    plot_rewards(saveto_filename=results_savetofilename, data=acc_rewards, ylim=(-5, 25), dpi=320)

env.close()

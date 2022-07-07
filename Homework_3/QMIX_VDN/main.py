import torch
import numpy as np
import argparse
from qmix import QMIX
from replay_buffer import ReplayBuffer
from normalization import Normalization
from make_env import make_env
from tqdm import tqdm
import wandb

class Runner:
    def __init__(self, args):
        self.args = args
        self.env_name = args.env_name
        self.seed = args.seed

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create env
        self.env = make_env(self.env_name)
        # And set params about the env

        # The number of agents = 3
        self.args.N = self.env.n
        # The dimensions of an agent's observation space = 18
        self.args.obs_dim = self.env.observation_space[0].shape[0]
        # Note: We concatenate all agents' local observation as the global state
        # as there is no state in MPE's simple_spread scenario.
        # The dimensions of global state space = 3 * 18
        self.args.state_dim = int(self.args.N) * int(self.args.obs_dim)
        # The dimensions of an agent's action space = 5
        self.args.action_dim = self.env.action_space[0].n
        # Maximum number of steps per episode
        self.args.episode_limit = self.args.episode_limit

        # Log
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        # Note: instantiate ONLY ONE agent here according to the weight sharing principle.
        self.agent_n = QMIX(self.args)

        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        # self.writer = SummaryWriter(log_dir='./runs/{}/{}_env_{}_number_{}_seed_{}'.format(
        #     self.args.algorithm, self.args.algorithm, self.env_name, self.number, self.seed))

        self.epsilon = self.args.epsilon  # Initialize the epsilon

        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

    def train(
        self,
    ):
        for _episode in tqdm(range(self.args.max_episode)):
            # Run an episode
            episode_reward, episode_steps = self.run_an_episode(evaluate=False)
            
            if self.args.wandb_log:
                wandb.log({"Reward": episode_reward}, step=_episode+1)
            else:
                print(f"Episode: {_episode}, Reward: {episode_reward}")

            if self.replay_buffer.current_size >= self.args.batch_size:
                # Training
                # for _ in range(25):
                self.agent_n.train(self.replay_buffer)

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(
        self,
    ):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_an_episode(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        print(
            "total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(
                self.total_steps, win_rate, evaluate_reward
            )
        )
        # self.writer.add_scalar('win_rate_{}'.format(
        #     self.env_name), win_rate, global_step=self.total_steps)
        # Save the win rates
        np.save(
            './data_train/{}_env_{}_number_{}_seed_{}.npy'.format(
                self.args.algorithm, self.env_name, self.number, self.seed
            ),
            np.array(self.win_rates),
        )

    def run_an_episode(self, evaluate=False):
        # win_tag = False
        episode_reward = 0
        obs_n = self.env.reset()  # (3, 18)

        # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
        if self.args.use_rnn:
            self.agent_n.eval_Q_net.rnn_hidden = None

        # Last actions of N agents(one-hot)
        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))

        # 25 steps each episode
        for episode_step in range(self.args.episode_limit):
            # obs_n = self.env.get_obs()  # obs_n.shape=(N, obs_dim)
            # s = self.env.get_state()  # s.shape=(state_dim,)
            s = np.reshape(obs_n, self.args.state_dim)
            # avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            # Note: Not one-hot
            a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, epsilon)
            # One-hot actions
            a_n_one_hot = np.eye(self.args.action_dim)[a_n]
            # Convert actions to one-hot vectors

            obs_next, r, done, info = self.env.step(last_onehot_a_n)
            r = r[0]

            # r, done, info = self.env.step(a_n)
            # win_tag = True if done and 'battle_won' in info and info['battle_won'] else False

            # self.replay_buffer.store_transition(episode_step, obs_n, s, last_onehot_a_n, a_n, r[0])

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                """"
                    When done or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if all(done) and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(
                    episode_step, obs_n, s, last_onehot_a_n, a_n, r, dw
                )
                # Decay the epsilon
                self.epsilon = (
                    self.epsilon - self.args.epsilon_decay
                    if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min
                    else self.args.epsilon_min
                )

            last_onehot_a_n = a_n_one_hot
            episode_reward += r
            obs_n = obs_next

            if all(done):
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            s = np.reshape(obs_n, self.args.state_dim)
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s)

        return episode_reward / (episode_step + 1), episode_step + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Hyperparameter setting for QMIX and VDN in MPE environment's Simple Spread scenario."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=int(25*50000),
        help=" Maximum number of training steps",
    )
    parser.add_argument(
        "--max_episode",
        type=int,
        default=int(50000),
        help=" Maximum number of episodes.",
    )
    parser.add_argument(
        "--evaluate_freq",
        type=float,
        default=5000,
        help="Evaluate the policy every 'evaluate_freq' steps",
    )
    parser.add_argument(
        "--evaluate_times", type=float, default=32, help="Evaluate times"
    )
    parser.add_argument(
        "--save_freq", type=int, default=1000, help="Save frequency"
    )
    parser.add_argument(
        "--episode_limit", type=int, default=25, help="Maximum episode length."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="simple_spread",
        help="Scenario name in MPE environment.",
    )
    parser.add_argument("--algorithm", type=str, default="VDN", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument(
        "--epsilon_decay_steps",
        type=float,
        default=50000,
        help="How many steps before the epsilon decays to the minimum",
    )
    parser.add_argument(
        "--epsilon_min", type=float, default=0.05, help="Minimum epsilon"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=int(1e6),
        help="The capacity of the replay buffer",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size (the number of episodes)"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--qmix_hidden_dim",
        type=int,
        default=32,
        help="The dimension of the hidden layer of the QMIX network",
    )
    parser.add_argument(
        "--hyper_hidden_dim",
        type=int,
        default=64,
        help="The dimension of the hidden layer of the hypernetwork",
    )
    parser.add_argument(
        "--hyper_layers_num",
        type=int,
        default=1,
        help="The number of layers of hyper-network",
    )
    parser.add_argument(
        "--rnn_hidden_dim",
        type=int,
        default=64,
        help="The dimension of the hidden layer of RNN",
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=64,
        help="The dimension of the hidden layer of MLP",
    )
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument(
        "--use_orthogonal_init",
        type=bool,
        default=True,
        help="Orthogonal initialization",
    )
    parser.add_argument(
        "--use_grad_clip", type=bool, default=True, help="Gradient clip"
    )
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument(
        "--use_RMS",
        type=bool,
        default=False,
        help="Whether to use RMS,if False, we will use Adam",
    )
    parser.add_argument(
        "--add_last_action",
        type=bool,
        default=True,
        help="Whether to add last actions into the observation",
    )
    parser.add_argument(
        "--add_agent_id",
        type=bool,
        default=True,
        help="Whether to add agent id into the observation",
    )
    parser.add_argument(
        "--use_double_q",
        type=bool,
        default=True,
        help="Whether to use double q-learning",
    )
    parser.add_argument(
        "--use_reward_norm",
        type=bool,
        default=False,
        help="Whether to use reward normalization",
    )
    parser.add_argument(
        "--use_hard_update", type=bool, default=False, help="Whether to use hard update"
    )
    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=200,
        help="Update frequency of the target network",
    )
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")

    # wandb logging
    parser.add_argument('--wandb_log', default=True, type=bool, help='Whether use wandb logging toolkit.')
    parser.add_argument('--project_name', default="QMIX", type=str, help="Project name for wandb logging.")
    parser.add_argument('--name', default="QMIX1", type=str, help="Run name for wandb logging.")



    args = parser.parse_args()
    args.save_id = np.random.randint(10000)
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    wandb.init(config=args, project="QMIX")
    args = wandb.config

    runner = Runner(args)
    runner.train()

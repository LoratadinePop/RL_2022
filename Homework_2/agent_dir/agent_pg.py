import numpy as np
from torch.distributions import Categorical
import torch
from torch import nn, optim
from agent_dir.agent import Agent
import torch.nn.functional as F
import wandb
class PGNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PGNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        ##################

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        x = torch.Tensor(inputs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        prob = F.softmax(x, dim=-1)
        return prob
        ##################


class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Trajectory buffer. It will clear the buffer after updating.
        """
        ##################
        # YOUR CODE HERE #
        self.buffer_size = buffer_size
        self.buffer = []
        ##################

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        return len(self.buffer)
        ##################

    def push(self, *transition):
        ##################
        # YOUR CODE HERE #
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)
        ##################

    def sample(self):
        """
        Sample all the data stored in the buffer
        """
        ##################
        # YOUR CODE HERE #
        return zip(*self.buffer)
        ##################

    def clean(self):
        ##################
        # YOUR CODE HERE #
        self.buffer.clear()
        ##################


class AgentPG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentPG, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        self.env = env
        self.args = args
        self.obs_dim = env.observation_space.shape[0] #4
        self.action_dim = env.action_space.n #

        self.pg_net = PGNetwork(self.obs_dim, self.args.hidden_size, self.action_dim)
        self.optimizer = optim.Adam(self.pg_net.parameters(), lr=self.args.lr)
        self.buffer = ReplayBuffer(buffer_size=self.args.buffer_size)

        ##################

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ckpt_path = self.args.model_dir
        self.pg_net = torch.load(ckpt_path)
        # print(f"Load model checkpoint from {ckpt_path}")
        ##################

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        log_probs, rewards = self.buffer.sample()
        log_probs = torch.stack(log_probs)
        trans_len = len(rewards)
        rets = np.empty(trans_len, dtype=np.float32)
        future_rets = 0.0
        for t in reversed(range(trans_len)):
            future_rets = rewards[t] + self.args.gamma * future_rets
            rets[t] = future_rets
        rets = torch.Tensor(rets)
        loss = -rets * log_probs
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()
        ##################

    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################
        # YOUR CODE HERE #
        prob = self.pg_net(observation)

        if test:
            action = prob.argmax(dim=-1)
            return int(action)

        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action), log_prob
        ##################

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        for episode_ in range(self.args.episode):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, log_prob = self.make_action(observation=obs, test=False)
                obs_, reward, done, info = self.env.step(action)
                self.buffer.push(log_prob, reward)
                obs = obs_
                episode_reward += reward

            self.train()
            self.buffer.clean()

            if self.args.wandb_log:
                wandb.log({
                    'Reward': episode_reward,
                }, step=episode_)

            if episode_ % 100 == 0:
                print(f"Episode: {episode_}, Reward: {episode_reward}")
                torch.save(self.pg_net, self.args.model_dir)
                print(f"Model has been saved to {self.args.model_dir}")
        ##################
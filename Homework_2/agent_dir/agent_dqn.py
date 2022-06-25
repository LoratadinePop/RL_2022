import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from wandb import agent
import wandb
from agent_dir.agent import Agent
import torch.nn.functional as F
import time
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device=None, double=False, dueling=False):
        super(QNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        self.device = device
        # Set the type of DQN
        self.double = double
        self.dueling = dueling

        # obs_shape # (4, 84, 84)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4) # 32, 20, 20
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2) # 64, 9, 9
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1) # 64, 7, 7

        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)

        if self.dueling:
            # Dueling DQN
            self.fc_value = nn.Linear(hidden_size, hidden_size)
            self.fc_adv = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, 1)
            self.adv = nn.Linear(hidden_size, output_size)
        else:
            # Vanilla DQN
            self.fc2 = nn.Linear(hidden_size, output_size)
        ##################

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        x = torch.Tensor(inputs).to(self.device)
        
        # expand a single observation to the batch size 1 form
        if len(x.shape) != 4:
            x = x.unsqueeze(0)
    
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))

        if self.dueling:
            # Dueling DQN
            value = F.relu(self.fc_value(x))
            adv = F.relu(self.fc_adv(x))

            value = self.value(value)
            adv = self.adv(adv)

            avg_adv = torch.mean(adv, dim=1, keepdim=True)
            x = value + adv - avg_adv
        else:
            # Vanilla DQN
            x = self.fc2(x)

        return x
        ##################


class ReplayBuffer:
    def __init__(self, buffer_size):
        ##################
        # YOUR CODE HERE #
        self.buffer = []
        self.buffer_size = buffer_size
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

    def sample(self, batch_size):
        ##################
        # YOUR CODE HERE #
        idxs = np.random.choice(len(self.buffer), batch_size)
        batch_sample = [self.buffer[i] for i in idxs]
        return zip(*batch_sample)
        ##################

    def clean(self):
        ##################
        # YOUR CODE HERE #
        self.buffer.clear()
        ##################


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        self.args = args
        self.env = env
        self.env.seed(self.args.seed)
        self.device = torch.device(f"cuda:{self.args.gpu}" if self.args.use_cuda else "cpu")

        self.obs_dim = env.observation_space.shape # 4,84,84
        self.action_dim = env.action_space.n # 6

        self.eval_net = QNetwork(self.obs_dim, self.args.hidden_size, self.action_dim, device=self.device, double=self.args.double_dqn, dueling=self.args.dueling_dqn)
        self.target_net = QNetwork(self.obs_dim, self.args.hidden_size, self.action_dim, device=self.device, double=self.args.double_dqn, dueling=self.args.dueling_dqn)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.eps = args.eps # Todo: add to parse
        self.buffer = ReplayBuffer(args.buffer_size)
        self.loss = nn.MSELoss()
        self.training_step = 0
        ##################
        
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ckpt_path = self.args.model_dir
        self.eval_net = torch.load(ckpt_path)
        ##################

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        self.eval_net.to(self.device)
        self.target_net.to(self.device)

        if self.eps > self.args.eps_min:
            self.eps *= self.args.eps_decay

        if self.training_step % self.args.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.training_step += 1

        obs, actions, rewards, obs_, dones = self.buffer.sample(self.args.batch_size)
        actions = torch.LongTensor(actions).to(self.device)
        dones = torch.IntTensor(dones).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        q_eval = self.eval_net(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_next = self.target_net(obs_).detach()

        if self.args.double_dqn:
            # Double DQN
            q_next_eval = self.eval_net(obs_).detach()
            _, max_index = q_next_eval.max(1)
            q_target = rewards + self.args.gamma * (1 - dones) * q_next.gather(-1, max_index.unsqueeze(-1)).squeeze(-1)
        else:
            # vanilla DQN
            q_target = rewards + self.args.gamma * (1 - dones) * torch.max(q_next, dim=-1)[0]

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ##################

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        # YOUR CODE HERE #
        # 训练阶段采用epsilon-greedy，以epsilon概率随机选取一个action
        if np.random.uniform() <= self.eps and not test:
            action = np.random.randint(0, self.action_dim)
        else:
            action_value = self.eval_net(observation)
            action = torch.max(action_value, dim=-1)[1].cpu().numpy()
        return int(action)
        ##################

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        print(f"Training environment: {self.device}")

        for episode_ in range(self.args.episode):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.make_action(obs, test=False)
                obs_, reward, done, info = self.env.step(action)
                self.buffer.push(obs, action, reward, obs_, done)
                episode_reward += reward
                obs = obs_
                if len(self.buffer) >= self.args.buffer_size and episode_ % self.args.learning_freq == 0:
                    # start_time = time.time()
                    self.train()
                    # end_time = time.time()
                    # print(f"Episode {episode_} is training, current training step is {self.training_step}, consuming {end_time - start_time} seconds.")
            if self.args.wandb_log:
                wandb.log({
                    'Reward': episode_reward,
                }, step=episode_)
            print(f"Episode: {episode_}, Reward: {episode_reward}")

            torch.save(self.eval_net, self.args.model_dir)
            print(f"Model has been saved to {self.args.model_dir}")
        ##################

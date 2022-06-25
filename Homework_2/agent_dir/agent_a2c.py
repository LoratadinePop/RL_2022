from re import L
from turtle import forward
from matplotlib.pyplot import cla
import numpy as np
from torch.distributions import Categorical
import torch
from torch import nn, optim
from agent_dir.agent import Agent
import torch.nn.functional as F
import wandb
from collections import namedtuple

Transition = namedtuple(
    "Transition",
    (
        "obs", "action", "log_pi", "reward", "next_obs", "done"
    )
)


class A2C(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(A2C, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.actor = nn.Linear(hidden_size, output_size)
        self.critic = nn.Linear(hidden_size, 1)

    def actor_forward(self, inputs):
        out = torch.Tensor(inputs)
        out = self.fc(out)
        out = self.relu(out)
        prob = F.softmax(self.actor(out), dim=-1)
        log_prob = torch.log(prob)
        return prob, log_prob

    def critic_forward(self, inputs):
        out = torch.Tensor(inputs)
        out = self.fc(out)
        out = self.relu(out)
        out = self.critic(out)
        return out


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, *transition):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*transition)
        self.position = (self.position + 1) % self.buffer_size

    def clean(self):
        self.position = 0
        self.buffer.clear()


class AgentA2C(Agent):
    def __init__(self, env, args):
        super(AgentA2C, self).__init__(env)

        self.args = args
        self.env = env
        self.obs_dim = env.observation_space.shape[0]  # 4
        self.action_dim = env.action_space.n

        self.buffer = ReplayBuffer(self.args.buffer_size)
        self.a2c = A2C(self.obs_dim, self.args.hidden_size, self.action_dim)
        self.optimizer = optim.Adam(self.a2c.parameters(), lr=self.args.lr)

    def init_game_setting(self):
        ckpt_path = self.args.model_dir
        self.a2c = torch.load(ckpt_path)

    def train(self):
        sample = self.buffer.buffer
        batch = Transition(*zip(*sample))
        obs_batch = torch.Tensor(batch.obs)  # (bs, 4)
        action_batch = torch.Tensor(batch.action)  # (bs)
        log_pi_batch = batch.log_pi  # (bs)
        reward_batch = torch.Tensor(batch.reward)  # (bs)
        next_obs_batch = torch.Tensor(batch.next_obs)  # (bs, 4)
        done_batch = torch.Tensor(batch.done)  # (bs)


        if self.args.n_step is None:
            # PG - baseline
            trans_len = len(reward_batch)
            rets = np.empty(trans_len, dtype=np.float32)
            future_rets = 0.0
            loss = 0
            for t in reversed(range(trans_len)):
                future_rets = reward_batch[t] + self.args.gamma * future_rets
                rets[t] = future_rets - self.a2c.critic_forward(obs_batch[t])
                loss += -rets[t]*log_pi_batch[t] 
        else:
            # n-step bootstrapping
            obs_v = self.a2c.critic_forward(obs_batch[0])  # v(s)
            next_ret = self.a2c.critic_forward(next_obs_batch[-1])

            rets = torch.zeros_like(reward_batch)
            for t in reversed(range(self.args.n_step)):
                next_ret = reward_batch[t] + self.args.gamma * next_ret.item() * (1 - done_batch[t])

            q = next_ret
            advantage = q - obs_v.detach() # groud_truth - prediction
            loss = -advantage * log_pi_batch[0] + (q - obs_v) ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_action(self, observation, test=False):
        prob, log_prob = self.a2c.actor_forward(observation)

        if test:
            action = prob.argmax(dim=-1)
            return int(action)

        action = int(prob.multinomial(1))
        return action, log_prob

    def run(self):
        for episode_ in range(self.args.episode):
            obs = self.env.reset()
            done = False
            step = 0
            episode_reward = 0
            while not done:
                step += 1
                action, log_prob = self.make_action(obs, test=False)
                next_obs, reward, done, info = self.env.step(action)
                log_pi = log_prob[action]

                if done:
                    reward = -10.0

                self.buffer.push(obs, action, log_pi, reward, next_obs, done)

                obs = next_obs
                episode_reward += reward

                if self.args.n_step is not None and step == self.args.n_step:
                    # TD method, update each step
                    step = 0
                    self.train()
                    self.buffer.clean()

            if self.args.n_step is None:
                # MC method, update each episode
                self.train()
                self.buffer.clean()

            if self.args.wandb_log:
                wandb.log({
                    'Reward': episode_reward,
                }, step=episode_)
            
            if (episode_+1) % 100 == 0:
                print(f"Episode: {episode_+1}, Reward: {episode_reward}")
                torch.save(self.a2c, self.args.model_dir)
                print(f"Model has been saved to {self.args.model_dir}")
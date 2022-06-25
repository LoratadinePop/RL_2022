import torch
import os
from tqdm import tqdm
from maddpg.maddpg import MADDPG
from utils.buffer import Buffer
import wandb


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.maddpg = MADDPG(args)
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train(self):
        total_steps = 0

        for _episode in tqdm(range(self.args.episode)):
            s = self.env.reset()

            # total reward of each episode
            episode_reward = 0

            # Maximum step of each episode must be 25
            for _step in range(self.args.max_episode_len):

                total_steps += 1

                with torch.no_grad():
                    # One-hot
                    action = self.maddpg.agent.step(s[:self.args.n_agents], explore=True)

                s_next, r, done, info = self.env.step(action)

                episode_reward += sum(r) / self.args.n_agents

                self.buffer.store_episode(
                    s[:self.args.n_agents], action, r[:self.args.n_agents], s_next[:self.args.n_agents])

                s = s_next

                if self.buffer.current_size >= self.args.batch_size and total_steps % self.args.train_interval == 0:
                    transitions = self.buffer.sample(self.args.batch_size)
                    self.maddpg.learn(transitions)

                if total_steps % self.args.save_interval == 0:
                    self.maddpg.agent.save_model(total_steps)

            # Logging training reward
            if self.args.wandb_log:
                wandb.log({"Reward": episode_reward /
                          self.args.max_episode_len}, step=_episode)
            else:
                print(episode_reward / self.args.max_episode_len)

    def evaluate(self):
        """Evaluate the policies learned

        Return:
            avg_reward: Average rewards of each agent during the 25-step interaction (* args.evaluate_episodes) with the environment.
        """
        rewards = []

        for _episode in range(self.args.evaluate_episodes):

            # reset the environment
            s = self.env.reset()
            episode_reward = 0
            # 25 steps
            for _step in range(self.args.evaluate_episode_len):

                # If you want to visualize the game, please uncomment below line of code
                # self.env.render()
                actions = []

                with torch.no_grad():
                    for agent_id in range(self.args.n_agents):
                        # One-hot
                        action = self.maddpg.agent.step(
                            s[agent_id], explore=False)
                        actions.append(action)

                s_next, r, done, info = self.env.step(actions)

                episode_reward += sum(r) / self.args.n_agents
                s = s_next

            rewards.append(episode_reward / self.args.evaluate_episode_len)
        return sum(rewards) / self.args.evaluate_episodes

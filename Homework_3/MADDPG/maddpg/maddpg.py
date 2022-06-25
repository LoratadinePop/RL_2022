import torch
from utils.agent import DDPGAgent
import copy

class MADDPG:
    """Wrapper for DDPG agents in MARL
    """
    def __init__(self, args):
        self.args = args
        self.agent_num = self.args.n_agents
        # Note: weight sharing, instantiate ONLY ONE agent
        self.agent = DDPGAgent(args, agent_id=0)
    
    def learn(self, transitions):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        
        obs, actions, obs_next, action_next = [], [], [], []

        for _agent_id in range(self.agent_num):
            obs.append(transitions['o_%d' % _agent_id])
            # One-hot
            actions.append(transitions['u_%d' % _agent_id])
            obs_next.append(transitions['o_next_%d' % _agent_id])
            # one-hot
            action_next.append(self.agent.step(obs_next[_agent_id], explore=True))
        
        # critic_loss_all = []
        # actor_loss_all = []

        for _agent_id in range(self.agent_num):
            # only local reward of agent itself is needed when training
            reward = transitions['r_%d' % _agent_id]

            with torch.no_grad():
                q_next = self.agent.critic_target_network(obs_next, action_next).detach()
                q_target = (reward.unsqueeze(1) + self.args.gamma * q_next).detach()
            
            q_value = self.agent.critic_network(obs, actions)
            critic_loss = (q_target - q_value).pow(2).mean()
            # critic_loss_all.append(critic_loss)

            self.agent.critic_optim.zero_grad()
            critic_loss.backward()
            self.agent.critic_optim.step()

            new_action = self.agent.step(obs[_agent_id], explore=True)
            actions[_agent_id] = new_action

            actor_loss = - self.agent.critic_network(obs, actions).mean()

            # actor_loss_all.append(actor_loss)
        
        # actor_loss = torch.tensor(actor_loss_all, requires_grad=True).mean()
        # critic_loss = torch.tensor(critic_loss_all, requires_grad=True).mean()
        
            self.agent.actor_optim.zero_grad()
            actor_loss.backward()
            self.agent.actor_optim.step()

            # 防止
            actions[_agent_id] = new_action.detach()


            self.agent._soft_update_target_network()

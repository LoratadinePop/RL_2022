import torch
import os
from utils.util import gumbel_softmax, onehot_from_logits
from utils.actor_critic import Actor, Critic

class DDPGAgent:
    def __init__(self, args, agent_id):  
        self.args = args
        self.agent_id = agent_id

        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)

        # Target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)

        # Replicate the weights from eval net
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # Setup optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # Prepare directories for storing the model checkpoints
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # Load the weights from pretrianed model
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))


    def step(self, obs, explore=False):
        """Take a step in the environment for a batch of observations

        Args:
            explore: whether or not to add exploration noise
        """
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = self.actor_network(obs)
        if explore:
            actions = gumbel_softmax(actions, hard=True)
        else:
            actions = onehot_from_logits(actions)
        
        return actions

    def _soft_update_target_network(self):
        """Soft update the target network
        """
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def save_model(self, train_step):
        """Save model checkpoints
        """
        num = str(train_step // self.args.save_interval)

        # Prepare directories for storing the model checkpoints
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')



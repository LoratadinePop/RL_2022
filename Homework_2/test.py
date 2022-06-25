import argparse
import numpy as np
import gym
from wrappers import make_env
import wandb
seed = 11037


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_a2c', action='store_true', help='whether test a2c')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('---model_dir', default=None, help='Dir for storing model checkpoints.')
    parser.add_argument('--wandb_log', default=False, type=bool, help='Whether use wandb logging toolkit.')
    parser.add_argument('--project_name', default=None, type=str, help="Project name for wandb logging.")
    parser.add_argument('--name', default=None, type=str, help="Run name for wandb logging.")
    try:
        # Note: You need to choose a appropriate arg parser to process the testing.
        from argument import dqn_arguments
        parser = dqn_arguments(parser)
        # from argument import pg_arguments
        # parser = pg_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        # print(state)
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Episode: {i}, Reward: {episode_reward}")
        wandb.log({
                    'Reward': episode_reward,
                }, step=i)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    


def run(args):
    print(args)
    if args.test_pg:
        print("test_pg")
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        test(agent, env, total_episodes=100)

    if args.test_a2c:
        print("test_a2c")
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_a2c import AgentA2C
        agent = AgentA2C(env, args)
        test(agent, env, total_episodes=100)

    if args.test_dqn:
        if args.double_dqn:
            print("Test Double DQN")
        elif args.dueling_dqn:
            print("Test Dueling DQN")
        else:
            print("test_dqn")
        env_name = args.env_name
        env = make_env(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    if args.wandb_log:
        wandb.init(project=args.project_name, config=args, name=args.name)
    run(args)


"""
Testing script

1. Test REINFORCE
python test.py --test_pg  --model_dir='result/PG/PG.pt'  --wandb_log=True --project_name='TestPG' --name='reinforce'

2. Test REINFORCE -baseline
python test.py --test_a2c  --model_dir='result/PG/PG_baseline.pt'  --wandb_log=True --project_name='TestPG' --name='reinforce-baseline'

3. Test A2C
python test.py --test_a2c  --model_dir='result/PG/A2C_TD.pt'  --wandb_log=True --project_name='TestPG' --name='a2c'

4. Test DQN
python test.py --test_dqn --model_dir='result/DQN/Pong_Vanilla_DQN_1000.pt'  --wandb_log=True --project_name='TestDQN' --name='Vanilla DQN'

5. Test Double DQN
python test.py --test_dqn --double_dqn=True --model_dir='result/DQN/Pong_Double_DQN_500.pt'  --wandb_log=True --project_name='TestDQN' --name='Double DQN'

6. Test Dueling DQN
python test.py --test_dqn --dueling_dqn=True --model_dir='result/DQN/Pong_Dueling_DQN_500.pt'  --wandb_log=True --project_name='TestDQN' --name='Dueling DQN'
"""
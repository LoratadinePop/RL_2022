import argparse

import wandb
from wrappers import make_env
import gym
from argument import dqn_arguments, pg_arguments


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=False, type=bool, help='whether train DQN')
    parser.add_argument('--wandb_log', default=False, type=bool, help='Whether use wandb logging toolkit.')
    parser.add_argument('--project_name', default=None, type=str, help="Project name for wandb logging.")
    parser.add_argument('--name', default=None, type=str, help="Run name for wandb logging.")
    # Note: Be care of the choice of arg
    parser = dqn_arguments(parser)
    # parser = pg_arguments(parser)
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name
        env = gym.make(env_name)

        if args.a2c:
            print("a2c")
            # Actor Critic
            from agent_dir.agent_a2c import AgentA2C
            agent = AgentA2C(env, args)
            agent.run()
        else:
            # vanilla PG
            print("pg")
            from agent_dir.agent_pg import AgentPG
            agent = AgentPG(env, args)
            agent.run()

    if args.train_dqn:
        if args.double_dqn:
            print("Double DQN")
        elif args.dueling_dqn:
            print("Dueling DQN")
        else:
            print("Vanilla DQN")
            
        env_name = args.env_name
        env = make_env(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.run()


if __name__ == '__main__':
    args = parse()
    if args.wandb_log:
        wandb.init(project=args.project_name, config=args, name=args.name)
    run(args)

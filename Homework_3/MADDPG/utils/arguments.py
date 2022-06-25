import argparse

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    
    # Environment
    parser.add_argument("--scenario_name", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--episode", type=int, default=50000, help="number of time steps")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    # parser.add_argument("--num_adversaries", type=int, default=0, help="number of adversaries")
    
    # Core training parameters
    parser.add_argument("--lr_actor", type=float, default=1e-2, help="learning rate of actor")
    parser.add_argument("--lr_critic", type=float, default=1e-2, help="learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--train_interval", type=int, default=100, help="How often to train the model")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="./ckpt/run", help="directory in which training state and model should be saved")
    parser.add_argument("--save_interval", type=int, default=1e5, help="save model once every time this many episodes are completed")
    parser.add_argument("--model_dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate_episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate_episode_len", type=int, default=25, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate_interval", type=int, default=1000, help="how often to evaluate model")
    
    # wandb logging
    parser.add_argument('--wandb_log', default=False, type=bool, help='Whether use wandb logging toolkit.')
    parser.add_argument('--project_name', default=None, type=str, help="Project name for wandb logging.")
    parser.add_argument('--name', default=None, type=str, help="Run name for wandb logging.")

    args = parser.parse_args()

    return args
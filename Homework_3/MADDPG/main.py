from runner import Runner
from utils.arguments import get_args
from make_env import make_env
import wandb

if __name__ == '__main__':

    args = get_args()

    env = make_env(args.scenario_name)

    args.n_agents = env.n
    args.obs_shape = [env.observation_space[i].shape[0]
                      for i in range(args.n_agents)]  # 每一维代表该agent的obs维度 [18, 18, 18]
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n) # 每一维代表该agent的act维度 [5, 5, 5]   
    args.action_shape = action_shape[:args.n_agents]

    if args.wandb_log:
        wandb.init(project=args.project_name, config=args, name=args.name)

    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.train()

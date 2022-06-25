from email.policy import default


def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="PongNoFrameskip-v4", help='environment name')
    parser.add_argument('--model_dir', default="./result/", help="Directoty for saving model checkpoints.")
    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--buffer_size", default=100000, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--episode", default=1000, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    # Double DQN & Dueling DQN
    parser.add_argument("--double_dqn", default=False, type=bool)
    parser.add_argument("--dueling_dqn", default=False, type=bool)

    # eps greedy & decay
    parser.add_argument("--eps", default=1.0, type=float)
    parser.add_argument("--eps_min", default=0.02, type=float)
    parser.add_argument("--eps_decay", default=0.999, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--n_frames", default=int(400000), type=int)
    parser.add_argument("--learning_freq", default=1, type=int)
    parser.add_argument("--target_update_freq", default=1000, type=int)

    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')
    parser.add_argument('--model_dir', default="./result/", help="Directoty for saving model checkpoints.")
    parser.add_argument("--episode", default=2000, type=int)
    parser.add_argument("--hidden_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--buffer_size", default=10000, type=int)
    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--grad_norm_clip", default=10, type=float)
    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)
    
    # param. for A2C
    parser.add_argument("--a2c", default=False, type=bool)
    parser.add_argument("--n_step", default=None, type=int, help='None means MC, 1 means TD.')

    return parser

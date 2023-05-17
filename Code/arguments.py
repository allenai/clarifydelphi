import torch
import argparse
from Code.utils.constants import GAIN, BIAS


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # dataset
    parser.add_argument(
        '--output-dir', type=str, default='outputs')
    parser.add_argument(
        '--dataset-train', type=str, default='data/train_social_chem_with_prefix_t5.jsonl',
        help='JSONL file containing train prompts. Each row must contain a prompt at `row["prompt"]["text"]`.')
    parser.add_argument(
        '--dataset-val', type=str, default='data/dev_social_chem_with_prefix_t5.jsonl',
        help='JSONL file containing dev prompts. Each row must contain a prompt at `row["prompt"]["text"]`.')
    parser.add_argument(
        '--perspective-rate-limit', type=int, default=15, help='number of perspective call per second')

    parser.add_argument(
        '--init-model', type=str, default='output_t5_large_qgen/',
        help='language model used for policy.')
    parser.add_argument(
        '--ref-model', type=str, default='output_t5_large_qgen/',
        help='language model used for reference policy.')
    parser.add_argument(
        '--response-length', type=int, default=16, help='number of tokens to generate for each prompt.')
    parser.add_argument(
        '--temperature', type=float, default=0.7, help='temperature for sampling policy.')

    # ppo
    parser.add_argument(
        '--total-episodes', type=int, default=1000000, help='total number of episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument(
        '--nminibatches', type=int, default=1, help='number of ppo minibatch per batch')
    parser.add_argument(
        '--noptepochs', type=int, default=4, help='number of ppo epochs reusing rollouts')
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument(
        '--vf_coef', type=float, default=1.0, help='value loss coefficient')
    parser.add_argument(
        '--cliprange', type=float, default=.2, help='clip parameter for policy gradient')
    parser.add_argument(
        '--cliprange_value', type=float, default=.2, help='clip parameter for value function')
    parser.add_argument(
        '--gamma', type=float, default=1.0, help='discount factor for rewards')
    parser.add_argument(
        '--lam', type=float, default=0.95, help='lambda parameter for generalized advantage estimation')
    parser.add_argument(
        '--whiten_rewards', action='store_false', default=True, help='whether to normalize reward in each minibatch')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')

    # reward
    parser.add_argument(
        '--kl_coef', type=float, default=0.15, help='coefficient for KL term in reward')
    parser.add_argument(
        '--adaptive_kl', action='store_false', default=True, help='whether to use adaptive KL controller')
    parser.add_argument(
        '--target', type=float, default=6.0, help='target value in adaptive KL controller')
    parser.add_argument(
        '--horizon', type=float, default=10000, help='horizon value in adaptive KL controller')
    parser.add_argument(
        '--gain', type=float, default=GAIN, help='normalization factor for reward')
    parser.add_argument(
        '--bias', type=float, default=BIAS, help='normalization factor for reward')

    # generation
    parser.add_argument(
        '--num-samples', type=int, default=25, help='number of samples to generate for each prompt.')
    parser.add_argument(
        '--top-p', type=float, default=1.0, help='hyperparameter for nucleus sampling')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
         '--log-interval', type=int, default=100, help='step interval to print out logs')
    parser.add_argument(
         '--save-interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument('--eval-interval', type=int, default=1000, help='step interval to do evaluation')
    parser.add_argument(
        '--cuda-deterministic', action='store_false', default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args

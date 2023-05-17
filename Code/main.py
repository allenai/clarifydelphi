import os
import sys

import torch
import json
import time
import logging
import random
import argparse
import numpy as np
import itertools
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from Code.arguments import get_args
from Code.policy import Policy
from Code.value import Value
from Code.reward import Reward, reward_to_toxicity
from Code.utils.utils import ensure_dir, ceil_div, exact_div, whiten, reduce_mean, reduce_sum, reduce_std, clamp, flatten_dict

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

class PromptDataset(Dataset):
    def __init__(self, path):
        self.prompts = [json.loads(s.strip())["prompt"]["text"].strip() for s in open(path, 'r').readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx]}

class PromptDatasetForDebug(Dataset):
    def __init__(self, situation):
        self.prompts = [situation]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx]}


class PromptCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        prompts = [sequence['prompt'] for sequence in sequences]

        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return input_ids, attention_mask


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class AdaptiveKLController:
    def __init__(self, init_kl_coef, params):
        self.value = init_kl_coef
        self.params = params

    def update(self, current, n_steps):
        target = self.params.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.params.horizon
        self.value *= mult


class PPOTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy: Policy,
                 ref_policy: Policy,
                 value_model: Value,
                 score_model: Reward,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR):

        self.params = params
        self.policy = policy
        self.ref_policy = ref_policy
        self.value_model = value_model
        self.score_model = score_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_sampler = iter(self.train_dataloader)
        self.writer = SummaryWriter()

        if self.params.adaptive_kl:
            self.kl_ctl = FixedKLController(self.params.kl_coef)
        else:
            self.kl_ctl = AdaptiveKLController(self.params.kl_coef, params=self.params)

        self.params.minibatch_size = exact_div(self.params.batch_size, self.params.nminibatches)

    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        kl = logprobs - ref_logprobs
        non_score_reward = -self.kl_ctl.value * kl
        response_length = logprobs.size(1)
        score_reward = torch.tensor([[0.] * (l-1) + [s] + [0.] * (response_length - l) for s, l
                                     in zip(scores, torch.sum(masks, dim=1).tolist())], device=logprobs.device)
        rewards = non_score_reward + score_reward
        return rewards, non_score_reward, self.kl_ctl.value

    def train_minibatch(self, rollouts):
        """One step of PPO training."""
        self.optimizer.zero_grad()
        ppo_loss, stats = self.loss(rollouts)
        ppo_loss.backward()
        if self.params.clip_grad:
            torch.nn.utils.clip_grad_norm_(itertools.chain(self.policy.model.parameters(),
                                                           self.value_model.model.parameters()),
                                           self.params.max_grad_norm)
        self.optimizer.step()
        return stats

    def train(self, rollouts):
        stat_list = []

        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(self.params.noptepochs):
            order = np.random.permutation(self.params.batch_size)
            for mb_start in range(0, self.params.batch_size, self.params.minibatch_size):
                mb_data = {k: v[order[mb_start:mb_start + self.params.minibatch_size]] if type(v) == torch.Tensor else
                              [v[i] for i in order[mb_start:mb_start + self.params.minibatch_size].tolist()]
                           for k, v in rollouts.items()}
                stats = self.train_minibatch(mb_data)
                stat_list.append(stats)
        # Collect the stats. (They will be averaged later.)
        return {k: [s[k] for s in stat_list] for k in stat_list[0].keys()}

    def step(self, step_num):
        step_started_at = time.time()
        try:
            input_ids, attention_mask = next(self.train_sampler)
            assert len(input_ids) == self.params.batch_size, 'insufficient batch'
        except (StopIteration, AssertionError):
            self.train_sampler = iter(self.train_dataloader)
            input_ids, attention_mask = next(self.train_sampler)


        with torch.no_grad():
            rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask)
            forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                              'query_mask': rollouts['query/mask'],
                              'response_input_ids': rollouts['response/input_ids'],
                              'response_mask': rollouts['response/mask']}
            rollouts['response/value'] = self.value_model.forward_pass(**forward_inputs)['response/value']
            rollouts['response/value'] *= rollouts['response/mask']

            ref_logprobs = self.ref_policy.forward_pass(**forward_inputs)['response/log_prob']
        logprobs, masks = rollouts['response/log_prob'], rollouts['response/mask']
        scores = self.score_model.get_reward(rollouts['query/text'], rollouts['response/text'])
        rewards, non_score_reward, kl_coef = self.compute_rewards(scores, logprobs, ref_logprobs, masks)
        rollouts['rewards'] = rewards
        train_stats = self.train(rollouts=rollouts)
        data = {'scores': scores, 'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'non_score_reward': non_score_reward, 'train_stats': train_stats, 'kl_coef': kl_coef}
        stats = self.record_step_stats(data, step_num)
        for metric in ['kl', 'entropy', 'reward', 'reward_total']:
            self.writer.add_scalar(f'Objective/{metric}', stats[f'objective/{metric}'], step_num)
        for metric in ['policy', 'value', 'total']:
            self.writer.add_scalar(f'Loss/{metric}', stats[f'ppo/loss/{metric}'], step_num)

        self.kl_ctl.update(stats['objective/kl'], self.params.batch_size)
        self.print_samples(queries=rollouts['query/text'], responses=rollouts['response/text'], scores=scores,
                           logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, step=step_num)

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params.batch_size) / step_time
        log.info(f"[ppo_step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")
        self.save(step=step_num)
        self.eval(step=step_num)

    def record_step_stats(self, data, step):
        masks = data['masks']
        kl = data['logprobs'] - data['ref_logprobs']
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        mean_non_score_reward = torch.mean(reduce_sum(data['non_score_reward'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/kl_coef': self.params.kl_coef,
            'objective/entropy': mean_entropy.item(),
        }
        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = np.mean([x.item() for x in v])
        stats['objective/reward'] = np.mean(data['scores'])
        stats['objective/reward_total'] = np.mean(data['scores']) + mean_non_score_reward.item()

        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        steps = step + 1
        stats.update({
            'elapsed/updates': steps,
            'elapsed/steps/serial': steps * self.params.response_length,
            'elapsed/steps/total': steps * self.params.batch_size * self.params.response_length,
            'elapsed/episodes': steps * self.params.batch_size,
        })
        return stats

    def print_samples(self, queries, responses, scores, logprobs, ref_logprobs, masks, step):
        if step % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(10, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(queries[i] + responses[i])
            print(f"  score = {scores[i]:+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {scores[i] - self.params.kl_coef * sample_kl:+.2f}")

    def save(self, step):
        if step % self.params.save_interval != 0:
            return
        torch.save({
            'policy_model': self.policy.model.state_dict(),
            'value_model': self.value_model.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'{self.params.model_dir}/ckp_{step}.pth')
        log.info(f"[ppo_step {step}] model checkpoint saved")

    def eval(self, step):
        if step % self.params.eval_interval != 0:
            return
        log.info(f"[ppo_step {step}] evaluating ...")

        perplexities, divergences = [], []
        for i, (input_ids, attention_mask) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask)
                forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                                  'query_mask': rollouts['query/mask'],
                                  'response_input_ids': rollouts['response/input_ids'],
                                  'response_mask': rollouts['response/mask']}
                ref_logprobs = self.ref_policy.forward_pass(**forward_inputs)['response/log_prob']
                perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].float(), axis=1)
                perplexities.extend(perplexity.cpu().detach().numpy().tolist())
                score = self.score_model.get_reward(rollouts['query/text'], rollouts['response/text'])
                divergences.extend(score)

        ppl_score, divergence_score = np.mean(perplexities), np.mean(divergences)
        print(f"  perplexity = {ppl_score:+.2f}")
        print(f"  divergence = {divergence_score:+.2f}")
        self.writer.add_scalar('Evaluation/perplexity', ppl_score, step)
        self.writer.add_scalar('Evaluation/divergence', divergence_score, step)

    def loss(self, rollouts):
        values = rollouts['response/value']
        old_logprob = rollouts['response/log_prob']
        rewards = rollouts['rewards']
        masks = rollouts['response/mask']

        with torch.no_grad():
            if self.params.whiten_rewards:
                rewards = whiten(rewards, masks, shift_mean=False)
            lastgaelam = 0
            advantages_reversed = []
            #gen_length = self.params.response_length
            gen_length = rewards.size(1)
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + self.params.gamma * nextvalues - values[:, t]
                lastgaelam = delta + self.params.gamma * self.params.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            returns = advantages + values
            advantages = whiten(advantages, masks).detach()

        forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                          'query_mask': rollouts['query/mask'],
                          'response_input_ids': rollouts['response/input_ids'],
                          'response_mask': rollouts['response/mask']}
        outputs = self.policy.forward_pass(**forward_inputs)
        outputs['response/value'] = self.value_model.forward_pass(**forward_inputs)['response/value']
        outputs['response/value'] *= rollouts['response/mask']

        vpred = outputs['response/value']
        vpredclipped = clamp(vpred, values - self.params.cliprange_value, values + self.params.cliprange_value)
        vf_losses1 = torch.square(vpred - returns)
        vf_losses2 = torch.square(vpredclipped - returns)
        vf_loss = .5 * reduce_mean(torch.max(vf_losses1, vf_losses2), masks)
        vf_clipfrac = reduce_mean(torch.gt(vf_losses2, vf_losses1).float(), masks)

        logprob = outputs['response/log_prob']
        ratio = torch.exp(logprob - old_logprob)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, min=1.0 - self.params.cliprange, max=1.0 + self.params.cliprange)
        pg_loss = reduce_mean(torch.max(pg_losses, pg_losses2), masks)
        pg_clipfrac = reduce_mean(torch.gt(pg_losses2, pg_losses).float(), masks)

        loss = pg_loss + self.params.vf_coef * vf_loss

        entropy = reduce_mean(outputs['response/entropy'], masks)
        approxkl = .5 * reduce_mean(torch.square(logprob - old_logprob), masks)

        return_mean, return_var = reduce_mean(returns, masks), reduce_std(returns, masks)
        value_mean, value_var = reduce_mean(values, masks), reduce_std(values, masks)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=reduce_mean(vpred, masks), error=reduce_mean((vpred - returns) ** 2, masks),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var)
        )
        return loss, flatten_dict(stats, sep='/')


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

    time = datetime.now()
    date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
    args.save_dir = os.path.join(args.output_dir, date_time)
    args.reward_dir = os.path.join(args.save_dir, 'reward')
    args.model_dir = os.path.join(args.save_dir, 'model')
    args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
    for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
        ensure_dir(d)
    log.info(f'Write to output directory: {args.save_dir}')

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    log.info(f'Initializing models ...')
    ref_policy = Policy(model_name=args.init_model, device=device)
    policy = Policy(model_name=args.init_model, device=device)
    value = Value(model_type=args.init_model, device=device)
    reward = Reward(save_path=args.reward_dir,
                    batch_size=args.batch_size, gain=args.gain, bias=args.bias, device=2)
    log.info(f'Initialization done!')


    prompt_collator = PromptCollator(tokenizer=policy.tokenizer)
    train_dataset = PromptDataset(path=args.dataset_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=prompt_collator)
    log.info(f'Load train set with {len(train_dataset)} examples')

    val_dataset = PromptDataset(path=args.dataset_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prompt_collator)
    log.info(f'Load val set with {len(val_dataset)} examples')

    # normalize the rewards to have mean 0, var 1
    reward.set_reward_norm(dataloader=train_dataloader, policy=policy)

    # set up optimizer and scheduler
    optimizer = Adam(itertools.chain(policy.model.parameters(), value.model.parameters()), lr=args.lr, eps=1e-5)
    args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.total_steps)

    trainer = PPOTrainer(params=args, policy=policy, ref_policy=ref_policy, value_model=value, score_model=reward,
                         train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                         optimizer=optimizer, scheduler=scheduler)

    for step_num in range(args.total_steps):
        print(step_num)
        try:
            trainer.step(step_num)
        except RuntimeError:
           torch.cuda.empty_cache()
           continue


if __name__ == "__main__":
    main()

import json
import math
import os
import re
import numpy as np
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from typing import Optional, List, Iterable, Dict, Any
from Code.policy import Policy
from Code.model.delphi import DelphiScorer
from Code.utils.utils import batchify, load_jsonl
from scipy.special import rel_entr
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)



def my_jensenshannon(p, q, base=None, *, axis=0, keepdims=False):
    """
    Compute the Jensen-Shannon distance (metric) between
    two probability arrays. This is the square root
    of the Jensen-Shannon divergence.

    The Jensen-Shannon distance between two probability
    vectors `p` and `q` is defined as,

    .. math::

       \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}

    where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
    and :math:`D` is the Kullback-Leibler divergence.

    This routine will normalize `p` and `q` if they don't sum to 1.0.

    Parameters
    ----------
    p : (N,) array_like
        left probability vector
    q : (N,) array_like
        right probability vector
    base : double, optional
        the base of the logarithm used to compute the output
        if not given, then the routine uses the default base of
        scipy.stats.entropy.
    axis : int, optional
        Axis along which the Jensen-Shannon distances are computed. The default
        is 0.

        .. versionadded:: 1.7.0
    keepdims : bool, optional
        If this is set to `True`, the reduced axes are left in the
        result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        Default is False.

        .. versionadded:: 1.7.0

    Returns
    -------
    js : double or ndarray
        The Jensen-Shannon distances between `p` and `q` along the `axis`.

    Notes
    -----

    .. versionadded:: 1.2.0

    Examples


    """
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p, axis=axis, keepdims=True)
    q = q / np.sum(q, axis=axis, keepdims=True)
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    left_sum = np.sum(left, axis=axis, keepdims=keepdims)
    right_sum = np.sum(right, axis=axis, keepdims=keepdims)
    js = left_sum + right_sum
    if base is not None:
        js /= np.log(base)
    return js / 2.0


def load_model_t5(model_name_or_path, cuda_devices = None):
    cuda_devices = cuda_devices or []
    if len(cuda_devices) > 0:
        device = f"cuda:{cuda_devices[0]}"
    else:
        device = "cpu"
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, local_files_only=True)

    tokenizer = T5Tokenizer.from_pretrained('t5-large')

    model.to(device)
    return {"model": model, "tokenizer": tokenizer, "model_name": model_name_or_path, "cuda_device": device}

def predict_merge(situations, questions, answers):
    tokenizer = model_dict_fusion["tokenizer"]
    model = model_dict_fusion["model"]
    task_prefix = 'merge: '
    inputs = []
    for situation, question, answer in zip(situations, questions, answers):
        input = task_prefix+'SITUATION: ' + situation.strip() + ' QUESTION: ' + question.strip() + ' ANSWER: ' + answer.strip()
        inputs.append(input)
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(model_dict_fusion['cuda_device'])
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=False,  # disable sampling to test if batching affects output
        max_length=100,
    )
    predicted_merge= tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return predicted_merge

model_nli = RobertaForSequenceClassification.from_pretrained('alisawuffles/roberta-large-wanli').to("cuda:5")
tokenizer_nli = RobertaTokenizer.from_pretrained('alisawuffles/roberta-large-wanli')
model_dict_answer = load_model_t5('out_dir_t5_large_answergen', cuda_devices=[2])
model_dict_fusion = load_model_t5('checkpoint-11000', cuda_devices=[2])


def get_answers_from_model_batch_t5(situations, questions, judgements):
    # generate question
    tokenizer = model_dict_answer["tokenizer"]
    model = model_dict_answer["model"]
    bad = []
    good = []
    model_inputs = []
    uts = []
    all_situations = []
    counter = 0
    qs = []
    for jud, sit, q in zip(judgements, situations, questions):
        for ut in ['weakener', 'strengthener']:
            try:
                if not sit[-1] == '.':
                    sit = sit + '.'
            except IndexError:
                print('oh no')
                print(situations)
                print(judgements)
                print(questions)
            sit = re.sub('question: ', '', sit)
            input = "answer: "+ jud + ' ' + sit + ' TYPE: ' + ut + ' QUESTION: ' + q
            model_inputs.append(input)
            uts.append(ut)
            all_situations.append(sit)
            qs.append(q)
        counter += 1
    inputs = tokenizer(model_inputs, return_tensors="pt", padding=True).to(model_dict_answer['cuda_device'])

    response_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=True,
        max_length=200,
        top_p=0.6,
        top_k=None,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=None,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1  # max(1, args.beams)
    )
    pred = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    for sit, answer, ut, question in zip(all_situations, pred, uts, qs):
        x = tokenizer_nli(sit, answer, return_tensors='pt', max_length=128, truncation=True).to("cuda:5")
        logits = model_nli(**x).logits
        probs = logits.softmax(dim=1).squeeze(0)
        label_id = torch.argmax(probs).item()
        prediction_sit_ans = model_nli.config.id2label[label_id]
        if prediction_sit_ans in ['contradiction', 'entailment']:
            answer = ' '
        if ut == 'weakener':
            bad.append(answer)
        else:
            good.append(answer)
    return bad, good



class Reward:
    def __init__(self, save_path: str, device: str, batch_size: int,
                 gain: float = None, bias: float = None):
        self.gain, self.bias = gain, bias
        self.path = save_path
        self.batch_size = batch_size
        self.delphi_scorer = DelphiScorer(device_id=device)

    def set_reward_norm(self, dataloader: DataLoader, policy: Policy,
                        new_mean: int = 0., new_std: int = 1.):
        if self.gain is None and self.bias is None:
            log.info('compute reward statistics before normalization ...')
        else:
            log.info(f'reward after normalization: mean={new_mean}, std={new_std}')
            log.info(f'normalization factor: gain={self.gain}, bias={self.bias}')
            return
        good_sentences = []
        bad_sentences = []
        for i, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc='sampling from policy')):
            input_ids, attention_mask = batch
            outputs = policy.sample(input_ids=input_ids, attention_mask=attention_mask)

            # only generate one question
            prompts, responses = outputs['query/text'], outputs['response/text']

            class_labels, judgements = self.delphi_scorer.generate_batch(prompts)

            bad_answers, good_answers = get_answers_from_model_batch_t5(prompts, responses, judgements)
            good_answers = predict_merge(prompts, responses, good_answers)
            bad_answers = predict_merge(prompts, responses, bad_answers)
            good_sentences.extend(good_answers)
            bad_sentences.extend(bad_answers)
        divergences = []
        for good_sents, bad_sents in tqdm(zip(batchify(good_sentences, self.batch_size), batchify(bad_sentences, self.batch_size)), total=math.ceil(len(good_sentences) / self.batch_size), desc='computing rewards'):
            good_scores = self.delphi_scorer.score_batch(good_sents)
            bad_scores = self.delphi_scorer.score_batch(bad_sents)
            div = my_jensenshannon(good_scores, bad_scores, base=2, axis=1)
            divergences.extend(div)

        rewards = divergences

        old_mean, old_std = np.mean(rewards), np.std(rewards)
        log.info('statistics:')
        log.info(f'reward before normalization: mean={old_mean}, std={old_std}')

        self.gain = new_std / old_std
        self.bias = new_mean - self.gain * old_mean
        log.info(f'reward after normalization: mean={new_mean}, std={new_std}')
        log.info(f'normalization factor: gain={self.gain}, bias={self.bias}')

        json.dump({'old_mean': old_mean, 'old_std': old_std,
                   'new_mean': new_mean, 'new_std': new_std,
                   'gain': self.gain, 'bias': self.bias,
                   }, open(os.path.join(self.path, 'reward_normalization.json'), 'w'), indent=4)

    def get_reward(self, prompts: List[str], responses: List[str]) -> List[float]:

        assert len(prompts) == len(responses), f'prompts({len(prompts)}) and responses({len(responses)}) mismatch'

        class_labels, judgements = self.delphi_scorer.generate_batch(prompts)
        bad_answers, good_answers = get_answers_from_model_batch_t5(prompts, responses, judgements)
        good_sentences = predict_merge(prompts, responses, good_answers)
        bad_sentences = predict_merge(prompts, responses, bad_answers)
        divergences = []
        for good_sents, bad_sents in tqdm(zip(batchify(good_sentences, self.batch_size), batchify(bad_sentences, self.batch_size)), total=math.ceil(len(good_sentences) / self.batch_size), desc='computing rewards'):
            good_scores = self.delphi_scorer.score_batch(good_sents)
            bad_scores = self.delphi_scorer.score_batch(bad_sents)
            div = my_jensenshannon(good_scores, bad_scores, base=2, axis=1)
            divergences.extend(div)
        rewards = divergences
        return [self.gain * x + self.bias for x in rewards]


def toxicity_to_reward(score):
    return - score


def reward_to_toxicity(score):
    return - score

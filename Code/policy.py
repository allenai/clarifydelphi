import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'

import torch
import torch.nn.functional as F
from typing import Union, List, Dict
from transformers import T5ForConditionalGeneration, T5Tokenizer
from Code.utils.constants import NEGATIVE_INF
from Code.utils.utils import logits_to_entropy, mask_pad


class Policy:
    def __init__(self, model_name, device):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.max_length = 50

        self.device = device
        self.model = self.model.to(self.device)
        device_map = None
        if torch.cuda.device_count() == 8:
            device_map = {
                4: [0],
                5: [1, 2, 3, 4, 5, 6, 7],
                6: [8, 9, 10, 11, 12, 13, 14, 15],
                7: [16, 17, 18, 19, 20, 21, 22, 23],
            }
        if torch.cuda.device_count() == 6:
            device_map = {
                0: [0],
                1: [1, 2, 3],
                2: [4, 5, 6, 7, 8],
                3: [9, 10, 11, 12, 13],
                4: [14, 15, 16, 17, 18],
                5: [19, 20, 21, 22, 23],
            }
        elif torch.cuda.device_count() == 4:
            device_map = {
                0: [0],
                1: [1, 2, 3, 4, 5, 6, 7],
                2: [8, 9, 10, 11, 12, 13, 14, 15],
                3: [16, 17, 18, 19, 20, 21, 22, 23],
            }
        self.model.parallelize(device_map=device_map)

    def sample(self,
               prompts: Union[str, List[str]] = None,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               max_len: int = 30,
               min_len: int = 3,
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None) -> Dict[str, Union[torch.Tensor, List[str]]]:

        if prompts is not None:
            assert input_ids is None and attention_mask is None, 'repeated input'
            if isinstance(prompts, str):
                prompts = [prompts]

            encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding='max_length', truncation='longest_first', max_length=self.max_length)
            input_ids = encodings_dict['input_ids'].to(self.device)
            attention_mask = encodings_dict['attention_mask'].to(self.device)

        else:
            assert input_ids is not None, 'no input'
            prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        response_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            min_length=min_len,
            do_sample=sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        ) # begins with 0 ([BOS]); ends with 1 ([EOS])
        response_ids = response_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])
        response_mask = (response_ids != self.model.config.pad_token_id).int()
        response_text = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with torch.no_grad():
            # It seems impossible to merge this into generate() -- the "scores" returned by generate() is not correct, it contains mostly -inf
            outputs = self.forward_pass(input_ids, attention_mask, response_ids, response_mask)
        response_logits = outputs['response/logits']
        response_logprobs = outputs['response/log_prob']
        response_entropy = outputs['response/entropy']

        return {
            'query/text': prompts,
            'query/input_ids': input_ids,
            'query/mask': attention_mask,
            'response/text': response_text,
            'response/input_ids': response_ids,
            'response/mask': response_mask,
            'response/logits': response_logits,
            'response/log_prob': response_logprobs,
            'response/entropy': response_entropy,
        }

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor):

        outputs = self.model(
            input_ids=query_input_ids,
            attention_mask=query_mask,
            labels=mask_pad(response_input_ids, response_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        response_logits = outputs.logits # (B, RL-1, V)
        logprobs = F.log_softmax(response_logits, dim=-1)
        response_logprobs = torch.gather(logprobs, 2, response_input_ids[:, :, None]).squeeze(2) # (B, RL-1)
        response_entropy = logits_to_entropy(response_logits) # (B, RL-1)

        return {
            'response/logits': response_logits,
            'response/log_prob': mask_pad(response_logprobs, response_mask),
            'response/entropy': mask_pad(response_entropy, response_mask),
        }

if __name__ == "__main__":
    test = Policy('t5-large', 0.7, 'cuda:0','t5-base')
    output = test.sample(prompts=['I like dogs.', 'a boy'], sample=False)
    test.forward_pass(output['query/input_ids'], output['query/mask'], output['response/input_ids'], output['response/mask'])
    from IPython import embed
    embed()

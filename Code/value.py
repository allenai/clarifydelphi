import torch
from transformers import T5Tokenizer
from Code.model.t5 import T5ForTokenRegression
from Code.utils.utils import mask_pad

from IPython import embed


class Value:
    def __init__(self, model_type, device):
        self.model = T5ForTokenRegression.from_pretrained(model_type)
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
        self.model.encoder.parallelize(device_map=device_map)
        self.model.decoder.parallelize(device_map=device_map)
        self.model.model_parallel = True
        self.tokenizer = T5Tokenizer.from_pretrained(model_type)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor):

        query_input_ids = query_input_ids.to(self.device)
        query_mask = query_mask.to(self.device)
        response_input_ids = response_input_ids.to(self.device)
        response_mask = response_mask.to(self.device)

        outputs = self.model(
            input_ids=query_input_ids,
            attention_mask=query_mask,
            labels=mask_pad(response_input_ids, response_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return {
            'response/value': mask_pad(outputs.logits, response_mask)
        }

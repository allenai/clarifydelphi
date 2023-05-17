import sys
sys.path.append(".")

import torch
from scipy.special import softmax
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


class DelphiScorer:
    def __init__(self, device_id="cuda:0", model="t5-11b", parallel=False):
        CUDA_DEVICE = device_id if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(CUDA_DEVICE)
        print(f"DelphiScorer device: {self.device}")

        if model == "t5-large":
            MODEL_BASE = "t5-large"
            MODEL_LOCATION = "large_commonsense_morality_hf"
            self.class_token_pos = 4
            self.sep_tokens = ["<unk> /class> <unk> text>", " class>", "<unk> /text>"]

        elif model == "t5-11b":
            MODEL_BASE = "t5-11b"
            MODEL_LOCATION = "11b_commonsense_morality_hf"
            self.class_token_pos = 4
            self.sep_tokens = ["<unk> /class> <unk> text>", " class>", "<unk> /text>"]

        else:
            print("ERROR: model doesn't exist")
            return

        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_LOCATION)
        self.model.to(self.device)
        if parallel:
            self.model.parallelize()
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_BASE)

        self.class1_pos = 0
        self.class0_pos = 1
        self.classminus1_pos = 2

    def score(self, input_string, normalize=None):
        input_string = f"[moral_single]: {input_string}"
        input_ids = self.tokenizer(input_string, return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids, max_length=200, output_scores=True, return_dict_in_generate=True)

        probs = [(self.tokenizer.decode(i), x) for (i, x) in enumerate(outputs['scores'][self.class_token_pos][0].softmax(0))]

        class1_prob = sum([v[1].item() for v in probs if v[0] == "1"])
        class0_prob = sum([v[1].item()  for v in probs if v[0] == "0"])
        classminus1_prob = sum([v[1].item()  for v in probs if v[0] == "-1"])

        probs = [class1_prob, class0_prob, classminus1_prob]
        probs_sum = sum(probs)

        if normalize == "regular":
            probs = [p / probs_sum for p in probs]
        elif normalize == "softmax":
            probs = softmax(probs)

        return probs

    def score_batch(self, input_strings, normalize=None):
        input_strings = [f"[moral_single]: {x}" for x in input_strings]
        inputs = {k: v.to(self.device) for k, v in self.tokenizer(input_strings, return_tensors='pt', padding=True).items()}
        outputs = self.model.generate(**inputs, max_length=200, output_scores=True, return_dict_in_generate=True)

        probs = outputs['scores'][self.class_token_pos].softmax(-1)

        class1_prob = probs[:, self.tokenizer.convert_tokens_to_ids("1")]
        class0_prob = probs[:, self.tokenizer.convert_tokens_to_ids("0")]
        classminus1_prob = probs[:, self.tokenizer.convert_tokens_to_ids("-1")]

        probs = torch.stack([class1_prob, class0_prob, classminus1_prob], dim=-1)
        probs_sum = torch.sum(probs, dim=1)

        if normalize == "regular":
            probs = probs / probs_sum
        elif normalize == "softmax":
            probs = probs.softmax(-1)

        return probs.tolist()

    def compute_toxicity(self, input_string, normalize=None):
        score = self.score(input_string, normalize)
        return score[self.classminus1_pos] - score[self.class1_pos]

    def compute_toxicity_batch(self, input_strings, normalize=None):
        scores = self.score_batch(input_strings, normalize)
        toxicities = [s[self.classminus1_pos] - s[self.class1_pos] for s in scores]
        return toxicities

    def generate(self, input_string):
        input_string = f"[moral_single]: {input_string}"
        input_ids = self.tokenizer(input_string, return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids, max_length=200, output_scores=True, return_dict_in_generate=True)

        decoded_sequence = self.tokenizer.decode(outputs["sequences"][0])
        class_label = int(decoded_sequence.split(self.sep_tokens[0])[0].split(self.sep_tokens[1])[-1])
        text_label = decoded_sequence.split(self.sep_tokens[0])[-1].split(self.sep_tokens[2])[0]

        return class_label, text_label

    def generate_batch(self, input_strings):
        input_strings = [f"[moral_single]: {input_string}" for input_string in input_strings]
        input_ids = self.tokenizer(input_strings, return_tensors='pt', padding=True, truncation=True).to(self.device).input_ids
        outputs = self.model.generate(input_ids, max_length=200, output_scores=True, return_dict_in_generate=True)

        decoded_sequences = [self.tokenizer.decode(output) for output in outputs["sequences"]]
        class_labels = [int(decoded_sequence.split(self.sep_tokens[0])[0].split(self.sep_tokens[1])[-1]) for decoded_sequence in decoded_sequences]
        text_labels = [decoded_sequence.split(self.sep_tokens[0])[-1].split(self.sep_tokens[2])[0] for decoded_sequence in decoded_sequences]

        return class_labels, text_labels

    def generate_beam(self,
                      input_string,
                      num_beams=5,
                      max_length=50,
                      num_return_sequences=5,):
        input_string = f"[moral_single]: {input_string}"
        input_ids = self.tokenizer(input_string,
                                   max_length=16,
                                   truncation=True,
                                   return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids,
                                      # output_scores=True,
                                      # return_dict_in_generate=True,
                                      num_beams=num_beams,
                                      max_length=max_length,
                                      num_return_sequences=num_return_sequences,)

        decoded_sequences = self.tokenizer.batch_decode(outputs)

        class_labels = [ds.split(self.sep_tokens[0])[0].split(self.sep_tokens[1])[-1] for ds in decoded_sequences]
        text_labels = [ds.split(self.sep_tokens[0])[-1].split(self.sep_tokens[2])[0] for ds in decoded_sequences]

        return class_labels, text_labels

    def generate_with_score(self, input_string):
        input_string = f"[moral_single]: {input_string}"
        input_ids = self.tokenizer(input_string, return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids, max_length=200, output_scores=True, return_dict_in_generate=True)

        probs = [(self.tokenizer.decode(i), x) for (i, x) in enumerate(outputs['scores'][self.class_token_pos][0].softmax(0))]

        class1_prob = sum([v[1].item() for v in probs if v[0] == "1"])
        class0_prob = sum([v[1].item()  for v in probs if v[0] == "0"])
        classminus1_prob = sum([v[1].item()  for v in probs if v[0] == "-1"])

        probs = [class1_prob, class0_prob, classminus1_prob]
        # probs_sum = sum(probs)

        decoded_sequence = self.tokenizer.decode(outputs["sequences"][0])
        class_label = int(decoded_sequence.split(self.sep_tokens[0])[0].split(self.sep_tokens[1])[-1])
        text_label = decoded_sequence.split(self.sep_tokens[0])[-1].split(self.sep_tokens[2])[0]

        return class_label, probs, text_label

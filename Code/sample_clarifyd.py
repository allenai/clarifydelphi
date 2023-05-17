import torch
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
from Code.policy import Policy
from torch.utils.data import DataLoader
from Code.lean_main import PromptDataset, PromptCollator


def expand(tensor, num_repeat):
    return torch.reshape(tensor[:, None].expand(-1, num_repeat, -1), [batch_size * num_repeat, -1])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def distinctness(responses, num_sample):
    generations_batch = list(chunks(responses, num_sample))
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for generations in tqdm(generations_batch, total=len(generations_batch), desc='Evaluating diversity'):
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)

if __name__ == "__main__":
    model = 'PATH_TO_MODEL_DIR'
    batch_size = 4
    num_samples = 1
    checkpoint_path = 'CHECKPOINT_PATH'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    policy = Policy(model_name=model, device=device)
    prompt_collator = PromptCollator(tokenizer=policy.tokenizer)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        policy.model.load_state_dict(checkpoint['policy_model'])

    print('model initialization done!')

    val_dataset = PromptDataset(path='data/dev_with_prefix.jsonl')
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=prompt_collator, drop_last=True)


    perplexities, prompts, responses = [], [], []
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        input_ids, attention_mask = batch

        outputs = policy.sample(input_ids=expand(input_ids, num_samples), attention_mask=expand(attention_mask, num_samples), top_p=0.6, sample=True)
        prompt, response = outputs['query/text'], outputs['response/text']
        prompts.extend([x for n, x in enumerate(prompt) if not n % num_samples])
        responses.extend(response)
    data = pd.DataFrame.from_dict({'prompt': prompts})
    outfile = csv.writer(open('predictions.tsv', 'w'))
    for d, r in zip(data["prompt"], responses):
        print(d)
        print(r)
        outfile.writerow([d, r])

import yaml
from pathlib import Path

import torch
import fire
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

from open_r1.choreo import parse_interleaved

def accuracy(preds, refs):
    pass

def eval(
    checkpoint_path: Path,
    train_config_path: Path,
    batch_size: int = 8,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    num_return_sequences: int = 1,
):
    with open(train_config_path) as f:
        train_config = yaml.safe_load(f)

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.chat_template = train_config['chat_template']
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model, tokenizer = accelerator.prepare(model, tokenizer)s

    ds = load_dataset('tjbai/open-rs-exp', split='test')
    preds, refs = [], []

    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds[i : i + batch_size]

        messages = [[
            {'role': 'system', 'content': train_config['system_prompt']},
            {'role': 'user', 'content': p}
        ] for p in batch['problem']]

        messages = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = tokenizer(
            batch['problem'],
            padding=True,
            padding_side='left',
            return_tensors='pt',
            add_special_tokens=False,
        ).to(accelerator.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences
        )

        completions = parse_interleaved(
            tokenizer,
            inputs,
            outputs,
            train_config['choreography_k']
        )

        preds.extend(completions)
        refs.extend(batch['solution'])

        print(completions[0])

if __name__ == "__main__":
    fire.Fire(eval)

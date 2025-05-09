from datasets import DatasetDict, load_dataset

# create a 90/10 train/dev from open-rs for experimentation
ds = load_dataset('knoveleng/open-rs')['train']
splits = ds.train_test_split(test_size=0.1, seed=42)

ds = DatasetDict({
    "train": splits["train"],
    "test":  splits["test"]
})

ds.push_to_hub('tjbai/open-rs-exp')

import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
import os
import re
from importlib.metadata import version
import tiktoken

print("tiktoken version: ", version("tiktoken"))

def get_training_dataset():
    file_folder = "data"
    file_name = "the-verdict.txt"
    file_path = os.path.join(file_folder, file_name)
    if os.path.exists(file_path) is False:
        # Create directory if it doesn't exist
        os.makedirs(file_folder, exist_ok=True)
        # Download file
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/" + file_name
        urllib.request.urlretrieve(url, file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()
    return raw_text

training_dataset = get_training_dataset()
print("Total number of characters: ", len(training_dataset))
print(training_dataset[:99])


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True,
                         drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,                # Dataset object
        batch_size=batch_size,  # Number of samples per batch
        shuffle=shuffle,        # Whether to shuffle data each epoch
        drop_last=drop_last,    # Whether to drop incomplete last batch
        num_workers=num_workers # Number of parallel workers
    )
    return dataloader

dataloader = create_dataloader_v1(training_dataset, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
print(next(data_iter))
import urllib.request
import torch
import os
import re

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


def preprocess_text(input_str):
    # Use regex to split on whitespace or any non-word character
    split_str = re.split(r'(\s|--|[,.:;?_!"()\'])', input_str)
    # Filter out empty strings (whitepsaces)
    split_str = [item for item in split_str if item.strip()]
    return split_str
#print(preprocess_text("Hello, world. Is this-- a test?"))
preprocessed_dataset = preprocess_text(training_dataset)
print("Total number of preprocessed characters: ", len(preprocessed_dataset))
print(preprocessed_dataset[:30])

def get_vocabular(input):
    vocab = sorted(set(input))
    return vocab

all_words = get_vocabular(preprocessed_dataset)
vocab_size = len(all_words)
print("Vocab size: ", vocab_size)
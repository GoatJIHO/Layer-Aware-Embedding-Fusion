# dataset_loader.py
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import torch

def load_label(data_dir, dataset_name):
    dataset = load_from_disk(f"{data_dir}/{dataset_name}")
    return dataset['train']['label'], dataset['test']['label']

def load_embedding(data_dir, dataset_name, model_name):
    train_path = f"{data_dir}/{dataset_name}/{model_name}_train_embedding.pt"
    test_path  = f"{data_dir}/{dataset_name}/{model_name}_test_embedding.pt"

    train_embed = torch.load(train_path)
    test_embed  = torch.load(test_path)

    return train_embed, test_embed

class CustomDataset(Dataset):
    def __init__(self, *args):
        self.embeddings = args[:-1]
        self.labels = args[-1]
        if len(args) < 2 or len(args) > 6:
            raise ValueError("CustomDataset: Unsupported dataset.")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if type(self.embeddings[0]) != tuple:
            return self.embeddings[0][idx], self.labels[idx]
        return tuple(embed[idx] for embed in self.embeddings[0]) + (self.labels[idx],)
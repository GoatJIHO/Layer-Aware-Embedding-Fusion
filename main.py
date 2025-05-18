import os, gc
import torch
import argparse
import numpy as np
from dotenv import load_dotenv

from trainer import train_fuc
from models.exp_models import ExpModels
from data_processing.dataset_loader import load_embedding, load_label

load_dotenv('.env')
EMBEDDING_DIR = os.getenv("EMBEDDING_DIR", "")
DATA_DIR = os.getenv("DATA_DIR", "")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_id", type=str, required=True)
    parser.add_argument("--model_id1", type=str, required=True)
    parser.add_argument("--model_id2", type=str, required=True)
    args = parser.parse_args()

    train_label, test_label = load_label(DATA_DIR, args.data_id)
    train_embedding1, test_embedding1 = load_embedding(EMBEDDING_DIR, args.data_id, args.model_id1)
    train_embedding2, test_embedding2 = load_embedding(EMBEDDING_DIR, args.data_id, args.model_id2)

    exp = ExpModels()

    for model_name in exp.get_available_model_for_dual_embedding():
        print(f"> Model: {model_name}")
        model = exp.create_model(model_name, train_embedding1.shape[1], train_embedding2.shape[1], len(np.unique(train_label)))
        model = model.to(device)
        train_fuc(model, (train_embedding1, train_embedding2), (test_embedding1, test_embedding2), train_label, test_label, device)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
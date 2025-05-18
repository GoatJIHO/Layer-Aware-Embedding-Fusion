## all model embedding extrat

import os
import gc
import torch
from dotenv import load_dotenv
from models.model_load import ModelLoader
from data_processing.embedding_extractor import run_extraction_for_all_splits
from datasets import load_from_disk

load_dotenv('.env')
EMBEDDING_DIR = os.getenv("EMBEDDING_DIR", "")
DATA_DIR = os.getenv("DATA_DIR", "")

model_list = ['gemma', 'qwen', 'llama', 'falcon', 'mistral', 'nv_embed', 'e5']
data_list = ['r52', 'mr', 'r8', 'r52']


for model_id in model_list:
    for data_id in data_list:
        dataset = load_from_disk(f"{DATA_DIR}/{data_id}/")
        model, tokenizer = ModelLoader(model_id).get_model()
    
        run_extraction_for_all_splits(
            model=model,
            tokenizer=tokenizer,
            dataset_name=data_id,
            file_model_name=model_id,
            datasets=dataset
        )
        
        del model, tokenizer
        model_loader.unload()

        gc.collect()
        torch.cuda.empty_cache()

    


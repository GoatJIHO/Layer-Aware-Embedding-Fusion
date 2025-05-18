# extraction_embedding.py
import os
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from dotenv import load_dotenv

load_dotenv('.env')
EMBEDDING_DIR = os.getenv("EMBEDDING_DIR", "")

def get_last_token_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str], 
    batch_size: int = 2,
):
    model.eval()
    embeddings = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            if "NV" in model.name_or_path or "e5" in model.name_or_path:
                outputs = model(**inputs)
            else:
                outputs = model(**inputs, output_hidden_states=True)
    
            if "NV" in model.name_or_path:
                batch_embeds = outputs["sentence_embeddings"].mean(dim=1).cpu().numpy()
            elif hasattr(outputs, "last_hidden_state"):
                batch_embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            elif hasattr(outputs, "hidden_states"):
                batch_embeds = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
            else:
                raise ValueError("Unsupported model output format.")

            embeddings.extend(batch_embeds)
            del inputs, outputs
            torch.cuda.empty_cache()

    return embeddings


def save_embedding_from_func(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    save_path: str,
):
    full_path = os.path.join(EMBEDDING_DIR, f"{save_path}.pt")
    embeddings = get_last_token_embeddings(model, tokenizer, texts)
    tensor = torch.tensor(embeddings)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    torch.save(tensor, full_path)


def run_extraction_for_all_splits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    file_model_name: str,
    datasets: Dict[str, Tuple[List[str], List[str]]],
):

    save_embedding_from_func(model, tokenizer, datasets['train']['text'], f"{dataset_name}/{file_model_name}_train_embedding")
    save_embedding_from_func(model, tokenizer, datasets['test']['text'], f"{dataset_name}/{file_model_name}_test_embedding")

    

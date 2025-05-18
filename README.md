# LLM Embedding Fusion Experiment

This project provides a framework for extracting embeddings from large language models (LLMs) and training fusion-based classifiers using the extracted representations.

## 📁 Project Structure

```
.
├── data/
│   ├── origin_data/          # Raw datasets stored in HuggingFace Dataset format
│   ├── extracted_embedding/  # Extracted embeddings (.pt files)
│
├── models/
│   ├── model_load.py         # LLM model loader (AutoModel / AutoTokenizer)
│   ├── exp_models.py         
|   ├── model_zoo.py          # Fusion model architectures
│
├── data/
│   ├── dataset_loader.py     # Functions to load labels and embeddings
│   ├── embedding_extractor.py # Functions to extract and save embeddings
│
├── main.py                  # Runs single experiment
├── run_all_experiments.py  # Runs batch experiments over all models and datasets
├── .env                    # Environment variables (paths)
```

---

## 🧾 Dataset Format

All datasets must be saved under:
```
data/origin_data/{dataset_name}/
```
Each dataset should follow the Hugging Face `DatasetDict` format:
```python
DatasetDict({
  'train': Dataset({ 'text': [...], 'label': [...] }),
  'test': Dataset({ 'text': [...], 'label': [...] })
})
```

---

## 📌 Embedding Extraction

To extract embeddings from supported LLMs:
```bash
python extract_embedding.py
```
This will save output `.pt` files under:
```
data/extracted_embedding/{dataset_name}/{model_name}_train_embedding.pt
```

---

## 🧪 Run a Training Experiment

To run a dual-embedding classification experiment:
```bash
python main.py \
  --data_id {dataset_name} \
  --model_id1 {model_name_1} \
  --model_id2 {model_name_2}
```
Example:
```bash
python main.py --data_id r52 --model_id1 gemma --model_id2 e5
```

---

## 🔧 Environment Variables (.env)

```dotenv
MODEL_CACHE_DIR=/your/hf_cache
DATA_DIR = ./data/origin_data
EMBEDDING_DIR = ./data/extracted_embedding
```

---

## ✅ Embedding Models

- `gemma`
- `qwen`
- `llama`
- `falcon`
- `mistral`
- `nv_embed`
- `e5`

---

## 🧱 Modular Components

- `embedding_extractor.py` handles embedding extrating logic 
- `exp_models.py` defines fusion architectures (e.g. Hadamard, Sum, Quaternion, MoE)
- `trainer.py` provides training loop with validation tracking

---
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from dotenv import load_dotenv
import torch
import os
import gc

class ModelLoader:
    def __init__(self, model_name: str):
        load_dotenv(".env")
        self.cache_dir = os.getenv("MODEL_CACHE_DIR", "")
        self.model_name = model_name
        self.model_dict = {
            "gemma": "google/gemma-2-2b", 
            "qwen": "Qwen/Qwen2.5-7B",
            "llama": "meta-llama/Llama-2-7b-chat-hf",
            "falcon": "tiiuae/Falcon3-7B-Instruct",
            "mistral": "mistralai/Mistral-7B-v0.3",
            "nv_embed": "nvidia/NV-Embed-v2",
            "e5": "intfloat/e5-large-v2"
        }

        if self.model_name not in self.model_dict:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        pretrained_id = self.model_dict[self.model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_id, cache_dir=self.cache_dir, trust_remote_code=True)
        if self.model_name == "e5":
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model = AutoModel.from_pretrained(pretrained_id, cache_dir=self.cache_dir, trust_remote_code=True, device_map="auto")
        elif self.model_name == "nv_embed":
            self.model = AutoModel.from_pretrained(pretrained_id, cache_dir=self.cache_dir, trust_remote_code=True, device_map="auto")
        elif self.model_name == "gemma":
            self.model = AutoModel.from_pretrained(pretrained_id, cache_dir=self.cache_dir, trust_remote_code=True).to("cuda:0")
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_id, cache_dir=self.cache_dir, device_map="auto")

    def get_model(self):
        return self.model, self.tokenizer

    def unload(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

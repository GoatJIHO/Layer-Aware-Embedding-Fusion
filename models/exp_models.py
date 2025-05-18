# exp_models.py
import torch.nn as nn
import torch.nn.functional as F
from .model_zoo import *  

class ExpModels:
    single_model_dict = {
        "SingleEmbeddingModel": SingleEmbeddingModel
    }
    dual_model_dict = {
        "TwoEmbeddingHadamardResidualModel": TwoEmbeddingHadamardResidualModel,
        "TwoEmbeddingSumResidualModel": TwoEmbeddingSumResidualModel,
        "TwoEmbeddingMatmulResidualModel": TwoEmbeddingMatmulResidualModel,
        "TwoEmbeddingQuaternionResidualModel": TwoEmbeddingQuaternionResidualModel
    }

    model_dict = {**single_model_dict, **dual_model_dict}

    @staticmethod
    def get_available_model_for_single_embedding():
        return list(ExpModels.single_model_dict.keys())

    @staticmethod
    def get_available_model_for_dual_embedding():
        return list(ExpModels.dual_model_dict.keys())

    @staticmethod
    def create_model(model_name, *args):
        if model_name not in ExpModels.model_dict:
            raise ValueError(f"Model '{model_name}' not found. Available models: {ExpModels.get_available_models()}")
        return ExpModels.model_dict[model_name](*args)

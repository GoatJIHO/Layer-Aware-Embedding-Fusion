import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def quaternion_mul(q1, q2):
    a1, b1, c1, d1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
    a2, b2, c2, d2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]

    a = a1*a2 - b1*b2 - c1*c2 - d1*d2
    b = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c = a1*c2 - b1*d2 + c1*a2 + d1*b2
    d = a1*d2 + b1*c2 - c1*b2 + d1*a2
    
    return torch.stack([a, b, c, d], dim=-1)


class BaseModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BaseModel, self).__init__()
        self.fusion_fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        output = self.fusion_fc(x)
        return output

class SingleEmbeddingModel(BaseModel):
    def __init__(self, embed_dim1, num_classes, normalize=False):
        super(SingleEmbeddingModel, self).__init__(embed_dim1, num_classes)
        self.normalize = normalize
    def forward(self, x1):
        if self.normalize:
            x1 = F.normalize(x1, p=2, dim=1)
        return super().forward(x1)
    
    
class TwoEmbeddingConcatModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        super(TwoEmbeddingConcatModel, self).__init__(embed_dim1 + embed_dim2, num_classes)
        self.normalize = normalize
    def forward(self, x1, x2):
        fusion_features = torch.cat((x1, x2), dim=-1)
        if self.normalize:
            fusion_features = F.normalize(fusion_features, p=2, dim=1)
        return super().forward(fusion_features)


class TwoEmbeddingHadamardModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        smaller_dim = min(embed_dim1, embed_dim2)
        super(TwoEmbeddingHadamardModel, self).__init__(smaller_dim, num_classes)
        self.normalize = normalize
        self.decrease_embed = None
        if embed_dim1 != embed_dim2: 
            larger_dim, smaller_dim = max(embed_dim1, embed_dim2), min(embed_dim1, embed_dim2)
            self.decrease_embed = nn.Linear(larger_dim, smaller_dim)
                
    def forward(self, x1, x2):
        if self.decrease_embed is not None:
            if x1.shape[1] > x2.shape[1]:
                x1 = self.decrease_embed(x1)
                x1 = F.normalize(x1, p=2, dim=1)
            else:
                x2 = self.decrease_embed(x2)
                x2 = F.normalize(x2, p=2, dim=1)
            
        x_hadamard = x1 * x2
        if self.normalize:
            x_hadamard = F.normalize(x_hadamard, p=2, dim=1)
        return super().forward(x_hadamard)

    
class TwoEmbeddingHadamardResidualModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        smaller_dim = min(embed_dim1, embed_dim2)
        super(TwoEmbeddingHadamardResidualModel, self).__init__(embed_dim1 + embed_dim2 + smaller_dim , num_classes)
        self.normalize = normalize
        self.decrease_embed = None
        if embed_dim1 != embed_dim2:  
            larger_dim, smaller_dim = max(embed_dim1, embed_dim2), min(embed_dim1, embed_dim2)
            self.decrease_embed = nn.Linear(larger_dim, smaller_dim)
            
    def forward(self, x1, x2):
        origin_x1 = x1
        origin_x2 = x2
        if self.decrease_embed is not None:
            if x1.shape[1] > x2.shape[1]:
                x1 = self.decrease_embed(x1)
                x1 = F.normalize(x1, p=2, dim=1)
            else:
                x2 = self.decrease_embed(x2)
                x2 = F.normalize(x2, p=2, dim=1)
        
        x_hadamard = x1 * x2
        
        if self.normalize:
            x_hadamard = F.normalize(x_hadamard, p=2, dim=1)
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
            
        fusion_features = torch.cat((origin_x1, origin_x2, x_hadamard), dim=-1)
        return super().forward(fusion_features)

class TwoEmbeddingSumModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        smaller_dim = min(embed_dim1, embed_dim2)
        super(TwoEmbeddingSumModel, self).__init__(smaller_dim , num_classes)
        self.normalize = normalize
        self.decrease_embed = nn.Linear(smaller_dim, smaller_dim)
        self.smaller_embed = nn.Linear(smaller_dim, smaller_dim)
        if embed_dim1 != embed_dim2: 
            larger_dim, smaller_dim = max(embed_dim1, embed_dim2), min(embed_dim1, embed_dim2)
            self.decrease_embed = nn.Linear(larger_dim, smaller_dim)
            self.smaller_embed = nn.Linear(smaller_dim, smaller_dim)
            
    def forward(self, x1, x2):
        if x1.shape[1] > x2.shape[1]:
            x1 = self.decrease_embed(x1)
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = self.smaller_embed(x2)
            x2 = F.normalize(x2, p=2, dim=1)
        else:
            x2 = self.decrease_embed(x2)
            x2 = F.normalize(x2, p=2, dim=1)
            x1 = self.smaller_embed(x1)
            x1 = F.normalize(x1, p=2, dim=1)

        x_sum = x1 + x2
        
        if self.normalize:
            x_hadamard = F.normalize(x_hadamard, p=2, dim=1)
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
            
        return super().forward(x_sum)

class TwoEmbeddingSumResidualModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        smaller_dim = min(embed_dim1, embed_dim2)
        super(TwoEmbeddingSumResidualModel, self).__init__(embed_dim1 + embed_dim2 + smaller_dim , num_classes)
        self.normalize = normalize
        self.decrease_embed = nn.Linear(smaller_dim, smaller_dim)
        self.smaller_embed = nn.Linear(smaller_dim, smaller_dim)
        if embed_dim1 != embed_dim2:  #차원이 다를 경우 통일 시켜야함
            larger_dim, smaller_dim = max(embed_dim1, embed_dim2), min(embed_dim1, embed_dim2)
            self.decrease_embed = nn.Linear(larger_dim, smaller_dim)
            self.smaller_embed = nn.Linear(smaller_dim, smaller_dim)
            
    def forward(self, x1, x2):
        origin_x1 = x1
        origin_x2 = x2
        if x1.shape[1] > x2.shape[1]:
            x1 = self.decrease_embed(x1)
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = self.smaller_embed(x2)
            x2 = F.normalize(x2, p=2, dim=1)
        else:
            x2 = self.decrease_embed(x2)
            x2 = F.normalize(x2, p=2, dim=1)
            x1 = self.smaller_embed(x1)
            x1 = F.normalize(x1, p=2, dim=1)

        x_sum = x1 + x2
        
        if self.normalize:
            x_hadamard = F.normalize(x_hadamard, p=2, dim=1)
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
            
        fusion_features = torch.cat((origin_x1, origin_x2, x_sum), dim=-1)
        return super().forward(fusion_features)

class TwoEmbeddingMatmulModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        super(TwoEmbeddingMatmulModel, self).__init__(1024 , num_classes)
        self.normalize = normalize
        self.decrease_embed = nn.Linear(embed_dim1, 1024)
        self.smaller_embed = nn.Linear(embed_dim2, 1024)
        
            
    def forward(self, x1, x2):
        x1 = self.decrease_embed(x1)
        x2 = self.smaller_embed(x2)

        x1 = x1.view(-1, 32, 32)
        x2 = x2.view(-1, 32, 32)

        x_matmul = torch.matmul(x1, x2)
        x_matmul = x_matmul.view(-1, 1024)
        x_matmul = F.normalize(x_matmul, p=2, dim=1)

        return super().forward(x_matmul)


class TwoEmbeddingMatmulResidualModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        super(TwoEmbeddingMatmulResidualModel, self).__init__(embed_dim1 + embed_dim2 + 1024 , num_classes)
        self.normalize = normalize
        self.decrease_embed = nn.Linear(embed_dim1, 1024)
        self.smaller_embed = nn.Linear(embed_dim2, 1024)
        
            
    def forward(self, x1, x2):
        origin_x1 = x1
        origin_x2 = x2
        x1 = self.decrease_embed(x1)
        x2 = self.smaller_embed(x2)

        x1 = x1.view(-1, 32, 32)
        x2 = x2.view(-1, 32, 32)

        x_matmul = torch.matmul(x1, x2)
        x_matmul = x_matmul.view(-1, 1024)
        x_matmul = F.normalize(x_matmul, p=2, dim=1)
        
        fusion_features = torch.cat((origin_x1, origin_x2, x_matmul), dim=-1)
        return super().forward(fusion_features)

class TwoEmbeddingQuaternionModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        super(TwoEmbeddingQuaternionModel, self).__init__(1024 , num_classes)
        self.normalize = normalize
        self.decrease_embed = nn.Linear(embed_dim1, 1024)
        self.smaller_embed = nn.Linear(embed_dim2, 1024)
        
            
    def forward(self, x1, x2):
        x1 = self.decrease_embed(x1)
        x2 = self.smaller_embed(x2)

        bsz = x1.shape[0]
        x1_q = x1.view(bsz, 1024 // 4, 4)
        x2_q = x2.view(bsz, 1024 // 4, 4)

        x_quat = quaternion_mul(x1_q, x2_q) 

        x_quat_flat = x_quat.view(bsz, -1)

        return super().forward(x_quat_flat)

class TwoEmbeddingQuaternionResidualModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        super(TwoEmbeddingQuaternionResidualModel, self).__init__(embed_dim1 + embed_dim2 + 1024 , num_classes)
        self.normalize = normalize
        self.decrease_embed = nn.Linear(embed_dim1, 1024)
        self.smaller_embed = nn.Linear(embed_dim2, 1024)
        
            
    def forward(self, x1, x2):
        origin_x1 = x1
        origin_x2 = x2
        x1 = self.decrease_embed(x1)
        x2 = self.smaller_embed(x2)

        bsz = x1.shape[0]
        x1_q = x1.view(bsz, 1024 // 4, 4)
        x2_q = x2.view(bsz, 1024 // 4, 4)

        x_quat = quaternion_mul(x1_q, x2_q) 

        x_quat_flat = x_quat.view(bsz, -1)

        fusion_features = torch.cat((origin_x1, origin_x2, x_quat_flat), dim=-1)
        return super().forward(fusion_features)


class AllFeatureModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        super(AllFeatureModel, self).__init__(embed_dim1 + embed_dim2 + 1024 * 4 , num_classes)
        self.normalize = normalize
        self.decrease_embed = nn.Linear(embed_dim1, 1024)
        self.smaller_embed = nn.Linear(embed_dim2, 1024)
            
    def forward(self, x1, x2):
        origin_x1 = x1
        origin_x2 = x2
        #sum
        x1_s = self.decrease_embed(x1)
        x2_s = self.smaller_embed(x2)
        x_sum = x1_s + x2_s
        x_sum = F.normalize(x_sum, p=2, dim=1)
        #hadar
        x_hardamard = x1_s * x2_s
        x_hardamard = F.normalize(x_hardamard, p=2, dim=1)
        #matmul
        x1_m = self.decrease_embed(x1)
        x2_m = self.smaller_embed(x2)
        x1_m = x1_m.view(-1, 32, 32)
        x2_m = x2_m.view(-1, 32, 32)
        x_matmul = torch.matmul(x1_m, x2_m).view(-1, 1024)
        x_matmul = F.normalize(x_matmul, p=2, dim=1)
        #Quan
        bsz = x1_m.shape[0]
        x1_q = x1_m.view(bsz, 1024 // 4, 4)
        x2_q = x2_m.view(bsz, 1024 // 4, 4)
        x_quat = quaternion_mul(x1_q, x2_q) 
        x_quat_flat = x_quat.view(bsz, -1)
        x_quat_flat = F.normalize(x_quat_flat, p=2, dim=1)
        

        fusion_features = torch.cat((origin_x1, origin_x2 , x_sum, x_quat_flat, x_matmul, x_hardamard), dim=-1)
        return super().forward(fusion_features)

        
class AllMethodFeatureModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes, normalize=False):
        super(AllMethodFeatureModel, self).__init__(1024 * 4 , num_classes)
        self.normalize = normalize
        self.decrease_embed = nn.Linear(embed_dim1, 1024)
        self.smaller_embed = nn.Linear(embed_dim2, 1024)
            
    def forward(self, x1, x2):
        x1_s = self.decrease_embed(x1)
        x2_s = self.smaller_embed(x2)
        x_sum = x1_s + x2_s
        x_sum = F.normalize(x_sum, p=2, dim=1)
        #hadar
        x_hardamard = x1_s * x2_s
        x_hardamard = F.normalize(x_hardamard, p=2, dim=1)
        #matmul
        x1_m = self.decrease_embed(x1)
        x2_m = self.smaller_embed(x2)
        x1_m = x1_m.view(-1, 32, 32)
        x2_m = x2_m.view(-1, 32, 32)
        x_matmul = torch.matmul(x1_m, x2_m).view(-1, 1024)
        x_matmul = F.normalize(x_matmul, p=2, dim=1)
        #Quan
        bsz = x1_m.shape[0]
        x1_q = x1_m.view(bsz, 1024 // 4, 4)
        x2_q = x2_m.view(bsz, 1024 // 4, 4)
        x_quat = quaternion_mul(x1_q, x2_q) 
        x_quat_flat = x_quat.view(bsz, -1)
        x_quat_flat = F.normalize(x_quat_flat, p=2, dim=1)
        

        fusion_features = torch.cat((x_sum, x_quat_flat, x_matmul, x_hardamard), dim=-1)
        return super().forward(fusion_features)

class TwoEmbeddingMOEModel(BaseModel):
    def __init__(self, embed_dim1, embed_dim2, num_classes):
        super(TwoEmbeddingMOEModel, self).__init__(1024, num_classes)
        self.num_experts = 5

        class MLPExpert(nn.Module):
            def __init__(self, embed_dim1, embed_dim2):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim1 + embed_dim2, 1024),
                    nn.ReLU()
                )
            
            def forward(self, x1, x2):
                fusion_input = torch.cat((x1, x2), dim=1)  # (batch, embed_dim1+embed_dim2)
                return self.mlp(fusion_input)  # (batch, hidden_dim)

        class GateMLP(nn.Module):
            def __init__(self, input_dim, num_experts):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_experts)  # (batch, num_experts) 로짓
                )

            def forward(self, x):
                return self.net(x)  

        self.experts = nn.ModuleList([MLPExpert(embed_dim1, embed_dim2) for _ in range(self.num_experts)])

        self.gate_mlp = GateMLP(embed_dim1 + embed_dim2, self.num_experts)

        self.fusion_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x1, x2):
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)
        gate_input = torch.cat((x1_norm, x2_norm), dim=1)  

        gate_logits = self.gate_mlp(gate_input)
        gate_probs = F.softmax(gate_logits, dim=1)  

        expert_outputs = torch.stack([expert(x1_norm, x2_norm) for expert in self.experts], dim=1)  # (batch, num_experts, hidden_dim)

        fused = torch.einsum("bn,bnd->bd", gate_probs, expert_outputs)  # (batch, hidden_dim)
        # (5) 최종 분류
        return self.fusion_fc(fused)  

        
class ThreeEmbeddingConcatModel(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, embed_dim3, num_classes):
        super(ThreeEmbeddingConcatModel, self).__init__()
        self.fusion_fc = nn.Sequential(
            nn.Linear(embed_dim1 + embed_dim2 + embed_dim3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x1, x2, x3):
        fusion_features = torch.cat((x1, x2, x3), dim=-1)
        output = self.fusion_fc(fusion_features)
        return output

class FourEmbeddingConcatModel(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, embed_dim3, embed_dim4, num_classes):
        super(FourEmbeddingConcatModel, self).__init__()
        self.fusion_fc = nn.Sequential(
            nn.Linear(embed_dim1 + embed_dim2 + embed_dim3 + embed_dim4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x1, x2, x3, x4):
        fusion_features = torch.cat((x1, x2, x3, x4), dim=-1)
        output = self.fusion_fc(fusion_features)
        return output


class FiveEmbeddingConcatModel(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, embed_dim3, embed_dim4, embed_dim5, num_classes):
        super(FiveEmbeddingConcatModel, self).__init__()
        self.fusion_fc = nn.Sequential(
            nn.Linear(embed_dim1 + embed_dim2 + embed_dim3 + embed_dim4 + embed_dim5, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x1, x2, x3, x4, x5):
        fusion_features = torch.cat((x1, x2, x3, x4, x5), dim=-1)
        output = self.fusion_fc(fusion_features)
        return output
    
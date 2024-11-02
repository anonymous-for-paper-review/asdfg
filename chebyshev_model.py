import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import yaml
import numpy as np
from einops import rearrange

# Load data
class LineMODDataset(Dataset):
    # skip as it is same as original code
    pass

# Chebyshev convolution layer
class ChebyshevGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, K):
        super(ChebyshevGraphConvLayer, self).__init__()
        self.K = K
        self.linear = nn.Linear(in_features * (K + 1), out_features)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
    
    def chebyshev_polynomials(self, x, K):
        P = [torch.ones_like(x), x]
        for k in range(2, K + 1):
            Pk = 2 * x * P[-1] - P[-2]
            P.append(Pk)
        return torch.stack(P, dim=-1)
    
    def forward(self, x, adj):
        B, N, _ = adj.shape
        I = torch.eye(N).expand(B, N, N).to(adj.device)
        A_hat = adj + I
        D_hat = torch.diag_embed(torch.pow(torch.sum(A_hat, dim=2), -0.5))
        L_hat = torch.matmul(D_hat, torch.matmul(A_hat, D_hat))

        CP = self.chebyshev_polynomials(L_hat, self.K).to(x.device)
        x = x.view(B, N, -1)
        out = torch.cat([torch.matmul(CP[:, :, :, k], x) for k in range(self.K + 1)], dim=2)
        out = self.linear(out)
        return out

class MultiHeadChebyshevGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, K, heads):
        super(MultiHeadChebyshevGraphConvLayer, self).__init__()
        self.heads = heads
        self.convs = nn.ModuleList([
            ChebyshevGraphConvLayer(in_features, out_features // heads, K) for _ in range(heads)
        ])
        self.linear = nn.Linear(out_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
    
    def forward(self, x, adj):
        out = torch.cat([conv(x, adj) for conv in self.convs], dim=-1)
        out = self.linear(out)
        return out

# GNN model with Chebyshev convolution
class GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, K, heads):
        super(GNN, self).__init__()
        self.lgc1 = MultiHeadChebyshevGraphConvLayer(in_features, hidden_features, K, heads)
        self.lgc2 = MultiHeadChebyshevGraphConvLayer(hidden_features, hidden_features, K, heads)
        self.lgc3 = MultiHeadChebyshevGraphConvLayer(hidden_features, hidden_features, K, heads)
        self.linear = nn.Linear(hidden_features, out_features)

        self.residual1 = nn.Linear(in_features, hidden_features)
        self.residual2 = nn.Linear(hidden_features, hidden_features)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.residual1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.residual2.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

    def forward(self, x, adj, object_ids):
        res1 = self.residual1(x)
        x = F.relu(self.lgc1(x, adj) + res1)

        res2 = self.residual2(x)
        x = F.relu(self.lgc2(x, adj) + res2)

        x = self.lgc3(x, adj)
        x = self.linear(x)
        x = torch.mean(x, dim=1)
        return x

# Image feature extractor
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )

    def forward(self, x, save_path=None, img_id=None, object_id=None):
        x = self.feature_extractor(x)
        return x

# Integrate transformer model with Chebyshev convolution, feature distillation and attention mechanism
class IntegratedModel(nn.Module):
    def __init__(self, image_feature_dim, gnn_feature_dim, transformer_dim, ff_hidden_dim, num_heads, num_transformer_blocks, K, heads, translation_means, translation_stds):
        super(IntegratedModel, self).__init__()
        self.img_feature_extractor = ImageFeatureExtractor()
        self.feature_distiller = nn.Linear(image_feature_dim, gnn_feature_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_dim, num_heads, ff_hidden_dim) for _ in range(num_transformer_blocks)
        ])
        self.gnn = GNN(in_features=gnn_feature_dim, hidden_features=gnn_feature_dim, out_features=9, K=K, heads=heads)
        self.translation_means = translation_means
        self.translation_stds = translation_stds
        self.translation_output_scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.rotation_output_scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)
    
    def forward(self, images, adj, object_ids, img_ids, save_feature_maps=False, save_attention_maps=False):
        img_features = self.img_feature_extractor(images)
        B, C, H, W = img_features.shape
        img_features_flat = img_features.view(B, C, -1).permute(0, 2, 1)
        distilled_features = self.feature_distiller(img_features_flat)
        transformer_input = rearrange(distilled_features, 'b n d -> n b d')
        for i, transformer_block in enumerate(self.transformer_blocks):
            transformer_input = transformer_block(transformer_input)
        transformer_output = rearrange(transformer_input, 'n b d -> b n d')
        gnn_output = self.gnn(transformer_output, adj, object_ids)
        euler_angles_cos_sin = gnn_output[:, :6]
        translation_vector = gnn_output[:, 6:]
        euler_angles = torch.atan2(euler_angles_cos_sin[:, 3:], euler_angles_cos_sin[:, :3])
        rotation_matrix = euler_angles_to_rotation_matrix(euler_angles)
        rotation_matrix = orthogonalize_rotation_matrix(rotation_matrix)
        rotation_matrix = self.rotation_output_scale * rotation_matrix
        translation_vector = self.translation_output_scale * translation_vector
        return rotation_matrix, translation_vector, euler_angles

# Train function
def train_model(root_dir, epochs=10, K=3, heads=4, num_transformer_blocks=4, num_heads=8):
    dataset = LineMODDataset(root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = IntegratedModel(image_feature_dim=256, gnn_feature_dim=256, transformer_dim=256, ff_hidden_dim=1024, 
                            num_heads=num_heads, num_transformer_blocks=num_transformer_blocks, K=K, heads=heads, 
                            translation_means=torch.tensor(dataset.translation_means, dtype=torch.float32, device=device), 
                            translation_stds=torch.tensor(dataset.translation_stds, dtype=torch.float32, device=device)).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            poses = batch['pose'].to(device)
            object_ids = batch['object_id']
            img_ids = batch['img_id']
            
            batch_size = images.size(0)
            img_features = model.img_feature_extractor(images)
            adj = get_adj_matrix(img_features)
            adj = adj.to(device)
            rotation_matrix, translation_vector, predicted_angles = model(images, adj, object_ids, img_ids)
            
            true_angles = torch.atan2(poses[:, 3:6], poses[:, :3])
            true_trans = poses[:, 6:9]
            rotation_loss = euler_angle_loss(predicted_angles, true_angles)
            trans_loss = translation_loss_with_postprocessing(translation_vector, true_trans, 
                                          model.translation_means, 
                                          model.translation_stds)
            
            rotation_loss_weight = 0.2
            translation_loss_weight = 0.8
            loss = rotation_loss_weight * rotation_loss + translation_loss_weight * trans_loss
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        average_loss = epoch_loss / len(dataloader)
        losses.append(average_loss)
        print(f'Epoch {epoch+1}, Loss: {average_loss}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
    
    plt.plot(range(1, epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# Invoke train
train_model('path/to/dataset', epochs=30, K=3, heads=4, num_transformer_blocks=4, num_heads=8)
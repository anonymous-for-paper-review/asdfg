import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms, models
from einops import rearrange

feature_map_dir = 'feature_maps'
attention_map_dir = 'attention_maps'
os.makedirs(feature_map_dir, exist_ok=True)
os.makedirs(attention_map_dir, exist_ok=True)


class LineMODDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.img_ids = []
        self.gt_data = {}
        self.translation_means = None
        self.translation_stds = None

        for object_id in sorted(os.listdir(root_dir)):
            object_path = os.path.join(root_dir, object_id)
            if not os.path.isdir(object_path):
                continue

            mode_file = os.path.join(object_path, f'{mode}.txt')
            with open(mode_file, 'r') as file:
                img_ids = [int(line.strip()) for line in file.readlines()]
                self.img_ids.extend([(object_id, img_id) for img_id in img_ids])
            
            gt_file = os.path.join(object_path, 'gt.yml')
            with open(gt_file, 'r') as file:
                gt_data = yaml.safe_load(file)
                for img_id, data in gt_data.items():
                    self.gt_data[(object_id, int(img_id))] = data

        all_translations = []
        for (object_id, img_id) in self.img_ids:
            cam_t_m2c = self.gt_data[(object_id, img_id)][0]['cam_t_m2c']
            all_translations.append(cam_t_m2c)
        
        all_translations = np.array(all_translations)
        self.translation_means = np.mean(all_translations, axis=0)
        self.translation_stds = np.std(all_translations, axis=0)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((120, 160)),  
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        object_id, img_id = self.img_ids[idx]
        img_path = os.path.join(self.root_dir, object_id, 'rgb', f'{img_id:04d}.png')
        mask_path = os.path.join(self.root_dir, object_id, 'mask', f'{img_id:04d}.png')

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"No image found at {img_path}")
        if mask is None:
            raise FileNotFoundError(f"No mask found at {mask_path}")

        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract object area with mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Crop extracted area
        x, y, w, h = cv2.boundingRect(mask)
        cropped_image = masked_image[y:y+h, x:x+w]

        # Process cropped image
        if cropped_image.size == 0:
            raise ValueError(f"Empty cropped image for {img_path}")
        
        cropped_image = self.transform(cropped_image)

        # Resize mask to match cropped image
        mask = cv2.resize(mask, (w, h))
        mask = torch.tensor(mask, dtype=torch.float32) / 255.0

        if (object_id, img_id) not in self.gt_data:
            raise KeyError(f"Pose data for object id {object_id}, image id {img_id} not found in gt.yml")
        
        cam_R_m2c = self.gt_data[(object_id, img_id)][0]['cam_R_m2c']
        cam_t_m2c = self.gt_data[(object_id, img_id)][0]['cam_t_m2c']

        # Normalize translation vector
        cam_t_m2c = (np.array(cam_t_m2c) - self.translation_means) / self.translation_stds
        
        # Convert rotation matrix to euler angles
        cam_R_m2c = np.array(cam_R_m2c).reshape(3, 3)
        euler_angles = self.rotation_matrix_to_euler_angles(cam_R_m2c)
        
        # Convert euler angles to cos and sin values
        euler_angles_cos_sin = np.hstack([np.cos(euler_angles), np.sin(euler_angles)])

        pose = euler_angles_cos_sin.tolist() + cam_t_m2c.tolist()
        pose_tensor = torch.FloatTensor(pose)

        return {'image': cropped_image, 'mask': mask, 'pose': pose_tensor, 'object_id': object_id, 'img_id': img_id, 'filename': f'{object_id}_{img_id:04d}.png'}

    def rotation_matrix_to_euler_angles(self, R):
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

# Multi-head Legendre convolution layer
class MultiHeadLegendreGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, K, heads):
        super(MultiHeadLegendreGraphConvLayer, self).__init__()
        self.heads = heads
        self.convs = nn.ModuleList([
            LegendreGraphConvLayer(in_features, out_features // heads, K) for _ in range(heads)
        ])
        self.linear = nn.Linear(out_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)  # random value initialization
    
    def forward(self, x, adj):
        out = torch.cat([conv(x, adj) for conv in self.convs], dim=-1)
        out = self.linear(out)
        return out

# Legendre graph convolution layer
class LegendreGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, K):
        super(LegendreGraphConvLayer, self).__init__()
        self.K = K  # order of Legendre polynomial
        self.linear = nn.Linear(in_features * (K + 1), out_features)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)  # random value initialization
    
    def legendre_polynomials(self, x, K):
        P = [torch.ones_like(x), x]
        for k in range(2, K + 1):
            Pk = ((2 * k - 1) * x * P[-1] - (k - 1) * P[-2]) / k
            P.append(Pk)
        return torch.stack(P, dim=-1)
    
    def forward(self, x, adj):
        B, N, _ = adj.shape  # get the shape of the adjacency matrix
        I = torch.eye(N).expand(B, N, N).to(adj.device)
        A_hat = adj + I
        D_hat = torch.diag_embed(torch.pow(torch.sum(A_hat, dim=2), -0.5))
        L_hat = torch.matmul(D_hat, torch.matmul(A_hat, D_hat))

        LP = self.legendre_polynomials(L_hat, self.K).to(x.device)
        x = x.view(B, N, -1)  # adjust the shape of x for matrix multiplication
        out = torch.cat([torch.matmul(LP[:, :, :, k], x) for k in range(self.K + 1)], dim=2)
        out = self.linear(out)  # perform linear transformation
        return out

# Transformer-related modules
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.ff[0].weight, mean=0.0, std=0.01)  # random value initialization
        nn.init.normal_(self.ff[2].weight, mean=0.0, std=0.01)  # random value initialization
    
    def forward(self, x, save_attention=False, attention_save_path=None):
        attn_output, attn_weights = self.attention(x, x, x)
        
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Define GNN model
class GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, K, heads):
        super(GNN, self).__init__()
        self.lgc1 = MultiHeadLegendreGraphConvLayer(in_features, hidden_features, K, heads)
        self.lgc2 = MultiHeadLegendreGraphConvLayer(hidden_features, hidden_features, K, heads)
        self.lgc3 = MultiHeadLegendreGraphConvLayer(hidden_features, hidden_features, K, heads)
        self.linear = nn.Linear(hidden_features, out_features)

        self.residual1 = nn.Linear(in_features, hidden_features)
        self.residual2 = nn.Linear(hidden_features, hidden_features)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.residual1.weight, mean=0.0, std=0.01)  # random value initialization
        nn.init.normal_(self.residual2.weight, mean=0.0, std=0.01)  # random value initialization
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)  # random value initialization

    def forward(self, x, adj, object_ids):
        res1 = self.residual1(x)
        x = F.relu(self.lgc1(x, adj) + res1)

        res2 = self.residual2(x)
        x = F.relu(self.lgc2(x, adj) + res2)

        x = self.lgc3(x, adj)
        x = self.linear(x)        
        x = torch.mean(x, dim=1) # average pooling
        return x

# Extract feature vector from image
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1  # use layer1 as final
        )

    def forward(self, x, save_path=None, img_id=None, object_id=None):
        x = self.feature_extractor(x)
        return x

# Convert euler angles to rotation matrix
def euler_angles_to_rotation_matrix(angles):
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)
    batch_size = angles.size(0)
    theta = angles[:, 0]
    phi = angles[:, 1]
    psi = angles[:, 2]
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_psi = torch.cos(psi)
    sin_psi = torch.sin(psi)
    
    rotation_matrix = torch.zeros((batch_size, 3, 3)).to(angles.device)
    
    rotation_matrix[:, 0, 0] = cos_phi * cos_psi
    rotation_matrix[:, 0, 1] = -cos_phi * sin_psi
    rotation_matrix[:, 0, 2] = sin_phi
    rotation_matrix[:, 1, 0] = sin_theta * sin_phi * cos_psi + cos_theta * sin_psi
    rotation_matrix[:, 1, 1] = -sin_theta * sin_phi * sin_psi + cos_theta * cos_psi
    rotation_matrix[:, 1, 2] = -sin_theta * cos_phi
    rotation_matrix[:, 2, 0] = -cos_theta * sin_phi * cos_psi + sin_theta * sin_psi
    rotation_matrix[:, 2, 1] = cos_theta * sin_phi * sin_psi + sin_theta * cos_psi
    rotation_matrix[:, 2, 2] = cos_theta * cos_phi
    
    return rotation_matrix

# Orthogonalize rotation matrix
def orthogonalize_rotation_matrix(matrix):
    U, _, V = torch.svd(matrix)
    Vt = V.transpose(-2, -1)  # transpose the last two dimensions
    return torch.matmul(U, Vt)

# Integrate transformer model with Legendre convolution, feature distillation and attention mechanism
class IntegratedModel(nn.Module):
    def __init__(self, image_feature_dim, gnn_feature_dim, transformer_dim, ff_hidden_dim, num_heads, num_transformer_blocks, K, heads, translation_means, translation_stds):
        super(IntegratedModel, self).__init__()
        self.img_feature_extractor = ImageFeatureExtractor()  # use the modified ImageFeatureExtractor
        self.feature_distiller = nn.Linear(image_feature_dim, gnn_feature_dim)  # direct linear transformation for distilling features
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_dim, num_heads, ff_hidden_dim) for _ in range(num_transformer_blocks)
        ])
        self.gnn = GNN(in_features=gnn_feature_dim, hidden_features=gnn_feature_dim, out_features=9, K=K, heads=heads)
        self.translation_means = translation_means
        self.translation_stds = translation_stds
        self.translation_output_scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.rotation_output_scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)
    
    def forward(self, images, adj, object_ids, img_ids, save_feature_maps=False, save_attention_maps=False):
        img_features = self.img_feature_extractor(images, save_path=feature_map_dir if save_feature_maps else None, img_id=img_ids[0], object_id=object_ids[0])
        B, C, H, W = img_features.shape
        img_features_flat = img_features.view(B, C, -1).permute(0, 2, 1)  # re-shape
        distilled_features = self.feature_distiller(img_features_flat)
        transformer_input = rearrange(distilled_features, 'b n d -> n b d')
        for i, transformer_block in enumerate(self.transformer_blocks):
            transformer_input = transformer_block(transformer_input, save_attention=save_attention_maps, attention_save_path=attention_map_dir if save_attention_maps else None)
        transformer_output = rearrange(transformer_input, 'n b d -> b n d')
        gnn_output = self.gnn(transformer_output, adj, object_ids)
        euler_angles_cos_sin = gnn_output[:, :6]
        translation_vector = gnn_output[:, 6:]

        # Convert cos and sin values to euler angles
        euler_angles = torch.atan2(euler_angles_cos_sin[:, 3:], euler_angles_cos_sin[:, :3])
        
        rotation_matrix = euler_angles_to_rotation_matrix(euler_angles)
        
        # Orthogonalize
        rotation_matrix = orthogonalize_rotation_matrix(rotation_matrix)
        
        # Scale output
        rotation_matrix = self.rotation_output_scale * rotation_matrix
        translation_vector = self.translation_output_scale * translation_vector

        return rotation_matrix, translation_vector, euler_angles  # 返回欧拉角

# Get euler angle loss
def euler_angle_loss(predicted_angles, true_angles):
    return F.mse_loss(predicted_angles, true_angles)

# Calculate translation loss with post-processing
def translation_loss_with_postprocessing(predicted_trans, true_trans, translation_means, translation_stds):
    true_trans_unstd = true_trans * translation_stds + translation_means
    predicted_trans_unstd = predicted_trans * translation_stds + translation_means
    
    # Sign consistency check
    for i in range(predicted_trans_unstd.size(0)):
        for j in range(3):  # translation vector has three components
            if torch.sign(predicted_trans_unstd[i, j]) != torch.sign(true_trans_unstd[i, j]):
                predicted_trans_unstd[i, j] *= -1  # adjust the sign if different

    return F.mse_loss(predicted_trans_unstd, true_trans_unstd)

# Calculate 4-neighborhood adjacency matrix
def get_adj_matrix(features):
    B, C, H, W = features.size()
    N = H * W
    adj = torch.zeros(B, N, N, device=features.device)

    for b in range(B):
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                if i > 0: adj[b, idx, (i-1)*W + j] = 1  # top
                if i < H-1: adj[b, idx, (i+1)*W + j] = 1  # bottom
                if j > 0: adj[b, idx, i*W + (j-1)] = 1  # left
                if j < W-1: adj[b, idx, i*W + (j+1)] = 1  # right

    return adj

# train loop
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
            
            # Extract image features
            img_features = model.img_feature_extractor(images, save_path=feature_map_dir, img_id=img_ids[0], object_id=object_ids[0])
            
            # Calculate 4-neighborhood adjacency matrix
            adj = get_adj_matrix(img_features)
            adj = adj.to(device)
            
            rotation_matrix, translation_vector, predicted_angles = model(images, adj, object_ids, img_ids, save_feature_maps=True, save_attention_maps=True)
            
            # Separate the true roation and translation
            true_angles = torch.atan2(poses[:, 3:6], poses[:, :3])  # true euler angles are caculated from cos and sin values
            true_trans = poses[:, 6:9]  # true translation vector is in the last 3 values of the pose

            # Calculate and weight losses for rotation and translation
            rotation_loss = euler_angle_loss(predicted_angles, true_angles)
            trans_loss = translation_loss_with_postprocessing(translation_vector, true_trans, 
                                          model.translation_means, 
                                          model.translation_stds)
            
            # Define weight coefficients
            rotation_loss_weight = 0.2
            translation_loss_weight = 0.8
            
            loss = rotation_loss_weight * rotation_loss + translation_loss_weight * trans_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        average_loss = epoch_loss / len(dataloader)
        losses.append(average_loss)
        print(f'Epoch {epoch+1}, Loss: {average_loss}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
    
    plt.figure()
    plt.plot(range(1, epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            poses = batch['pose'].to(device)
            filenames = batch['filename']
            object_ids = batch['object_id']
            img_ids = batch['img_id']
            
            batch_size = images.size(0)
            
            # Extract image features
            img_features = model.img_feature_extractor(images, save_path=feature_map_dir, img_id=img_ids[0], object_id=object_ids[0])
            
            # Calculate 4-neighborhood adjacency matrix
            adj = get_adj_matrix(img_features)
            adj = adj.to(device)
            
            rotation_matrix, translation_vector, predicted_angles = model(images, adj, object_ids, img_ids)
            
            for i in range(batch_size):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                mask = masks[i].cpu().numpy()
                
                # Inverse normalization
                actual_translation = (poses[i, 6:9].cpu().numpy() * dataset.translation_stds) + dataset.translation_means
                predicted_translation = (translation_vector[i].cpu().numpy() * dataset.translation_stds) + dataset.translation_means
                
                # Sign consistency check
                for j in range(3):  # translation vector has three components
                    if np.sign(predicted_translation[j]) != np.sign(actual_translation[j]):
                        predicted_translation[j] *= -1  # adjust the sign if different
                
                actual_rotation = euler_angles_to_rotation_matrix(torch.atan2(poses[i, 3:6], poses[i, :3])).cpu().numpy()
                predicted_rotation = rotation_matrix[i].cpu().numpy()
                
                plt.figure()
                plt.imshow(img)
                plt.imshow(mask, alpha=0.5, cmap='gray')
                plt.title(f'Filename: {filenames[i]}, Predicted: {predicted_rotation} {predicted_translation}\nActual: {actual_rotation} {actual_translation}')
                plt.show()

# Invoke train
train_model('path/to/dataset', epochs=30, K=3, heads=4, num_transformer_blocks=4, num_heads=8)

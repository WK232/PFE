import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
import argparse
import genotypes

parser = argparse.ArgumentParser("segmentation")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset', help='location of the dataset')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='initial channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='auxiliary loss weight')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='architecture genotype')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


# Load your pruned model
from model import Network  
model = Network(args.init_channels, 2, args.layers, args.auxiliary, genotypes.DARTS)
model.load_state_dict(torch.load("/home/kharratw/Documents/tessssst/PFE/DARTS_WITH_DATASET/eval-EXP-20250612-164244/DARTS.pth", map_location='cpu'))
model.eval()

# Set quantization parameters
bit_width = 8
num_clusters = 2 ** bit_width  # = 256
no_of_epochs = 10  # for centroid refinement
batch_size = 64

# Function to quantize a weight tensor using k-means
def kmeans_quantize_tensor(weight_tensor, num_clusters, epochs=10):
    weight_flat = weight_tensor.detach().cpu().numpy().reshape(-1, 1)
    
    # Initial K-Means
    kmeans = KMeans(n_clusters=num_clusters, n_init=1, max_iter=epochs, algorithm='elkan')
    kmeans.fit(weight_flat)
    
    # Replace weights with centroids
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    quantized_tensor = clustered.reshape(weight_tensor.shape)
    
    # Convert back to torch tensor
    return torch.tensor(quantized_tensor, dtype=weight_tensor.dtype)

# Apply K-Means quantization to Conv2D layers only
with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Quantizing {name}...")

            # Get and quantize weight
            weight = module.weight.data
            quantized_weight = kmeans_quantize_tensor(weight, num_clusters, no_of_epochs)
            module.weight.data = quantized_weight
            
# Save the quantized model
torch.save(model.state_dict(), "DARTS_kmeans_quantized.pth")

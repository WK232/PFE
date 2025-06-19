import torch

def summarize_model(path):
    state_dict = torch.load(path, map_location='cpu')
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}, unique_vals={v.unique().numel() if v.numel() < 100000 else 'large tensor'}")

# Original model
print("Original:")
summarize_model("DARTS_WITH_DATASET/eval-EXP-20250612-164244/DARTS.pth")

# Quantized model
print("\nQuantized:")
summarize_model("DARTS_WITH_DATASET/DARTS_kmeans_quantized.pth")

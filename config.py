import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 8
CHECKPOINT_PATH = "checkpoint.pth.tar"

ANCHORS = torch.tensor([
    [(0.0771, 0.0679), (0.0500, 0.0293), (0.0281, 0.0407)],
    [(0.0279, 0.0164), (0.0157, 0.0257), (0.0093, 0.0179)],
    [(0.0150, 0.0105), (0.0064, 0.0100), (0.0036, 0.0057)]
], dtype=torch.float32)
"""MLP with ReLU (§5.1) and Pr(x | m)."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


@torch.no_grad()
def logits(model, X, batch=8192):
    model.eval()
    return torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)])


@torch.no_grad()
def true_prob(model, X, y):
    """Pr(x | m): softmax probability (temperature 1) of the true class, float64 numpy array."""
    z = logits(model, X).double()
    return torch.softmax(z, dim=1).gather(1, y.view(-1, 1)).squeeze(1).cpu().numpy()


@torch.no_grad()
def correct(model, X, y):
    return (logits(model, X).argmax(1) == y).cpu().numpy()

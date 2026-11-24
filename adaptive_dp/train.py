"""train(D, seed) (§5.2): the only training routine, used for every model."""
import torch
import torch.nn.functional as F

from .model import MLP

LR = 1e-3
BATCH = 256
EPOCHS = 30


def train(X, y, seed, epochs=EPOCHS, batch=BATCH, lr=LR):
    """Adam (lr 1e-3), batch 256 with shuffling, cross-entropy, no regularisation.

    The seed fixes the initialisation (torch.manual_seed) and the shuffling generator.
    """
    torch.manual_seed(seed)
    model = MLP(X.shape[1]).to(X.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device=X.device).manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=gen, device=X.device)
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(X[idx]), y[idx]).backward()
            opt.step()
    model.eval()
    return model

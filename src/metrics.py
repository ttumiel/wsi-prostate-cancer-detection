import torch
from sklearn.metrics import cohen_kappa_score

def accuracy(preds, target):
    assert preds.shape == target.shape, f"preds.shape ({preds.shape}) != target.shap ({target.shape})"
    preds,target=preds.detach(),target.detach()
    return torch.mean((preds.long()==target.long()).float())

def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(y_hat.detach().long().cpu().numpy(), y.detach().long().cpu().numpy(), weights='quadratic'))

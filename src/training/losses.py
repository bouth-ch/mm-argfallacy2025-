import torch as th
import torch.nn.functional as F


class FocalLoss(th.nn.Module):
    """
    Multiclass Focal Loss - reduces the weight on easy examples so the model
    pays more attention to hard/rare classes.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None
        ce = F.cross_entropy(inputs, targets, weight=alpha, reduction='none')
        pt = th.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

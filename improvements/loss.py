import numpy as np
import torch
from torch import einsum
try:
    from scipy.ndimage import distance_transform_edt as edt
except Exception:
    from skimage.morphology import distance_transform_edt as edt

from utils import simplex, sset
from losses import CrossEntropy  

# improves region overlap
class SoftDiceLoss:
    """Computes the soft Dice loss, so it measures overlap between prediction and ground truth"""
    def __init__(self, exclude_bg=True, idk=None, smooth=1e-6):
        self.exclude_bg = exclude_bg
        self.idk = idk
        self.smooth = smooth

    def __call__(self, pred_softmax, weak_target):
        # pred_softmax: model probabilities (after softmax)
        # weak_target: one-hot encoded ground truth
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax) and sset(weak_target, [0,1])

        p, g = pred_softmax, weak_target.float()

        if self.idk is not None:
            p, g = p[:, self.idk], g[:, self.idk]

        # optionally drop background
        if self.exclude_bg and p.shape[1] > 1:
            p, g = p[:,1:], g[:,1:]

        # Dice formula = 1 - (2 * intersection) / (sum of squares)
        num = 2.0 * einsum("bkhw,bkhw->k", p, g)           # numerator = 2 * (p*g)
        den = (einsum("bkhw,bkhw->k", p, p) +              # denominator = p^2 + g^2
               einsum("bkhw,bkhw->k", g, g)) + self.smooth
        return 1.0 - (num/den).mean()                      # average over classes → loss value


# improves contour alignment
class BoundaryLoss:
    """Computes loss that focuses on boundaries of organs instead of their whole area"""
    def __init__(self, exclude_bg=True):
        self.exclude_bg = exclude_bg

    @torch.no_grad()
    def _gt_sdf(self, gt_onehot):
        """Compute Signed Distance Map (SDM) for each class in the ground truth"""
        B,K,H,W = gt_onehot.shape
        out = torch.zeros_like(gt_onehot, dtype=torch.float32)
        npgt = gt_onehot.detach().cpu().numpy().astype(np.uint8)
        rng = range(1,K) if self.exclude_bg and K>1 else range(K)
        for b in range(B):
            for k in rng:
                g = npgt[b,k].astype(bool)
                # edt(~g) gives distance to the object; edt(g) inside the object
                # subtracting gives positive outside, negative inside → signed distance map
                sdf = edt(~g) - edt(g) if g.any() else np.zeros((H,W), np.float32)
                out[b,k] = torch.from_numpy(sdf)
        return out

    def __call__(self, pred_softmax, gt_onehot):
        # create SDM for each ground-truth class
        sdfs = self._gt_sdf(gt_onehot).to(pred_softmax.device)

        if self.exclude_bg and pred_softmax.shape[1]>1:
            p, d = pred_softmax[:,1:], sdfs[:,1:]
        else:
            p, d = pred_softmax, sdfs

        return (p*d).mean()

# mixes both with CE
class CEDiceBoundary:
    """Combines three losses: CrossEntropy + λ*Dice + λ*Boundary."""
    def __init__(self, lambda_dice=0.3, lambda_boundary=0.1, idk=None, exclude_bg_in_dice=True):
        self.lambda_dice = float(lambda_dice)                               # how much weight to give to Dice loss
        self.lambda_boundary = float(lambda_boundary)                       # how much weight to give to Boundary loss
        self.ce = CrossEntropy(idk=idk if idk is not None else [])          # base pixel-wise CE loss
        self.dice = SoftDiceLoss(exclude_bg=exclude_bg_in_dice, idk=idk)    # overlap loss
        self.boundary = BoundaryLoss(exclude_bg=True)                       # shape loss

    def __call__(self, pred_softmax, weak_target):
        # compute each part and combine
        return ( self.ce(pred_softmax, weak_target)
               + self.lambda_dice * self.dice(pred_softmax, weak_target)
               + self.lambda_boundary * self.boundary(pred_softmax, weak_target) )


def build_loss(K:int, mode:str="full", lambda_dice:float=0.3, lambda_boundary:float=0.1):
    """What was used by main.py when improvement is enabled."""
    if mode=="full":
        idk = list(range(K))
    elif mode=="partial":
        idk = [0,1,3,4]  
    else:
        raise ValueError(mode)

    return CEDiceBoundary(lambda_dice=lambda_dice, lambda_boundary=lambda_boundary,
                          idk=idk, exclude_bg_in_dice=True)

"""Stage 1 detector: full-resolution U-Net producing a filament probability map."""
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from .config import DETECTOR, FULL, PROB


def normalise(img):
    """2048 px grey image (uint8) -> (3, H, W) float tensor input (the three channels are identical)."""
    x = (img.astype(np.float32) / 255.0 - 0.45) / 0.225
    return torch.from_numpy(np.repeat(x[None], 3, 0))


def build_detector(weights=None, device="cuda"):
    model = getattr(smp, DETECTOR["arch"])(DETECTOR["encoder"], encoder_weights=None, in_channels=DETECTOR["in_channels"], classes=1)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_prob(model, img, device="cuda", tta8=True, amp=True):
    """Probability map (2048 x 2048, float32) with 8-way dihedral TTA averaged in probability space."""
    x = normalise(img)[None].to(device)
    p = 0
    for f in ((False, True) if tta8 else (False,)):
        xf = x.flip(-1) if f else x
        for k in (range(4) if tta8 else (0,)):
            with torch.autocast(device_type=device.split(":")[0], enabled=amp and device.startswith("cuda")):
                q = torch.sigmoid(model(torch.rot90(xf, k, (-2, -1)))).float()
            q = torch.rot90(q, -k, (-2, -1))
            p = p + (q.flip(-1) if f else q)
    n = 8 if tta8 else 1
    return (p[0, 0] / n).cpu().numpy()


def to_small(p2048):
    """Downsample a 2048 px probability map to the 1024 px size used by the later stages (float16 as stored on disk)."""
    return cv2.resize(p2048, (PROB, PROB), interpolation=cv2.INTER_AREA).astype(np.float16)


def ensemble_prob(models, img, device="cuda", tta8=True):
    """Average 1024 px probability map of several detector folds (float32)."""
    acc = None
    for m in models:
        p = to_small(predict_prob(m, img, device, tta8)).astype(np.float32)
        acc = p if acc is None else acc + p
    return acc / len(models)

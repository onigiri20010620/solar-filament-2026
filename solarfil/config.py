"""Fixed settings of the submitted configuration (all values were tuned on 5-fold out-of-fold predictions)."""

FULL = 2048            # image size (px)
PROB = 1024            # size at which probability maps are stored / candidates are extracted (then x2)
CROP = 256             # refiner crop size (px)

# Stage 1 detector: segmentation_models_pytorch arguments
DETECTOR = dict(arch="Unet", encoder="resnet34", in_channels=3)

# Candidate extraction from the fold-averaged probability map (lengths/areas are in 1024 px units; x2 for 2048 px)
CANDIDATES = dict(
    method="join2",
    lo=0.15,           # hysteresis: components of p > lo ...
    hi=0.8,            # ... are kept if they contain a pixel with p > hi
    out=0.6,           # final mask = p > out inside a kept component
    min_area=25,       # minimum area (1024 px units -> 100 px at 2048)
    close_r=2,         # morphological closing radius
    gap=0, angle=35, fill=0,                       # (axis-based joining, disabled)
    gap2=30, theta=30, R=20, fill2=2, min_elong=3.0,  # end-point gap bridging
)

# Stage 2 refiners (averaged); crops are square windows of side 1.3*extent + 48 px resized to CROP
REFINERS = [
    dict(name="refiner_pp", arch="UnetPlusPlus", encoder="resnet34"),
    dict(name="refiner_cx", arch="Unet", encoder="tu-convnext_tiny"),
]
REFINE = dict(thr=0.5, keep_small=0.3, comp="seed")   # threshold; fall back to the candidate if area < 30 %; keep all components touching the candidate

# Instance scorer rule: keep if  h*m >= clf  and  mean probability >= p_mean  and  normalised disk radius <= r_max
SCORER = dict(clf=0.2, p_mean=0.7, r_max=0.95)

# Final erosion of every mask (px); 0.5 = erode at 2x resolution
SHRINK_PX = 0.5

# build_lambda_npz.py
import os, glob, argparse
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

def load_first_array(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    # common keys: 'arr_0', 'image', 'mask', etc.
    for k in ['image', 'images', 'arr_0', 'x', 'data']:
        if k in d and isinstance(d[k], np.ndarray):
            return d[k]
    # fallback: first ndarray-like
    for k in d.files:
        if isinstance(d[k], np.ndarray):
            return d[k]
    raise ValueError(f"No ndarray found in {npz_path}")

def to_uint8_or_float01(img):
    img = np.asarray(img)
    # squeeze channel dims if needed
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    # normalize if needed
    if img.dtype == np.uint8:
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = img.astype(np.float32)
        if img_f.max() > 1.5:  # likely 0..255 float
            img_f = img_f / 255.0
        img_f = np.clip(img_f, 0.0, 1.0)
    return img_f

def binarize_label_from_mask(mask):
    mask = np.asarray(mask)
    return 1 if np.any(mask > 0) else 0

def resize_2d(img, size=128):
    if img.shape == (size, size):
        return img
    if not HAS_CV2:
        # fallback: simple nearest (keeps it dependency-free if cv2 not present)
        zoom_y = size / img.shape[0]
        zoom_x = size / img.shape[1]
        # crude nearest-neighbor resize
        yy = (np.arange(size) / zoom_y).round().astype(int)
        xx = (np.arange(size) / zoom_x).round().astype(int)
        yy = np.clip(yy, 0, img.shape[0]-1)
        xx = np.clip(xx, 0, img.shape[1]-1)
        return img[yy[:, None], xx[None, :]].astype(np.float32)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)

def simulate_poisson_ldct(x01, lam):
    rng = np.random.default_rng(42)
    return np.clip(rng.poisson(lam * x01) / lam, 0.0, 1.0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./dataset", help="Root with nested image.npz/mask.npz")
    ap.add_argument("--out", type=str, default="/notebooks/LDCT/dataset/lambda10.npz")
    ap.add_argument("--lambda_val", type=float, default=10.0)
    ap.add_argument("--resize", type=int, default=128, help="Final H=W size")
    args = ap.parse_args()

    image_files = sorted(glob.glob(os.path.join(args.root, "**", "image.npz"), recursive=True))
    if not image_files:
        raise FileNotFoundError(f"No image.npz found under {args.root}")

    X_list, y_list = [], []
    for im_path in image_files:
        mask_path = os.path.join(os.path.dirname(im_path), "mask.npz")
        if not os.path.exists(mask_path):
            # skip if mask missing
            continue
        try:
            im = load_first_array(im_path)
            ms = load_first_array(mask_path)
        except Exception as e:
            print(f"Skip {im_path}: {e}")
            continue

        # If arrays have extra dims, squeeze to 2D
        im = np.squeeze(im)
        ms = np.squeeze(ms)

        # normalize and optionally resize
        im = to_uint8_or_float01(im)
        ms = (ms > 0).astype(np.uint8)

        if args.resize:
            im = resize_2d(im, size=args.resize)
            # mask resize (nearest)
            if HAS_CV2:
                ms = cv2.resize(ms.astype(np.uint8), (args.resize, args.resize), interpolation=cv2.INTER_NEAREST)
            else:
                ms = resize_2d(ms, size=args.resize)
                ms = (ms > 0.5).astype(np.uint8)

        # label from mask
        y = binarize_label_from_mask(ms)

        # simulate LDCT at lambda
        im_ld = simulate_poisson_ldct(im, args.lambda_val)

        X_list.append(im_ld)
        y_list.append(y)

    if not X_list:
        raise RuntimeError("No valid (image, mask) pairs collected.")

    X = np.stack(X_list, axis=0)  # (N, H, W)
    y = np.asarray(y_list, dtype=np.int64)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, images=X, labels=y)
    print(f"Saved: {args.out} | images {X.shape} | labels {y.shape} | pos={y.sum()} neg={len(y)-y.sum()}")

if __name__ == "__main__":
    main()


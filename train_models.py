# train_models.py
import argparse
import torch, random, numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # set seed before anything else
    seed_everything(args.seed)

    # ---- your existing dataset loading, model, training code ----
    # e.g.,
    # train_loader, val_loader, test_loader = build_dataloaders(...)
    # model = build_model(...)
    # train_and_eval(model, train_loader, val_loader, test_loader)
    # ------------------------------------------------------------

    # Example: call your existing pipeline here
    metrics = train_and_evaluate(
        train_dir="dataset/train",
        val_dir="dataset/val",
        test_dir="dataset/test",
        # other hyperparams...
    )

    print(f"[Seed {args.seed}] Results: {metrics}")

if __name__ == "__main__":
    main()


import argparse
from pathlib import Path
import numpy as np
import torch
from models.backbone import model_dict


def windows(path, length, step):
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1,), dtype=np.float32)
    result = []
    for start in range(0, len(values) - length + 1, step):
        x = values[start:start + length].copy()
        x = (x - x.mean()) / (x.std() + 1e-8)
        result.append(torch.from_numpy(x[None, :]))
    if not result:
        raise ValueError(f"{path} is shorter than length={length}")
    return torch.stack(result)


def main():
    p = argparse.ArgumentParser(description="Predict one paired acoustic/vibration recording")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--acoustic", required=True)
    p.add_argument("--vibration", required=True)
    p.add_argument("--num_cls", type=int, default=7)
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--step", type=int, default=1024)
    p.add_argument("--layer_args", default="100,64,32")
    args = p.parse_args()
    acoustic = windows(args.acoustic, args.length, args.step)
    vibration = windows(args.vibration, args.length, args.step)
    if len(acoustic) != len(vibration):
        raise ValueError("acoustic and vibration files produce different window counts")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_dict["MCSAT_AV"](h_args=[int(x) for x in args.layer_args.split(",")], in_c=1, num_cls=args.num_cls).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(acoustic.to(device), vibration.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
    counts = np.bincount(pred, minlength=args.num_cls)
    print(f"windows={len(pred)}")
    print(f"window_predictions={pred.tolist()}")
    print(f"majority_class={int(counts.argmax())}")
    print(f"class_counts={counts.tolist()}")


if __name__ == "__main__":
    main()

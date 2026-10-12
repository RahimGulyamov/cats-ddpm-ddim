"""Plot raw training CSV, a trailing 100-update mean and fixed-level validation."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "results/training"
    with (folder / "train.csv").open() as handle:
        training = list(csv.DictReader(handle))
    with (folder / "validation.csv").open() as handle:
        validation = list(csv.DictReader(handle))
    steps = np.array([int(row["step"]) for row in training])
    losses = np.array([float(row["train_mse"]) for row in training])
    val_steps = [int(row["step"]) for row in validation]
    val_loss = [float(row["val_noise_mse"]) for row in validation]
    window = min(100, len(training))
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10}):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(steps, losses, color="#738c91", alpha=0.17, linewidth=0.5, label="Train raw / random t")
        axes[0].plot(steps[window - 1:], smoothed, color="#147d84", linewidth=1.8,
                     label=f"Train / trailing mean of {window} updates")
        axes[0].plot(val_steps, val_loss, "o-", color="#c25f35", markersize=4,
                     label="Validation EMA / fixed t")
        best = int(np.argmin(val_loss))
        axes[0].annotate(f"Validation minimum: {val_steps[best]}",
                         xy=(val_steps[best], val_loss[best]), xytext=(3700, 0.18),
                         arrowprops={"arrowstyle": "->", "color": "#c25f35"}, fontsize=9)
        axes[0].set(title="Noise prediction MSE", ylabel="MSE", yscale="log")
        axes[0].legend(fontsize=8, frameon=False, loc="upper right")
        for t in (0, 100, 250, 500, 750, 900, 999):
            axes[1].plot(val_steps, [float(row[f"noise_mse_t{t}"]) for row in validation],
                         marker="o", markersize=3, label=f"t={t}")
        axes[1].set(title="Held-out MSE at each fixed noise level", ylabel="MSE", yscale="log")
        axes[1].legend(fontsize=8, frameon=False, ncol=2)
        for axis in axes:
            axis.set_xlabel("Optimizer updates")
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(folder / "loss.png", dpi=180)
        plt.close(figure)
    print(f"Saved {folder / 'loss.png'} from {len(training)} actual training rows")


if __name__ == "__main__":
    main()

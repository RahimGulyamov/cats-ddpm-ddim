"""CSV, JSONL and TensorBoard logging, with static plots from measured values."""
import csv
import json
import time
from pathlib import Path

from .artifacts import write_csv


class Journal:
    def __init__(self, output, tensorboard, training_history, validation_history):
        self.output = Path(output)
        self.training = training_history
        self.validation = validation_history
        for name, rows in (("train.csv", self.training), ("validation.csv", self.validation)):
            path = self.output / name
            if rows:
                write_csv(path, rows)
            elif path.exists():
                path.unlink()
        self.handle = (self.output / "events.jsonl").open("a", encoding="utf8")
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            # A new event directory avoids showing stale steps beyond a resumed checkpoint.
            self.writer = SummaryWriter(str(self.output / "tensorboard" / f"session-{time.time_ns()}"))

    def event(self, kind, **values):
        payload = {"time_unix": time.time(), "event": kind, **values}
        self.handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.handle.flush()

    def row(self, name, row):
        path = self.output / f"{name}.csv"
        existing = path.exists()
        with path.open("a", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
            if not existing:
                writer.writeheader()
            writer.writerow(row)
        (self.training if name == "train" else self.validation).append(row)

    def scalars(self, namespace, values, step):
        if self.writer:
            for key, value in values.items():
                if isinstance(value, (int, float)) and key != "step":
                    self.writer.add_scalar(f"{namespace}/{key}", value, step)
            self.writer.flush()

    def plots(self):
        self.plot_history(self.output, self.training, self.validation)

    @staticmethod
    def plot_history(output, training, validation):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not training:
            return
        steps = [r["step"] for r in training]
        with plt.rc_context({"figure.facecolor": "#f2eee6", "axes.facecolor": "#f2eee6"}):
            figure, axes = plt.subplots(2, 3, figsize=(15, 8))
            axis = axes[0, 0]
            axis.plot(steps, [r["train_mse"] for r in training], alpha=0.6, color="#247b7b", label="Train noise MSE")
            if validation:
                axis.plot([r["step"] for r in validation], [r["val_noise_mse"] for r in validation],
                          color="#c16b42", marker="o", markersize=4, label="Fixed-level holdout MSE (EMA)")
            axis.set(yscale="log", title="Noise prediction loss")
            axis.legend(fontsize=8)
            axes[0, 1].plot(steps, [r["learning_rate"] for r in training], color="#247b7b")
            axes[0, 1].set(title="Learning rate")
            axes[0, 2].plot(steps, [r["gradient_norm"] for r in training], color="#247b7b")
            axes[0, 2].set(title="Gradient norm before clipping", yscale="log")
            axes[1, 0].plot(steps, [r["images_per_second"] for r in training], color="#247b7b")
            axes[1, 0].set(title="Train images / second (all ranks)")
            axes[1, 1].plot(steps, [r["peak_allocated_mb"] for r in training], label="Allocated")
            axes[1, 1].plot(steps, [r["peak_reserved_mb"] for r in training], label="Reserved")
            axes[1, 1].set(title="Peak GPU memory / max over ranks (MiB)")
            axes[1, 1].legend(fontsize=8)
            if validation:
                keys = [key for key in validation[0] if key.startswith("noise_mse_t")]
                for key in keys:
                    axes[1, 2].plot([r["step"] for r in validation],
                                    [r[key] for r in validation], marker="o", markersize=4, label=key.replace("noise_mse_", ""))
                axes[1, 2].set(yscale="log")
                axes[1, 2].legend(fontsize=7, ncol=2)
            axes[1, 2].set(title="Holdout noise MSE by level")
            for axis in axes.flat:
                axis.set_xlabel("Optimizer updates")
                axis.grid(alpha=0.2)
            figure.tight_layout()
            figure.savefig(Path(output) / "training-dashboard.png", dpi=140)
            plt.close(figure)

    def close(self):
        self.handle.close()
        if self.writer:
            self.writer.close()

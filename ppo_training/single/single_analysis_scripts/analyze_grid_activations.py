"""Analyze PPO actor activations on a normalized distance/angle input grid.

Examples:
    python ppo_training/single/single_analysis_scripts/analyze_grid_activations.py \
        --checkpoint ppo_training/single/results/single_results_eating_reward/checkpoints/brain_update_020.pth

    python ppo_training/single/single_analysis_scripts/analyze_grid_activations.py \
        --checkpoint-dir ppo_training/single/results/single_results_eating_reward/checkpoints \
        --updates 0,50,100,150,200,250

The state vector is assumed to be:
    [food_seen_flag, normalized_distance, normalized_angle, normalized_velocity, normalized_energy]
Only normalized_distance and normalized_angle are swept over [-1, 1].
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppo_training.ppo_brain import PPOBrain


INPUT_DIM = 5
ACTION_DIM = 2
HIDDEN_DIM = 64
ACTION_LABELS = ("Magnitude / Accel output", "Direction / Turn output")
CHECKPOINT_RE = re.compile(r"brain_update_(\d+)\.pth$")


def parse_updates(update_text: str | None) -> set[int] | None:
    if update_text is None:
        return None

    updates: set[int] = set()
    for item in update_text.split(","):
        item = item.strip()
        if not item:
            continue
        updates.add(int(item))
    return updates


def checkpoint_update(checkpoint_path: Path) -> int | None:
    match = CHECKPOINT_RE.search(checkpoint_path.name)
    if match is None:
        return None
    return int(match.group(1))


def sorted_checkpoints(checkpoint_dir: Path, updates: set[int] | None) -> list[Path]:
    checkpoints = sorted(
        checkpoint_dir.glob("brain_update_*.pth"),
        key=lambda path: (
            checkpoint_update(path) is None,
            checkpoint_update(path) if checkpoint_update(path) is not None else path.name,
        ),
    )

    if updates is None:
        return checkpoints

    return [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint_update(checkpoint) in updates
    ]


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    checkpoints = [Path(path).expanduser().resolve() for path in args.checkpoint]
    updates = parse_updates(args.updates)

    checkpoint_dirs: list[Path] = []
    if args.checkpoint_dir is not None:
        checkpoint_dirs.append(Path(args.checkpoint_dir).expanduser().resolve())

    if args.results_folder is not None:
        results_root = Path(args.results_root).expanduser().resolve()
        checkpoint_dirs.append(results_root / args.results_folder / "checkpoints")

    if not checkpoints and not checkpoint_dirs:
        default_dirs = [
            REPO_ROOT / "ppo_training" / "single" / "results" / "single_results_eating_reward" / "checkpoints",
            REPO_ROOT / "eating_reward" / "checkpoints",
        ]
        checkpoint_dirs.extend(path for path in default_dirs if path.exists())

    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        checkpoints.extend(sorted_checkpoints(checkpoint_dir, updates))

    checkpoints = list(dict.fromkeys(checkpoints))
    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint file does not exist: {missing[0]}")

    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints found. Provide --checkpoint, --checkpoint-dir, or --results-folder."
        )

    return checkpoints


def default_output_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "activation_analysis"
    return checkpoint_path.parent / "activation_analysis"


def load_actor(checkpoint_path: Path, device: torch.device) -> PPOBrain:
    model = PPOBrain(INPUT_DIM, ACTION_DIM, HIDDEN_DIM).to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_grid_states(
    grid_res: int,
    food_seen: float,
    norm_vel: float,
    norm_energy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance_values = np.linspace(-1.0, 1.0, grid_res)
    angle_values = np.linspace(-1.0, 1.0, grid_res)
    angle_grid, distance_grid = np.meshgrid(angle_values, distance_values)

    states = np.zeros((grid_res * grid_res, INPUT_DIM), dtype=np.float32)
    states[:, 0] = food_seen
    states[:, 1] = distance_grid.ravel()
    states[:, 2] = angle_grid.ravel()
    states[:, 3] = norm_vel
    states[:, 4] = norm_energy

    return states, distance_grid, angle_grid, distance_values


def capture_actor_activations(
    model: PPOBrain,
    states: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)

    with torch.no_grad():
        hidden1_pre = model.actor[0](state_tensor)
        hidden1 = model.actor[1](hidden1_pre)
        hidden2_pre = model.actor[2](hidden1)
        hidden2 = model.actor[3](hidden2_pre)
        outputs = model.actor[4](hidden2)

    return {
        "states": states,
        "hidden1_pre": hidden1_pre.cpu().numpy(),
        "hidden1": hidden1.cpu().numpy(),
        "hidden2_pre": hidden2_pre.cpu().numpy(),
        "hidden2": hidden2.cpu().numpy(),
        "outputs": outputs.cpu().numpy(),
    }


def pca_scores(activations: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    centered = activations - activations.mean(axis=0, keepdims=True)
    _, singular_values, components_t = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ components_t[:n_components].T

    variances = singular_values**2 / max(len(activations) - 1, 1)
    explained = variances[:n_components] / np.maximum(variances.sum(), np.finfo(float).eps)
    return scores, explained


def plot_hidden_pca(
    checkpoint_label: str,
    hidden_name: str,
    hidden_activations: np.ndarray,
    states: np.ndarray,
    output_dir: Path,
    show: bool,
) -> None:
    scores, explained = pca_scores(hidden_activations)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    color_specs = [
        ("Normalized distance", states[:, 1], "viridis"),
        ("Normalized angle", states[:, 2], "coolwarm"),
    ]

    for ax, (label, values, cmap) in zip(axes, color_specs):
        scatter = ax.scatter(
            scores[:, 0],
            scores[:, 1],
            c=values,
            cmap=cmap,
            s=18,
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(f"{checkpoint_label}: {hidden_name} PCA colored by {label.lower()}")
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.colorbar(scatter, ax=ax, label=label)

    fig.tight_layout()
    output_path = output_dir / f"{checkpoint_label}_{hidden_name}_pca.png"
    fig.savefig(output_path, dpi=180)
    print(f"Saved hidden PCA plot: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_output_maps(
    checkpoint_label: str,
    outputs: np.ndarray,
    distance_grid: np.ndarray,
    angle_grid: np.ndarray,
    output_dir: Path,
    show: bool,
) -> None:
    grid_shape = distance_grid.shape
    accel = outputs[:, 0].reshape(grid_shape)
    turn = outputs[:, 1].reshape(grid_shape)
    action_magnitude = np.linalg.norm(outputs, axis=1).reshape(grid_shape)
    turn_dominance = np.abs(turn) - np.abs(accel)

    maps = [
        (ACTION_LABELS[0], accel, "viridis"),
        (ACTION_LABELS[1], turn, "coolwarm"),
        ("Output vector magnitude", action_magnitude, "magma"),
        ("|Turn| - |Accel| dominance", turn_dominance, "bwr"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, (title, values, cmap) in zip(axes.ravel(), maps):
        contour = ax.contourf(angle_grid, distance_grid, values, levels=30, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("Normalized target angle")
        ax.set_ylabel("Normalized target distance")
        fig.colorbar(contour, ax=ax)

    fig.suptitle(f"{checkpoint_label}: actor output activations over input grid", y=1.01)
    fig.tight_layout()
    output_path = output_dir / f"{checkpoint_label}_output_maps.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved output activation maps: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_output_layer_weights(
    checkpoint_label: str,
    model: PPOBrain,
    output_dir: Path,
    show: bool,
    top_k: int,
) -> None:
    final_weights = model.actor[4].weight.detach().cpu().numpy()
    hidden_indices = np.arange(final_weights.shape[1])
    accel_abs = np.abs(final_weights[0])
    turn_abs = np.abs(final_weights[1])
    preference = turn_abs - accel_abs

    top_k = min(top_k, len(hidden_indices))
    most_accel = np.argsort(accel_abs - turn_abs)[-top_k:][::-1]
    most_turn = np.argsort(turn_abs - accel_abs)[-top_k:][::-1]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    axes[0].bar(hidden_indices - 0.2, final_weights[0], width=0.4, label=ACTION_LABELS[0])
    axes[0].bar(hidden_indices + 0.2, final_weights[1], width=0.4, label=ACTION_LABELS[1])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Final actor layer weights from hidden units to output neurons")
    axes[0].set_xlabel("Hidden unit")
    axes[0].set_ylabel("Weight")
    axes[0].legend()
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(hidden_indices, preference, color=np.where(preference >= 0, "tab:red", "tab:blue"))
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("|Direction weight| - |Magnitude weight| by hidden unit")
    axes[1].set_xlabel("Hidden unit")
    axes[1].set_ylabel("Positive = more direction-linked")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        f"{checkpoint_label}: top magnitude units {most_accel.tolist()} | "
        f"top direction units {most_turn.tolist()}",
        y=1.02,
    )
    fig.tight_layout()
    output_path = output_dir / f"{checkpoint_label}_output_layer_weights.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved output-layer weight plot: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def safe_correlations(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    feature_std = features.std(axis=0, ddof=1)
    target_std = targets.std(axis=0, ddof=1)
    valid_features = feature_std > np.finfo(float).eps
    valid_targets = target_std > np.finfo(float).eps

    correlations = np.zeros((features.shape[1], targets.shape[1]), dtype=np.float64)
    if not np.any(valid_features) or not np.any(valid_targets):
        return correlations

    feature_z = (features[:, valid_features] - features[:, valid_features].mean(axis=0)) / feature_std[valid_features]
    target_z = (targets[:, valid_targets] - targets[:, valid_targets].mean(axis=0)) / target_std[valid_targets]
    correlations[np.ix_(valid_features, valid_targets)] = feature_z.T @ target_z / max(len(features) - 1, 1)
    return correlations


def plot_hidden_output_correlations(
    checkpoint_label: str,
    hidden_name: str,
    hidden_activations: np.ndarray,
    outputs: np.ndarray,
    output_dir: Path,
    show: bool,
    top_k: int,
) -> None:
    correlations = safe_correlations(hidden_activations, outputs)
    top_k = min(top_k, correlations.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for output_idx, ax in enumerate(axes):
        top_units = np.argsort(np.abs(correlations[:, output_idx]))[-top_k:][::-1]
        ax.bar(
            np.arange(top_k),
            correlations[top_units, output_idx],
            color=np.where(correlations[top_units, output_idx] >= 0, "tab:green", "tab:orange"),
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(top_k))
        ax.set_xticklabels(top_units, rotation=45)
        ax.set_title(ACTION_LABELS[output_idx])
        ax.set_xlabel(f"Top {hidden_name} unit")
        ax.set_ylabel("Activation/output correlation")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(f"{checkpoint_label}: hidden units most correlated with output neurons", y=1.02)
    fig.tight_layout()
    output_path = output_dir / f"{checkpoint_label}_{hidden_name}_output_correlations.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved hidden/output correlation plot: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def save_activation_arrays(
    checkpoint_label: str,
    activations: dict[str, np.ndarray],
    distance_grid: np.ndarray,
    angle_grid: np.ndarray,
    output_dir: Path,
) -> None:
    output_path = output_dir / f"{checkpoint_label}_grid_activations.npz"
    np.savez_compressed(
        output_path,
        distance_grid=distance_grid,
        angle_grid=angle_grid,
        **activations,
    )
    print(f"Saved activation arrays: {output_path}")


def analyze_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    checkpoint_label = checkpoint_path.stem
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing checkpoint: {checkpoint_path}")
    model = load_actor(checkpoint_path, device)
    states, distance_grid, angle_grid, _ = make_grid_states(
        grid_res=args.grid_res,
        food_seen=args.food_seen,
        norm_vel=args.norm_vel,
        norm_energy=args.norm_energy,
    )
    activations = capture_actor_activations(model, states, device)
    hidden_layers = ("hidden1", "hidden2") if args.hidden_layer == "both" else (args.hidden_layer,)

    for hidden_layer in hidden_layers:
        hidden_activations = activations[hidden_layer]
        plot_hidden_pca(
            checkpoint_label,
            hidden_layer,
            hidden_activations,
            states,
            output_dir,
            args.show,
        )
        plot_hidden_output_correlations(
            checkpoint_label,
            hidden_layer,
            hidden_activations,
            activations["outputs"],
            output_dir,
            args.show,
            args.top_k,
        )

    plot_output_maps(
        checkpoint_label,
        activations["outputs"],
        distance_grid,
        angle_grid,
        output_dir,
        args.show,
    )
    plot_output_layer_weights(
        checkpoint_label,
        model,
        output_dir,
        args.show,
        args.top_k,
    )

    if args.save_activations:
        save_activation_arrays(checkpoint_label, activations, distance_grid, angle_grid, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load PPOBrain checkpoints and plot actor hidden/output activations for a "
            "normalized distance/angle grid."
        )
    )
    parser.add_argument("--checkpoint", action="append", default=[], help="Path to a .pth checkpoint. Can be repeated.")
    parser.add_argument("--checkpoint-dir", help="Directory containing brain_update_###.pth checkpoints.")
    parser.add_argument("--results-folder", help="Folder below --results-root that contains a checkpoints directory.")
    parser.add_argument(
        "--results-root",
        default=REPO_ROOT / "ppo_training" / "single" / "results",
        help="Root directory used with --results-folder.",
    )
    parser.add_argument("--updates", help="Comma-separated checkpoint updates to analyze, e.g. 0,50,100.")
    parser.add_argument("--output-dir", help="Directory for plots. Defaults beside each checkpoint set.")
    parser.add_argument("--grid-res", type=int, default=50, help="Number of distance and angle values per axis.")
    parser.add_argument(
        "--hidden-layer",
        choices=("hidden1", "hidden2", "both"),
        default="both",
        help="Actor hidden activation layer to use for PCA and output correlations.",
    )
    parser.add_argument("--food-seen", type=float, default=1.0, help="Constant food_seen input value.")
    parser.add_argument("--norm-vel", type=float, default=0.0, help="Constant normalized velocity input value.")
    parser.add_argument("--norm-energy", type=float, default=0.5, help="Constant normalized energy input value.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of units to annotate in output/correlation plots.")
    parser.add_argument("--device", default="cpu", help="Torch device used to load and evaluate the model.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively as well as saving them.")
    parser.add_argument("--save-activations", action="store_true", help="Save grid states and activations to .npz files.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.grid_res < 2:
        parser.error("--grid-res must be at least 2.")

    checkpoint_paths = resolve_checkpoint_paths(args)
    device = torch.device(args.device)
    for checkpoint_path in checkpoint_paths:
        analyze_checkpoint(checkpoint_path, args, device)


if __name__ == "__main__":
    main()

"""Command-line entry points.

    glacierview infer  --glims-id G007026E45991N --checkpoint <path>
    glacierview areas  --checkpoint <path>
    glacierview train  --data-dir <dir> --epochs 10
    glacierview config

Everything that used to be a top-to-bottom script now goes through here, so
there is one place that parses arguments, configures logging and picks a
device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from glacierview import __version__, config
from glacierview.log import configure, get_logger

logger = get_logger(__name__)


def _device(requested: str | None):
    import torch

    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def cmd_config(args) -> int:
    """Print resolved paths — the first thing to check when a path is wrong."""
    label = args.data_label or config.DEFAULT_DATA_LABEL
    paths = {
        "repo_root": config.REPO_ROOT,
        "data_root": config.DATA_ROOT,
        "data_label": label,
        "landing_zone": config.landing_zone(label),
        "landsat_dir": config.landsat_dir(label),
        "processed_metadata": config.processed_metadata_dir(label),
        "glims_data": config.glims_data_dir(),
        "training_data": config.training_data_dir(),
        "analysis_data": config.ANALYSIS_DIR,
        "sql": config.SQL_DIR,
        "outputs": config.outputs_dir(),
    }
    width = max(len(k) for k in paths)
    for key, value in paths.items():
        exists = "" if key in {"data_label"} else (
            "  " if Path(value).exists() else "  (missing)"
        )
        print(f"{key:<{width}}  {value}{exists}")
    return 0


def cmd_infer(args) -> int:
    """Segment one glacier and render a GIF plus an area plot."""
    from glacierview.inference import predict as predict_mod
    from glacierview.inference import render
    from glacierview.models import load_checkpoint

    device = _device(args.device)
    model = load_checkpoint(args.checkpoint, device)

    result = predict_mod.segment_glacier(
        model, args.glims_id, device, args.data_label, variants=("binary",)
    )
    out_dir = Path(args.out_dir) if args.out_dir else config.outputs_dir()

    if not args.no_gif:
        render.write_gif(
            gif_path=out_dir / "gifs" / f"{args.glims_id}.gif",
            scratch_dir=out_dir / "tmp" / "gif_creation",
            images=result["smoothed"],
            probabilities=result["probabilities"],
            file_names=result["file_names"],
        )

    areas = result["areas"]["binary"]
    render.plot_area_series(
        out_dir / "gifs" / f"{args.glims_id}.png",
        result["dates"], areas,
        f"{args.glims_id} estimated surface area",
    )
    print(
        f"{args.glims_id}: {len(areas)} scenes, "
        f"{areas[0]:.2f} -> {areas[-1]:.2f} km^2"
    )
    return 0


def cmd_areas(args) -> int:
    """Segment every glacier in the landing zone."""
    from glacierview.inference import batch
    from glacierview.models import load_checkpoint

    device = _device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    out_dir = Path(args.out_dir) if args.out_dir else config.outputs_dir()

    summary = batch.run(
        model, device, out_dir,
        data_label=args.data_label,
        glims_ids=args.glims_id or None,
        write_gifs=not args.no_gif,
    )
    print(json.dumps(summary, indent=2))
    return 0 if not summary["failures"] else 1


def cmd_train(args) -> int:
    """Train the segmentation model."""
    from glacierview.models import save_state_dict
    from glacierview.training import GlacierDataset, TrainConfig, build_manifest, fit

    data_dir = Path(args.data_dir) if args.data_dir else config.training_data_dir()
    cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        loss=args.loss,
        freeze_encoder=args.freeze_encoder,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    logger.info("training config: %s", cfg.as_dict())

    train_df, holdout = build_manifest(data_dir, test_size=args.test_size, seed=args.seed)
    # Halve the holdout into eval (epoch selection) and test (final score),
    # which is the published 90/5/5 split rather than a bare 90/10.
    half = len(holdout) // 2
    eval_df, test_df = holdout[:half], holdout[half:]
    logger.info("%d train / %d eval / %d test", len(train_df), len(eval_df), len(test_df))

    datasets = (
        GlacierDataset(train_df, data_dir, augment=cfg.augment, normalize=cfg.normalize),
        GlacierDataset(eval_df, data_dir, augment=False, normalize=cfg.normalize),
        GlacierDataset(test_df, data_dir, augment=False, normalize=cfg.normalize),
    )

    out_dir = Path(args.out_dir)
    model, result = fit(*datasets, cfg=cfg, device=_device(args.device), out_dir=out_dir)

    save_state_dict(model, out_dir / "model.state_dict.pt")
    (out_dir / "run.json").write_text(json.dumps({
        "config": cfg.as_dict(),
        "epoch_losses": result.epoch_losses,
        "eval_losses": result.eval_losses,
        "best_epoch": result.best_epoch,
        "dice": result.dice,
        "jaccard": result.jaccard,
    }, indent=2))
    print(f"Dice {result.dice:.4f}  Jaccard {result.jaccard:.4f}  -> {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glacierview", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"glacierview {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--logfile", help="also write logs here")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-label", help=f"default: {config.DEFAULT_DATA_LABEL}")
    common.add_argument("--device", help="cuda, mps or cpu; autodetected by default")
    common.add_argument("--out-dir", help="where to write outputs")

    p = sub.add_parser("config", parents=[common], help="print resolved paths")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("infer", parents=[common], help="segment one glacier")
    p.add_argument("--glims-id", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--no-gif", action="store_true")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("areas", parents=[common], help="segment the whole landing zone")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--glims-id", action="append", help="repeat to limit the run")
    p.add_argument("--no-gif", action="store_true")
    p.set_defaults(func=cmd_areas)

    p = sub.add_parser("train", parents=[common], help="train the model")
    p.add_argument("--data-dir", help="holds images/ and masks/")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32, help="the paper used 32")
    p.add_argument("--loss", default="ce", choices=["ce", "dice", "mse", "combo"])
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.set_defaults(func=cmd_train, out_dir="experiments/latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure(verbose=args.verbose, logfile=args.logfile)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())

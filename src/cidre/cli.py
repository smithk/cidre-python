"""Command-line interface: ``cidre run``, ``cidre fit`` and ``cidre correct``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .io import list_images
from .model import CORRECTION_MODES, CidreModel, correct_files, fit, run


def _add_fit_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model parameters")
    g.add_argument("--lambda-v", type=float, default=None,
                   help="spatial regularisation of v (log10); higher = smoother. Default: 6, more for <200 images")
    g.add_argument("--lambda-z", type=float, default=0.5,
                   help="zero-light regularisation (log10); higher = more uniform z (default: 0.5)")
    g.add_argument("--z-limits", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="known limits of the zero-light level, e.g. from dark frames")
    g.add_argument("--q-percent", type=float, default=0.25,
                   help="fraction of pixels used for the robust mean Q (default: 0.25)")
    g.add_argument("--bit-depth", type=int, default=None, choices=[2**8, 2**12, 2**14, 2**16],
                   help="max integer value of the images; estimated when omitted")
    g.add_argument("--max-iterations", type=int, default=500, help="max L-BFGS iterations (default: 500)")


def _fit_kwargs(a: argparse.Namespace) -> dict:
    return dict(
        lambda_v=a.lambda_v,
        lambda_z=a.lambda_z,
        z_limits=tuple(a.z_limits) if a.z_limits else None,
        q_percent=a.q_percent,
        bit_depth=a.bit_depth,
        max_iterations=a.max_iterations,
    )


def _mode_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--mode", default="zero_light_preserved", choices=list(CORRECTION_MODES),
                   help="correction mode (default: zero_light_preserved)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cidre",
        description="CIDRE illumination correction for optical microscopy. Quote glob patterns, "
        "e.g. cidre run 'plate1/*_w1.tif' -o corrected/",
    )
    parser.add_argument("--version", action="version", version=f"cidre {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="only report warnings and errors")
    parser.add_argument("-j", "--workers", type=int, default=None, help="threads for reading/writing images")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="learn a model from the images and write corrected copies")
    p.add_argument("source", help="folder, glob pattern or single image")
    p.add_argument("-o", "--output", required=True, help="destination folder for corrected images and the model")
    _mode_arg(p)
    _add_fit_args(p)

    p = sub.add_parser("fit", help="learn a model and save it (.npz or .mat)")
    p.add_argument("source", help="folder, glob pattern or single image")
    p.add_argument("-o", "--output", required=True, help="model file to write, e.g. model_w1.npz")
    _add_fit_args(p)

    p = sub.add_parser("correct", help="apply a saved model to images")
    p.add_argument("source", help="folder, glob pattern or single image")
    p.add_argument("--model", required=True,
                   help="model file (.npz/.mat, including MATLAB cidre_model.mat) or a Fiji model folder")
    p.add_argument("-o", "--output", required=True, help="destination folder")
    _mode_arg(p)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="cidre: %(message)s")
    log = logging.getLogger("cidre")
    try:
        if args.command == "run":
            run(args.source, args.output, correction_mode=args.mode, workers=args.workers, **_fit_kwargs(args))
        elif args.command == "fit":
            model = fit(list_images(args.source), workers=args.workers, **_fit_kwargs(args))
            out = model.save(args.output)
            log.info("saved model to %s", out)
        elif args.command == "correct":
            model = CidreModel.load(args.model)
            files = list_images(args.source)
            correct_files(model, files, args.output, args.mode, args.workers)
            log.info("wrote %d corrected images to %s", len(files), Path(args.output))
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

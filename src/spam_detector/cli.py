from __future__ import annotations

import argparse
from pathlib import Path

from spam_detector.config import default_paths
from spam_detector.data import load_dataset
from spam_detector.models import ModelKind, TrainConfig, load_model, predict_label, save_model
from spam_detector.train import train_all_models, train_model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spam_detector")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train one model (train/val/test split) and save it")
    p_train.add_argument("--data", type=str, required=True, help="CSV path (text,label)")
    p_train.add_argument("--kind", type=str, default="logreg", choices=["nb", "logreg", "sgd"])
    p_train.add_argument("--out", type=str, default="", help="Output model path (.joblib). If empty => artifacts/models")
    p_train.add_argument("--sgd-epochs", type=int, default=12)

    p_cmp = sub.add_parser("compare", help="Train NB baseline + LogReg + SGD and compare on TEST")
    p_cmp.add_argument("--data", type=str, required=True, help="CSV path (text,label)")
    p_cmp.add_argument("--sgd-epochs", type=int, default=12)

    p_pred = sub.add_parser("predict", help="Predict one text with a saved model")
    p_pred.add_argument("--model", type=str, required=True, help="Model path (.joblib)")
    p_pred.add_argument("--text", type=str, required=True)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    paths = default_paths()

    if args.cmd == "train":
        ds = load_dataset(Path(args.data))
        cfg = TrainConfig()
        kind: ModelKind = args.kind

        res = train_model(ds.texts, ds.labels, cfg, kind=kind, sgd_epochs=int(args.sgd_epochs))

        out = args.out.strip()
        if not out:
            paths.models_dir.mkdir(parents=True, exist_ok=True)
            out = str(paths.models_dir / f"spam_model_{kind}.joblib")

        save_model(res.model, out)
        print(f"Saved model to: {out}")
        print("VAL:")
        print(f"  accuracy={res.val.accuracy:.4f} f1_macro={res.val.f1_macro:.4f}")
        print("TEST:")
        print(f"  accuracy={res.test.accuracy:.4f} f1_macro={res.test.f1_macro:.4f}")
        return

    if args.cmd == "compare":
        ds = load_dataset(Path(args.data))
        cfg = TrainConfig()
        results = train_all_models(ds.texts, ds.labels, cfg, sgd_epochs=int(args.sgd_epochs))

        paths.models_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            out = paths.models_dir / f"spam_model_{r.kind}.joblib"
            save_model(r.model, str(out))

        print("TEST comparison:")
        for r in results:
            print(f"- {r.kind:5s}  acc={r.test.accuracy:.4f}  f1_macro={r.test.f1_macro:.4f}")
        print(f"Saved models to: {paths.models_dir}")
        return

    if args.cmd == "predict":
        model = load_model(args.model)
        label, proba = predict_label(model, args.text)
        print(f"label={label} spam_proba={proba:.4f}")
        return


if __name__ == "__main__":
    main()
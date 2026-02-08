from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score
from sklearn.pipeline import Pipeline

from spam_detector.config import default_paths
from spam_detector.data import load_dataset
from spam_detector.evaluate import evaluate_model, format_evaluation, save_text_report
from spam_detector.models import ModelKind, TrainConfig, load_model, predict_label, save_model
from spam_detector.plots import save_reports
from spam_detector.train import train_all_models, train_model


def _default_model_path(kind: str) -> Path:
    paths = default_paths()
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    return paths.models_dir / f"spam_model_{kind}.joblib"


@st.cache_resource(show_spinner=False)
def _load_model_cached(model_path_str: str) -> Pipeline:
    return load_model(model_path_str)


def _try_load_model(path: Path) -> Optional[Pipeline]:
    if not path.exists():
        return None
    try:
        return _load_model_cached(str(path))
    except Exception:
        st.error("The model cannot be loaded!")
        return None


@st.cache_data(show_spinner=False)
def _read_uploaded_csv_to_tmp(upload_name: str, bytes_data: bytes) -> Path:
    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"uploaded_{upload_name}"
    tmp_path.write_bytes(bytes_data)
    return tmp_path


def _precision_by_class(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    labels = ["ham", "spam"]
    vals = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {labels[i]: float(vals[i]) for i in range(len(labels))}


def main() -> None:
    st.set_page_config(page_title="E-mail Spam Detector", layout="wide")
    st.title("E-mail Spam Detector")

    paths = default_paths()

    with st.sidebar:
        st.header("Settings")

        st.subheader("Избор на модел")
        model_kind: ModelKind = st.selectbox(
            "Model kind",
            options=["nb", "logreg", "sgd"],
            format_func=lambda k: {"nb": "Baseline: Naive Bayes", "logreg": "TF-IDF + LogisticRegression", "sgd": "SGD (epochs + loss curve)"}[k],
        )

        default_model_path = _default_model_path(model_kind)
        model_path_str = st.text_input(
            "Път до модела (.joblib)",
            value=str(default_model_path),
        )
        model_path = Path(model_path_str)

        st.subheader("Данни за обучение/оценка")
        default_csv = paths.data_raw / "emails.csv"
        data_path_str = st.text_input("Път до CSV (text,label)", value=str(default_csv))
        data_path = Path(data_path_str)

        uploaded = st.file_uploader("или качи CSV", type=["csv"])
        st.divider()

        st.subheader("Параметри (TrainConfig)")
        max_features = st.number_input("max_features", min_value=1000, max_value=200000, value=50000, step=1000)
        ngram_min = st.number_input("ngram min", min_value=1, max_value=3, value=1, step=1)
        ngram_max = st.number_input("ngram max", min_value=1, max_value=3, value=2, step=1)
        c_val = st.number_input("C (LogReg)", min_value=0.01, max_value=20.0, value=2.0, step=0.1)
        max_iter = st.number_input("max_iter", min_value=100, max_value=10000, value=2000, step=100)
        nb_alpha = st.number_input("NB alpha", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
        sgd_epochs = st.number_input("SGD epochs", min_value=3, max_value=100, value=12, step=1)
        random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1)

        cfg = TrainConfig(
            max_features=int(max_features),
            ngram_range=(int(ngram_min), int(ngram_max)),
            c=float(c_val),
            max_iter=int(max_iter),
            nb_alpha=float(nb_alpha),
            random_state=int(random_state),
        )

    resolved_csv_path: Optional[Path]
    if uploaded is not None:
        resolved_csv_path = _read_uploaded_csv_to_tmp(uploaded.name, uploaded.getvalue())
        st.info(f"Използвам качения файл: {uploaded.name}")
    else:
        resolved_csv_path = data_path

    model = _try_load_model(model_path)

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.subheader("1) Класифициране на текст")
        text = st.text_area("Въведи текст (e-mail):", height=180, placeholder="Paste e-mail here...")
        if st.button("Класифицирай", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("Въведи текст.")
            elif model is None:
                st.warning("Няма зареден модел. Обучи модел от секцията вдясно.")
            else:
                label, proba = predict_label(model, text)
                if label == "spam":
                    st.error(f"SPAM (spam_proba={proba:.4f})")
                else:
                    st.success(f"HAM (spam_proba={proba:.4f})")

        st.divider()
        st.subheader("2) Статус на модела")
        if model is None:
            st.warning(f"Няма намерен/зареден модел на: {model_path}")
        else:
            st.success(f"Моделът е зареден: {model_path}")

    with col_right:
        st.subheader("Обучение и оценка (train/val/test)")

        with st.expander("Обучи избрания модел и запази", expanded=True):
            st.write("Обучава върху TRAIN, мери върху VAL, и връща и TEST метрики (test set е отделен!).")
            if st.button("Train selected model", use_container_width=True):
                try:
                    ds = load_dataset(resolved_csv_path)
                    res = train_model(ds.texts, ds.labels, cfg, kind=model_kind, sgd_epochs=int(sgd_epochs))

                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    save_model(res.model, str(model_path))

                    st.cache_resource.clear()
                    st.success(f"Запазих модела в: {model_path}")

                    st.write("**VAL**")
                    st.code(format_evaluation(res.val))

                    st.write("**TEST**")
                    st.code(format_evaluation(res.test))

                    if model_kind == "sgd" and len(res.val_loss_curve) > 0:
                        # loss curve PNG
                        out_dir = paths.reports_dir
                        plot_paths = save_reports(
                            confusion_matrix_2x2=res.val.confusion,
                            precision_by_label={"ham": 0.0, "spam": 0.0},
                            out_dir=out_dir,
                            prefix="sgd_train",
                            loss_curve=res.val_loss_curve,
                        )
                        if plot_paths.loss_curve_png and plot_paths.loss_curve_png.exists():
                            st.image(str(plot_paths.loss_curve_png), caption="Validation loss curve (SGD)")

                except Exception as exc:
                    st.error(f"Грешка при обучение: {exc}")

        with st.expander("Train ALL models + baseline comparison", expanded=False):
            st.write("Тренира **NB (baseline)**, **LogReg**, **SGD** и сравнява TEST метрики.")
            prefix = st.text_input("Prefix за сравнение", value="compare_all")
            if st.button("Train ALL + Compare", use_container_width=True):
                try:
                    ds = load_dataset(resolved_csv_path)
                    results = train_all_models(ds.texts, ds.labels, cfg, sgd_epochs=int(sgd_epochs))

                    out_dir = paths.reports_dir
                    rows = [(r.kind, r.test.accuracy, r.test.f1_macro) for r in results]
                    comp_png = save_reports(
                        confusion_matrix_2x2=results[0].test.confusion,
                        precision_by_label={"ham": 0.0, "spam": 0.0},
                        out_dir=out_dir,
                        prefix=prefix,
                        comparison_rows=rows,
                    ).comparison_png

                    paths = default_paths()
                    paths.models_dir.mkdir(parents=True, exist_ok=True)
                    for r in results:
                        p = paths.models_dir / f"spam_model_{r.kind}.joblib"
                        save_model(r.model, str(p))

                    st.success("Готово! Запазих моделите в artifacts/models и сравнение в artifacts/reports.")
                    st.write("TEST metrics:")
                    st.table(
                        pd.DataFrame(
                            [{"model": r.kind, "test_accuracy": r.test.accuracy, "test_f1_macro": r.test.f1_macro} for r in results]
                        )
                    )
                    if comp_png and comp_png.exists():
                        st.image(str(comp_png), caption="Model comparison (TEST)")

                except Exception as exc:
                    st.error(f"Грешка: {exc}")

        with st.expander("Оцени текущия зареден модел върху CSV + графики", expanded=False):
            st.write("Оценява текущия модел върху избрания CSV и записва отчет + PNG графики в `artifacts/reports`.")
            prefix = st.text_input("Prefix за evaluate", value="eval_single")
            if st.button("Evaluate loaded model", use_container_width=True):
                if model is None:
                    st.warning("Първо обучи или зареди модел.")
                else:
                    try:
                        ds = load_dataset(resolved_csv_path)
                        ev = evaluate_model(model, ds.texts, ds.labels)

                        preds = model.predict(ds.texts)
                        prec = _precision_by_class(ds.labels, list(preds))

                        out_txt = paths.reports_dir / f"{prefix}_metrics.txt"
                        save_text_report(ev, out_txt)

                        plot_paths = save_reports(
                            confusion_matrix_2x2=ev.confusion,
                            precision_by_label=prec,
                            out_dir=paths.reports_dir,
                            prefix=prefix,
                            labels=("ham", "spam"),
                        )

                        st.success("Готово! Записах отчет и графики в artifacts/reports")
                        st.caption(f"Report: {out_txt}")
                        st.code(format_evaluation(ev))

                        if plot_paths.confusion_matrix_png and plot_paths.confusion_matrix_png.exists():
                            st.image(str(plot_paths.confusion_matrix_png), caption="Confusion matrix")
                        if plot_paths.precision_bar_png and plot_paths.precision_bar_png.exists():
                            st.image(str(plot_paths.precision_bar_png), caption="Precision by class")

                    except Exception as exc:
                        st.error(f"Грешка при оценка: {exc}")

        st.divider()
        st.subheader("Бърз преглед на CSV")
        try:
            if resolved_csv_path is not None and resolved_csv_path.exists():
                df = pd.read_csv(resolved_csv_path).head(10)
                st.dataframe(df, use_container_width=True)
        except Exception:
            st.info("Не може да се визуализира CSV preview.")


if __name__ == "__main__":
    main()
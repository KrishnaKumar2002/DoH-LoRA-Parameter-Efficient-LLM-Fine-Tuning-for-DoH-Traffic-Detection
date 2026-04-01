"""Streamlit frontend to test fine-tuned DoH-LoRA models."""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.doh_lora.config import Config
from src.doh_lora.data import read_and_clean, select_numeric_features
from src.doh_lora.evaluation import predict_single_label


@st.cache_resource(show_spinner=False)
def load_model_with_adapter(adapter_dir: str):
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = Config.PADDING_SIDE

    base = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL,
        torch_dtype=Config.DTYPE,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    model.to(Config.DEVICE)
    return model, tokenizer


def predict_label(
    model,
    tokenizer,
    row: pd.Series,
    feature_cols: List[str],
    task_name: str,
    classes: List[str],
) -> Tuple[str, str]:
    pred, score_map = predict_single_label(
        model=model,
        tokenizer=tokenizer,
        row=row,
        feature_cols=feature_cols,
        task_name=task_name,
        classes=classes,
        device=Config.DEVICE,
    )
    details = " | ".join([f"{k}: {v:.4f}" for k, v in score_map.items()])
    return pred, details


def load_samples(csv_path: str, target_col: str, sample_size: int = 20) -> pd.DataFrame:
    df = read_and_clean(csv_path, target_col)
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)
    return df.reset_index(drop=True)


def display_feature_table(row: pd.Series, feature_cols: List[str]) -> None:
    feature_map: Dict[str, str] = {c: row.get(c, "NA") for c in feature_cols}
    st.dataframe(
        pd.DataFrame(
            [{"feature": k, "value": v} for k, v in feature_map.items()]
        ),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="DoH-LoRA Demo", layout="wide")
    st.title("DoH-LoRA Streamlit Testing UI")
    st.caption("Test stage-1 and stage-2 fine-tuned models with sample dropdowns.")

    default_stage1_adapter = str(Config.RESULTS_DIR / "stage1_doh" / "adapter")
    default_stage2_adapter = str(Config.RESULTS_DIR / "stage2_malicious" / "adapter")

    with st.sidebar:
        st.header("Model Paths")
        stage1_adapter = st.text_input("Stage 1 adapter dir", value=default_stage1_adapter)
        stage2_adapter = st.text_input("Stage 2 adapter dir", value=default_stage2_adapter)
        st.divider()
        st.write(f"Device: `{Config.DEVICE}`")
        st.write(f"Base model: `{Config.BASE_MODEL}`")

    tab1, tab2, tab3 = st.tabs(
        ["Stage 1 (DoH)", "Stage 2 (Malicious)", "Complete Two-Stage Flow"]
    )

    with tab1:
        st.subheader("Stage 1: DoH vs Not-DoH")
        try:
            stage1_df = load_samples(Config.FIRST_LAYER_CSV, Config.STAGE1_TARGET_COL, sample_size=20)
            stage1_feature_cols = select_numeric_features(
                stage1_df, Config.STAGE1_TARGET_COL, Config.EXCLUDE_COLS
            )
            model1, tok1 = load_model_with_adapter(stage1_adapter)
        except Exception as exc:
            st.error(f"Stage 1 setup failed: {exc}")
            st.stop()

        options = [
            f"sample_{i:02d} | true={stage1_df.iloc[i][Config.STAGE1_TARGET_COL]}"
            for i in range(len(stage1_df))
        ]
        selected = st.selectbox("Choose sample", options, index=0)
        idx = options.index(selected)
        row = stage1_df.iloc[idx]
        st.write("Selected flow features")
        display_feature_table(row, stage1_feature_cols)

        if st.button("Run Stage 1 Prediction", type="primary"):
            pred, raw = predict_label(
                model1,
                tok1,
                row,
                stage1_feature_cols,
                Config.STAGE1_TASK_NAME,
                Config.STAGE1_CLASSES,
            )
            st.success(f"Predicted: **{pred}**")
            st.write(f"Ground truth: `{row[Config.STAGE1_TARGET_COL]}`")
            with st.expander("Class score details"):
                st.code(raw)

    with tab2:
        st.subheader("Stage 2: Malicious vs Benign")
        try:
            stage2_df = load_samples(Config.SECOND_LAYER_CSV, Config.STAGE2_TARGET_COL, sample_size=20)
            stage2_feature_cols = select_numeric_features(
                stage2_df, Config.STAGE2_TARGET_COL, Config.EXCLUDE_COLS
            )
            model2, tok2 = load_model_with_adapter(stage2_adapter)
        except Exception as exc:
            st.error(f"Stage 2 setup failed: {exc}")
            st.stop()

        options2 = [
            f"sample_{i:02d} | true={stage2_df.iloc[i][Config.STAGE2_TARGET_COL]}"
            for i in range(len(stage2_df))
        ]
        selected2 = st.selectbox("Choose sample ", options2, index=0)
        idx2 = options2.index(selected2)
        row2 = stage2_df.iloc[idx2]
        st.write("Selected flow features")
        display_feature_table(row2, stage2_feature_cols)

        if st.button("Run Stage 2 Prediction", type="primary"):
            pred2, raw2 = predict_label(
                model2,
                tok2,
                row2,
                stage2_feature_cols,
                Config.STAGE2_TASK_NAME,
                Config.STAGE2_CLASSES,
            )
            st.success(f"Predicted: **{pred2}**")
            st.write(f"Ground truth: `{row2[Config.STAGE2_TARGET_COL]}`")
            with st.expander("Class score details"):
                st.code(raw2)

    with tab3:
        st.subheader("Complete Use Case (Stage 1 -> Stage 2)")
        st.write(
            "Select one sample from first-layer data. The app predicts DoH first; if DoH is "
            "predicted, it runs stage-2 model on shared numeric features."
        )
        try:
            stage1_df = load_samples(Config.FIRST_LAYER_CSV, Config.STAGE1_TARGET_COL, sample_size=20)
            stage2_df_full = read_and_clean(Config.SECOND_LAYER_CSV, Config.STAGE2_TARGET_COL)
            f1 = set(select_numeric_features(stage1_df, Config.STAGE1_TARGET_COL, Config.EXCLUDE_COLS))
            f2 = set(select_numeric_features(stage2_df_full, Config.STAGE2_TARGET_COL, Config.EXCLUDE_COLS))
            shared_features = sorted(list(f1.intersection(f2)))
            model1, tok1 = load_model_with_adapter(stage1_adapter)
            model2, tok2 = load_model_with_adapter(stage2_adapter)
        except Exception as exc:
            st.error(f"Two-stage setup failed: {exc}")
            st.stop()

        options3 = [
            f"sample_{i:02d} | true_stage1={stage1_df.iloc[i][Config.STAGE1_TARGET_COL]}"
            for i in range(len(stage1_df))
        ]
        selected3 = st.selectbox("Choose sample  ", options3, index=0)
        idx3 = options3.index(selected3)
        row3 = stage1_df.iloc[idx3]
        st.write("Selected flow features")
        display_feature_table(row3, sorted(list(f1)))

        if st.button("Run Full Flow", type="primary"):
            pred1, raw1 = predict_label(
                model1,
                tok1,
                row3,
                sorted(list(f1)),
                Config.STAGE1_TASK_NAME,
                Config.STAGE1_CLASSES,
            )
            st.info(f"Stage 1 prediction: **{pred1}**")
            with st.expander("Stage 1 class score details"):
                st.code(raw1)

            if pred1 != Config.STAGE1_POSITIVE_LABEL:
                st.warning("Flow predicted as not_doh. Stage 2 skipped.")
            else:
                if not shared_features:
                    st.error("No shared numeric features between stage datasets for stage-2 handoff.")
                else:
                    pred2, raw2 = predict_label(
                        model2,
                        tok2,
                        row3,
                        shared_features,
                        Config.STAGE2_TASK_NAME,
                        Config.STAGE2_CLASSES,
                    )
                    st.success(f"Stage 2 prediction: **{pred2}**")
                    with st.expander("Stage 2 class score details"):
                        st.code(raw2)


if __name__ == "__main__":
    main()

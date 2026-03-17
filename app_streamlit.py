from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "底框结构破坏等级快速评估"


SPI_OPTIONS: List[int] = [6, 7, 8]
ISR_OPTIONS: Dict[int, List[float]] = {6: [1.0, 2.5], 7: [1.0, 2.5], 8: [1.0, 2.0]}
PGA_OPTIONS: Dict[int, List[float]] = {6: [18, 50, 125], 7: [35, 100, 220], 8: [70, 200, 400]}
ECC_RANGES: Dict[str, Tuple[float, float]] = {
    "ecc_x1": (0.0, 0.30),
    "ecc_y1": (0.0, 0.30),
    "ecc_x2": (0.0, 0.13),
    "ecc_y2": (0.0, 0.22),
}


DEFAULT_FEATURE_COLUMNS: List[str] = [
    "设防烈度",
    "刚度比",
    "一层X向偏心率",
    "一层Y向偏心率",
    "二层X向偏心率",
    "二层Y向偏心率",
    "PGA",
]


ROMAN_LEVELS = {1: "Ⅰ级", 2: "Ⅱ级", 3: "Ⅲ级", 4: "Ⅳ级", 5: "Ⅴ级"}


@dataclass(frozen=True)
class ModelPaths:
    rf_joblib: Path
    xgb_json: Path


def _inject_css() -> None:
    st.markdown(
        """
<style>
  :root{
    --bg0:#fbf7f0;
    --bg1:#f6efe4;
    --ink:#1b1b1f;
    --muted:#5a5a66;
    --card:#ffffffcc;
    --stroke:#00000010;
    --accent:#0f766e;
    --accent2:#b45309;
    --danger:#b91c1c;
  }

  .stApp{
    background:
      radial-gradient(1200px 600px at 10% 0%, #e7f8f6 0%, transparent 60%),
      radial-gradient(1000px 600px at 90% 20%, #fff2df 0%, transparent 55%),
      linear-gradient(180deg, var(--bg0), var(--bg1));
    color: var(--ink);
  }

  h1, h2, h3 { letter-spacing: .2px; }
  p, li, label { color: var(--ink); }

  .card{
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 18px 18px 14px 18px;
    box-shadow: 0 10px 24px rgba(0,0,0,.06);
  }

  .kpi{
    display:flex;
    align-items:baseline;
    gap:12px;
    padding: 14px 16px;
    border-radius: 16px;
    border:1px solid var(--stroke);
    background: #fff;
  }
  .kpi .label{ color: var(--muted); font-size: 14px; }
  .kpi .value{ font-size: 28px; font-weight: 700; color: var(--ink); }
  .kpi .tag{
    margin-left:auto;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 999px;
    background: #0f766e14;
    color: var(--accent);
    border:1px solid #0f766e2a;
  }

  .badge{
    display:inline-block;
    font-size: 14px;
    padding: 6px 10px;
    border-radius: 999px;
    border:1px solid var(--stroke);
    background: #ffffff;
    color: var(--muted);
  }
  .badge-strong{
    background: #0f766e10;
    color: var(--accent);
    border-color: #0f766e2a;
    font-weight: 700;
  }
  .badge-warn{
    background: #b4530910;
    color: var(--accent2);
    border-color:#b453092a;
    font-weight: 700;
  }
  .badge-danger{
    background: #b91c1c10;
    color: var(--danger);
    border-color:#b91c1c2a;
    font-weight: 700;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def _normalize_damage_class(pred: int) -> int:
    """
    Model training script maps damage to 0..4 (5 classes). User wants I..V.
    This accepts either 0..4 or 1..5 and returns 1..5.
    """
    try:
        v = int(pred)
    except Exception:
        return 1

    if 0 <= v <= 4:
        return v + 1
    if 1 <= v <= 5:
        return v
    return int(np.clip(v, 1, 5))


def _damage_to_label(level_1to5: int) -> str:
    return ROMAN_LEVELS.get(int(level_1to5), "Ⅰ级")


@st.cache_resource
def _load_rf_model(path_str: str):
    import joblib

    return joblib.load(path_str)


@st.cache_resource
def _load_xgb_booster(path_str: str):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(path_str)
    return booster


def _build_input_df(
    feature_columns: List[str],
    spi: int,
    isr: float,
    ecc_x1: float,
    ecc_y1: float,
    ecc_x2: float,
    ecc_y2: float,
    pga: float,
) -> pd.DataFrame:
    values = [[int(spi), float(isr), float(ecc_x1), float(ecc_y1), float(ecc_x2), float(ecc_y2), float(pga)]]
    df = pd.DataFrame(values, columns=feature_columns)
    return df


def _predict_rf_damage(rf_model, X_df: pd.DataFrame) -> Tuple[int, Optional[np.ndarray]]:
    pred = rf_model.predict(X_df)[0]
    prob = None
    if hasattr(rf_model, "predict_proba"):
        try:
            prob = np.asarray(rf_model.predict_proba(X_df))[0]
        except Exception:
            prob = None
    return _normalize_damage_class(int(pred)), prob


def _predict_xgb_damage(booster, X_df: pd.DataFrame) -> Tuple[int, Optional[np.ndarray]]:
    import xgboost as xgb

    # Training uses enable_categorical=True and treats "设防烈度" as categorical by default.
    # If the column exists, enforce int->category so Streamlit input doesn't break categorical splits.
    for col in X_df.columns:
        if col.strip() == "设防烈度":
            X_df[col] = np.round(X_df[col]).astype(int).astype("category")

    # Avoid feature-name mismatch by aligning to booster.feature_names when present.
    feat_names = getattr(booster, "feature_names", None)
    if feat_names and len(feat_names) == X_df.shape[1] and list(feat_names) != list(X_df.columns):
        X_df = X_df.copy()
        X_df.columns = list(feat_names)

    dmat = xgb.DMatrix(X_df, enable_categorical=True)
    probs = booster.predict(dmat)
    probs = np.asarray(probs)
    if probs.ndim == 1:
        # multi:softprob may be returned as flattened (n_classes,) for a single row.
        if probs.size <= 1:
            pred_idx = int(probs[0]) if probs.size else 0
            return _normalize_damage_class(pred_idx), None
        row = probs
    else:
        row = probs[0]
    pred_idx = int(np.argmax(row))
    return _normalize_damage_class(pred_idx), row


def _kpi(label: str, value: str, tag: str, tag_class: str = "tag") -> None:
    st.markdown(
        f"""
<div class="kpi">
  <div>
    <div class="label">{label}</div>
    <div class="value">{value}</div>
  </div>
  <div class="{tag_class}">{tag}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _badge(text: str, kind: str = "normal") -> str:
    cls = "badge"
    if kind == "strong":
        cls += " badge-strong"
    elif kind == "warn":
        cls += " badge-warn"
    elif kind == "danger":
        cls += " badge-danger"
    return f'<span class="{cls}">{text}</span>'


def _resolve_model_paths() -> ModelPaths:
    base = Path(__file__).resolve().parent
    rf = base / "model_rf_底框层间位移角_fold4.joblib"
    xgb = base / "model_xgb_上部砌体层间位移角_fold4.json"
    return ModelPaths(rf_joblib=rf, xgb_json=xgb)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _inject_css()

    st.markdown(f"<h1 style='margin-bottom:6px'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.caption("输入设防烈度、刚度比、偏心率与 PGA，输出框架层/砌体层破坏等级，并给出底框结构整体破坏等级（取最大值）。")

    paths = _resolve_model_paths()

    with st.sidebar:
        st.markdown("### 模型文件")
        st.write(f"RF：`{paths.rf_joblib.name}`")
        st.write(f"XGBoost：`{paths.xgb_json.name}`")

        with st.expander("高级设置：特征列名（如与模型不一致可修改）", expanded=False):
            cols = []
            for i, c in enumerate(DEFAULT_FEATURE_COLUMNS):
                cols.append(st.text_input(f"第{i+1}列", value=c))
            feature_columns = cols
            st.caption("提示：RF(joblib) 通常按列名匹配；XGBoost(json) 若保存了特征名，会自动对齐。")

    # Use defaults when expander not opened (Streamlit still defines variables; keep explicit).
    feature_columns = locals().get("feature_columns", DEFAULT_FEATURE_COLUMNS)

    col_left, col_right = st.columns([1.3, 1.0], gap="large")

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 输入特征")
        a, b = st.columns(2)
        with a:
            spi = st.selectbox("设防烈度（SPI）", SPI_OPTIONS, index=0)
            pga = st.selectbox("峰值地面加速度（PGA）", PGA_OPTIONS[int(spi)], index=0)
            ecc_y1 = st.number_input(
                f"一层Y向偏心率  [{ECC_RANGES['ecc_y1'][0]:.2f}–{ECC_RANGES['ecc_y1'][1]:.2f}]",
                value=0.0,
                step=0.01,
                format="%.4f",
            )
            ecc_y2 = st.number_input(
                f"二层Y向偏心率  [{ECC_RANGES['ecc_y2'][0]:.2f}–{ECC_RANGES['ecc_y2'][1]:.2f}]",
                value=0.0,
                step=0.01,
                format="%.4f",
            )
        with b:
            isr = st.selectbox("刚度比（ISR）", ISR_OPTIONS[int(spi)], index=0)
            ecc_x1 = st.number_input(
                f"一层X向偏心率  [{ECC_RANGES['ecc_x1'][0]:.2f}–{ECC_RANGES['ecc_x1'][1]:.2f}]",
                value=0.0,
                step=0.01,
                format="%.4f",
            )
            ecc_x2 = st.number_input(
                f"二层X向偏心率  [{ECC_RANGES['ecc_x2'][0]:.2f}–{ECC_RANGES['ecc_x2'][1]:.2f}]",
                value=0.0,
                step=0.01,
                format="%.4f",
            )

        st.markdown("### 规则提示")
        floor1_active = (ecc_x1 > 0) or (ecc_y1 > 0)
        floor2_active = (ecc_x2 > 0) or (ecc_y2 > 0)
        if floor1_active and floor2_active:
            st.error("不允许一层与二层同时存在非零偏心率，请将其中一层偏心率置为 0。")

        out_of_range = (
            not (ECC_RANGES["ecc_x1"][0] <= ecc_x1 <= ECC_RANGES["ecc_x1"][1])
            or not (ECC_RANGES["ecc_y1"][0] <= ecc_y1 <= ECC_RANGES["ecc_y1"][1])
            or not (ECC_RANGES["ecc_x2"][0] <= ecc_x2 <= ECC_RANGES["ecc_x2"][1])
            or not (ECC_RANGES["ecc_y2"][0] <= ecc_y2 <= ECC_RANGES["ecc_y2"][1])
        )
        if out_of_range:
            st.warning("存在偏心率超出建议范围，结果可能外推。")

        do_predict = st.button("开始预测", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 输出结果")
        st.markdown(
            _badge("框架层：RF(joblib)", "strong")
            + " "
            + _badge("砌体层：XGBoost(json)", "strong")
            + " "
            + _badge("整体：取最大值", "warn"),
            unsafe_allow_html=True,
        )
        st.caption("破坏等级按：Ⅰ级 Ⅱ级 Ⅲ级 Ⅳ级 Ⅴ级。")

        if do_predict:
            if floor1_active and floor2_active:
                st.stop()

            if not paths.rf_joblib.exists():
                st.error(f"未找到 RF 模型文件：`{paths.rf_joblib.name}`")
                st.stop()
            if not paths.xgb_json.exists():
                st.error(f"未找到 XGBoost 模型文件：`{paths.xgb_json.name}`")
                st.stop()

            X = _build_input_df(feature_columns, spi, float(isr), ecc_x1, ecc_y1, ecc_x2, ecc_y2, float(pga))

            rf_model = _load_rf_model(str(paths.rf_joblib))
            xgb_booster = _load_xgb_booster(str(paths.xgb_json))

            rf_level, rf_prob = _predict_rf_damage(rf_model, X)
            xgb_level, xgb_prob = _predict_xgb_damage(xgb_booster, X.copy())
            overall_level = max(rf_level, xgb_level)

            _kpi("框架层破坏等级（RF）", _damage_to_label(rf_level), "RF")
            st.write("")
            _kpi("砌体层破坏等级（XGBoost）", _damage_to_label(xgb_level), "XGBoost")
            st.write("")
            _kpi("底框结构破坏等级（max）", _damage_to_label(overall_level), "Overall", tag_class="tag")

            st.markdown("---")
            with st.expander("查看输入与置信度（可选）", expanded=False):
                st.dataframe(X, use_container_width=True, hide_index=True)
                if rf_prob is not None:
                    st.write("RF 各等级概率（按模型内部类别顺序）：")
                    st.write(np.round(rf_prob, 4))
                if xgb_prob is not None:
                    st.write("XGBoost 各等级概率（按 0..4）：")
                    st.write(np.round(xgb_prob, 4))
        else:
            st.info("在左侧输入特征后，点击“开始预测”。")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

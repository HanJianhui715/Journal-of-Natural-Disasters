from __future__ import annotations

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


ROMAN_LEVELS = {1: "Ⅰ", 2: "Ⅱ", 3: "Ⅲ", 4: "Ⅳ", 5: "Ⅴ"}
LEVEL_TEXT = {
    1: "基本完好",
    2: "轻微破坏",
    3: "中等破坏",
    4: "严重破坏",
    5: "倒塌",
}


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

  .scale{
    display:grid;
    grid-template-columns: repeat(5, 1fr);
    gap:10px;
    margin-top: 6px;
  }
  .scale .pill{
    background:#fff;
    border:1px solid var(--stroke);
    border-radius: 16px;
    padding: 10px 10px;
    text-align:center;
  }
  .scale .pill .r{ font-weight: 800; font-size: 18px; }
  .scale .pill .t{ color: var(--muted); font-size: 12px; margin-top: 4px; }
  .hint{ color: var(--muted); font-size: 12px; }
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
    lvl = int(level_1to5)
    return f"{ROMAN_LEVELS.get(lvl, 'Ⅰ')}（{LEVEL_TEXT.get(lvl, '基本完好')}）"


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


def _rf_expected_feature_names(rf_model) -> Optional[List[str]]:
    names = getattr(rf_model, "feature_names_in_", None)
    if names is not None:
        try:
            out = [str(x) for x in list(names)]
            return out or None
        except Exception:
            pass

    steps = getattr(rf_model, "named_steps", {}) or {}
    prep = steps.get("prep")
    if prep is not None and hasattr(prep, "transformers_"):
        cols: List[str] = []
        try:
            for _, _, c in prep.transformers_:
                if isinstance(c, (list, tuple, np.ndarray, pd.Index)):
                    cols.extend([str(x) for x in list(c)])
            cols = [c for c in cols if c]
            if cols:
                # de-duplicate while keeping order
                seen = set()
                dedup = []
                for c in cols:
                    if c in seen:
                        continue
                    seen.add(c)
                    dedup.append(c)
                return dedup
        except Exception:
            return None
    return None


def _canonical_values(
    spi: int,
    isr: float,
    ecc_x1: float,
    ecc_y1: float,
    ecc_x2: float,
    ecc_y2: float,
    pga: float,
) -> List[float]:
    return [int(spi), float(isr), float(ecc_x1), float(ecc_y1), float(ecc_x2), float(ecc_y2), float(pga)]


def _semantic_index(name: str) -> Optional[int]:
    s = str(name).strip()
    up = s.upper()
    low = s.lower()

    # 0: SPI, 1: ISR, 2: ecc_x1, 3: ecc_y1, 4: ecc_x2, 5: ecc_y2, 6: PGA
    if "PGA" in up:
        return 6
    if ("设防" in s) or (low in {"spi", "intensity"}):
        return 0
    if ("刚度" in s) or (low in {"isr", "stiffness_ratio"}):
        return 1
    if ("一层" in s) and ("X" in up):
        return 2
    if ("一层" in s) and ("Y" in up):
        return 3
    if ("二层" in s) and ("X" in up):
        return 4
    if ("二层" in s) and ("Y" in up):
        return 5
    return None


def _make_df_for_model(expected_names: Optional[List[str]], values: List[float]) -> pd.DataFrame:
    if expected_names and len(expected_names) == len(values):
        mapping: Dict[int, int] = {}
        used = set()
        for pos, col in enumerate(expected_names):
            idx = _semantic_index(str(col))
            if idx is None or idx in used:
                continue
            mapping[pos] = idx
            used.add(idx)

        # If we can confidently match most columns, assign by semantics; otherwise keep canonical order.
        row = [None] * len(values)
        if len(mapping) >= 4:
            for pos in range(len(expected_names)):
                if pos in mapping:
                    row[pos] = values[mapping[pos]]
            remaining_vals = [values[i] for i in range(len(values)) if i not in used]
            j = 0
            for pos in range(len(row)):
                if row[pos] is None:
                    row[pos] = remaining_vals[j]
                    j += 1
        else:
            row = list(values)

        return pd.DataFrame([row], columns=list(expected_names))
    # Fallback: use default names (7 features) in canonical order.
    return pd.DataFrame([values], columns=DEFAULT_FEATURE_COLUMNS)


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

    # Keep DataFrame to preserve categorical support; align columns if model has names.
    feat_names = getattr(booster, "feature_names", None)
    if feat_names and len(feat_names) == X_df.shape[1]:
        X_df = X_df.copy()
        X_df.columns = list(feat_names)

    # Make the intensity feature categorical (best-effort: match by name, otherwise first column).
    intensity_col = None
    for col in X_df.columns:
        if "设防" in str(col) or str(col).strip().lower() in {"spi", "intensity"}:
            intensity_col = col
            break
    if intensity_col is None and len(X_df.columns) > 0:
        intensity_col = X_df.columns[0]
    if intensity_col is not None:
        try:
            X_df[intensity_col] = np.round(X_df[intensity_col]).astype(int).astype("category")
        except Exception:
            pass

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
    tag_html = f'<div class="{tag_class}">{tag}</div>' if str(tag).strip() else ""
    st.markdown(
        f"""
<div class="kpi">
  <div>
    <div class="label">{label}</div>
    <div class="value">{value}</div>
  </div>
  {tag_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def _resolve_model_paths() -> Tuple[Path, Path]:
    base = Path(__file__).resolve().parent
    rf = base / "model_rf_底框层间位移角_fold4.joblib"
    xgb = base / "model_xgb_上部砌体层间位移角_fold4.json"
    return rf, xgb


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _inject_css()

    st.markdown(f"<h1 style='margin-bottom:6px'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.caption("输入地震与结构参数，输出各层与整体破坏等级。")
    rf_path, xgb_path = _resolve_model_paths()

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

        do_predict = st.button("开始预测", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 输出结果")
        st.markdown(
            """
<div class="scale">
  <div class="pill"><div class="r">Ⅰ</div><div class="t">基本完好</div></div>
  <div class="pill"><div class="r">Ⅱ</div><div class="t">轻微破坏</div></div>
  <div class="pill"><div class="r">Ⅲ</div><div class="t">中等破坏</div></div>
  <div class="pill"><div class="r">Ⅳ</div><div class="t">严重破坏</div></div>
  <div class="pill"><div class="r">Ⅴ</div><div class="t">倒塌</div></div>
</div>
            """,
            unsafe_allow_html=True,
        )

        if do_predict:
            if not rf_path.exists():
                st.error("模型文件缺失，无法完成预测。")
                st.stop()
            if not xgb_path.exists():
                st.error("模型文件缺失，无法完成预测。")
                st.stop()

            values = _canonical_values(spi, float(isr), ecc_x1, ecc_y1, ecc_x2, ecc_y2, float(pga))

            rf_model = _load_rf_model(str(rf_path))
            xgb_booster = _load_xgb_booster(str(xgb_path))

            rf_expected = _rf_expected_feature_names(rf_model)
            X_rf = _make_df_for_model(rf_expected, values)
            X_xgb = _make_df_for_model(getattr(xgb_booster, "feature_names", None), values)

            try:
                rf_level, _ = _predict_rf_damage(rf_model, X_rf)
            except Exception as exc:
                st.error("预测失败，请确认输入特征与模型训练时一致。")
                with st.expander("技术细节", expanded=False):
                    st.write(str(exc))
                    st.write("RF期望列名：", rf_expected)
                    st.write("当前列名：", list(X_rf.columns))
                st.stop()

            try:
                xgb_level, _ = _predict_xgb_damage(xgb_booster, X_xgb.copy())
            except Exception as exc:
                st.error("预测失败，请确认输入特征与模型训练时一致。")
                with st.expander("技术细节", expanded=False):
                    st.write(str(exc))
                    st.write("XGBoost列名：", getattr(xgb_booster, "feature_names", None))
                    st.write("当前列名：", list(X_xgb.columns))
                st.stop()

            overall_level = max(rf_level, xgb_level)

            _kpi("框架层破坏等级", _damage_to_label(rf_level), "")
            st.write("")
            _kpi("砌体层破坏等级", _damage_to_label(xgb_level), "")
            st.write("")
            _kpi("底框结构破坏等级", _damage_to_label(overall_level), "")

            st.markdown("---")
            X_show = _build_input_df(DEFAULT_FEATURE_COLUMNS, spi, float(isr), ecc_x1, ecc_y1, ecc_x2, ecc_y2, float(pga))
            st.markdown("<div class='hint'>输入参数</div>", unsafe_allow_html=True)
            st.dataframe(X_show, use_container_width=True, hide_index=True)
        else:
            st.info("输入参数后，点击“开始预测”。")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

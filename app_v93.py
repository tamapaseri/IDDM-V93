# -*- coding: utf-8 -*-
"""
IDDM V83（3モデル統合）× V93安定化ロジック 3クラス分類 Streamlitアプリ
（Streamlit Community Cloud 対応・複数枚静止画推論・スマホ向けリセットボタン付き）
"""

from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# パス・定数設定
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_SPECS = [
    {
        "key": "norm",
        "class_id": 0,
        "label": "正常範囲 (Normal: 70-179 mg/dL)",
        "candidates": [
            BASE_DIR / "results_v83_class" / "model_v83_norm.pth",
            BASE_DIR / "model_v83_norm.pth",
        ],
    },
    {
        "key": "hyper",
        "class_id": 1,
        "label": "高血糖 (Hyperglycemia: >=180 mg/dL)",
        "candidates": [
            BASE_DIR / "results_v83_class" / "model_v83_hyper.pth",
            BASE_DIR / "model_v83_hyper.pth",
        ],
    },
    {
        "key": "hypo",
        "class_id": 2,
        "label": "低血糖 (Hypoglycemia: <70 mg/dL)",
        "candidates": [
            BASE_DIR / "results_v83_class" / "model_v83_hypo.pth",
            BASE_DIR / "model_v83_hypo.pth",
        ],
    },
]

IMAGE_SIZE = 224
MODEL_NAME = "mobilenetv3_small_100"
DEVICE = torch.device("cpu")

# V93 安定化パラメータ
CONFIDENCE_MARGIN_THRESHOLD = 0.25
HISTORY_WINDOW_SIZE = 5


# ---------------------------------------------------------------------------
# モデル定義・読み込み
# ---------------------------------------------------------------------------
@st.cache_resource
def load_face_cascade():
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if Path(cascade_path).exists():
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                return cascade
    except Exception:
        pass
    return None


FACE_CASCADE = load_face_cascade()


class BinaryClassifier(nn.Module):
    """V83 2値分類モデル（該当クラス:1 vs その他:0）"""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def resolve_candidate_path(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


@st.cache_resource
def load_v83_models():
    temp_backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    with torch.no_grad():
        feat_dim = temp_backbone(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)).shape[1]

    models = {}
    resolved_paths = {}
    errors = []

    for spec in MODEL_SPECS:
        key = spec["key"]
        target_path = resolve_candidate_path(spec["candidates"])
        resolved_paths[key] = target_path

        model = BinaryClassifier(feat_dim).to(DEVICE)
        if not target_path.exists():
            cand_str = ", ".join(str(p.relative_to(BASE_DIR)) for p in spec["candidates"])
            errors.append(f"[{key}] モデルが見つかりません（候補: {cand_str}）")
            models[key] = None
            continue

        try:
            checkpoint = torch.load(target_path, map_location=DEVICE)
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            models[key] = model
        except Exception as e:
            errors.append(f"[{key}] 読み込み失敗 ({target_path.name}): {e}")
            models[key] = None

    all_loaded = all(models[s["key"]] is not None for s in MODEL_SPECS)
    return models, resolved_paths, all_loaded, errors


def get_3class_label_text(class_id: int) -> str:
    for spec in MODEL_SPECS:
        if spec["class_id"] == class_id:
            return spec["label"]
    return "不明"


# ---------------------------------------------------------------------------
# 画像前処理（V57 / V93 相当）
# ---------------------------------------------------------------------------
def robust_lighting_normalization(img_bgr: np.ndarray) -> np.ndarray:
    try:
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(y), cr, cb]), cv2.COLOR_YCrCb2BGR)
    except Exception:
        return img_bgr


def process_image_v57(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]

    if FACE_CASCADE is not None and not FACE_CASCADE.empty():
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(int(w * 0.25), int(h * 0.25)),
            )
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                if (fw >= w * 0.2) and (fh >= h * 0.2):
                    cx, cy = x + fw / 2.0, y + fh / 2.0
                    box_size = int(max(fw, fh) * 2.0)
                    x1, y1 = int(cx - box_size / 2.0), int(cy - box_size / 2.0)
                    cropped = img_bgr[
                        max(0, y1) : min(h, y1 + box_size),
                        max(0, x1) : min(w, x1 + box_size),
                    ]
                    if cropped.size > 0:
                        return robust_lighting_normalization(cropped), "Face_ROI_Zoomed"
        except Exception:
            pass

    min_dim = min(h, w)
    cx, cy = w / 2.0, h / 2.0
    box_size = int(min_dim * 0.95)
    x1, y1 = int(cx - box_size / 2.0), int(cy - box_size / 2.0)
    cropped = img_bgr[
        max(0, y1) : min(h, y1 + box_size),
        max(0, x1) : min(w, x1 + box_size),
    ]
    if cropped.size == 0:
        cropped = img_bgr
    return robust_lighting_normalization(cropped), "Fallback_Zoomed"


def to_model_tensor(processed_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(
        img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float() / 255.0
    return tensor.unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# 3モデル統合推論＆V93安定化
# ---------------------------------------------------------------------------
def predict_3class_from_v83(models: dict, tensor: torch.Tensor) -> np.ndarray:
    """
    各2値モデルの「該当クラス(ID=1)である確率」を抽出し、3クラス合計が1.0になるよう正規化。
    出力順: [正常(0), 高血糖(1), 低血糖(2)]
    """
    raw_pos_probs = []
    for spec in MODEL_SPECS:
        m = models[spec["key"]]
        if m is None:
            raw_pos_probs.append(0.0)
            continue
        logits = m(tensor)
        bin_probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        raw_pos_probs.append(float(bin_probs[1]))

    vec = np.array(raw_pos_probs, dtype=np.float64)
    total = float(vec.sum())
    if total > 1e-8:
        return (vec / total).astype(np.float32)
    return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)


def stabilize_v93(
    raw_id: int,
    probs: np.ndarray,
    last_stable_id,
    recent_history: list,
    margin_threshold: float = CONFIDENCE_MARGIN_THRESHOLD,
    window_size: int = HISTORY_WINDOW_SIZE,
):
    stable_id = raw_id
    if last_stable_id is not None and raw_id != last_stable_id:
        sorted_probs = np.sort(probs)
        conf_margin = float(sorted_probs[-1] - sorted_probs[-2])
        if conf_margin < margin_threshold:
            stable_id = last_stable_id

    updated_history = list(recent_history)
    updated_history.append(stable_id)
    if len(updated_history) > window_size:
        updated_history.pop(0)

    final_id = int(np.bincount(updated_history).argmax())
    final_conf = float(probs[final_id])
    return final_id, final_conf, updated_history


def run_inference_on_bytes(
    models: dict,
    file_bytes: bytes,
    use_stabilization: bool,
    last_stable_id,
    recent_history: list,
):
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return (
            {"error": "画像のデコードに失敗しました。"},
            last_stable_id,
            recent_history,
        )

    processed_bgr, mode_str = process_image_v57(img_bgr)
    tensor = to_model_tensor(processed_bgr)
    probs = predict_3class_from_v83(models, tensor)
    raw_id = int(np.argmax(probs))

    next_last_id = last_stable_id
    next_history = recent_history

    if use_stabilization:
        final_id, final_conf, next_history = stabilize_v93(
            raw_id=raw_id,
            probs=probs,
            last_stable_id=last_stable_id,
            recent_history=recent_history,
        )
        next_last_id = final_id
    else:
        final_id = raw_id
        final_conf = float(probs[final_id])

    return (
        {
            "error": None,
            "display_rgb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            "processed_rgb": cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB),
            "mode_str": mode_str,
            "probs": probs,
            "raw_id": raw_id,
            "final_id": final_id,
            "final_conf": final_conf,
        },
        next_last_id,
        next_history,
    )


# ---------------------------------------------------------------------------
# UI描画
# ---------------------------------------------------------------------------
def init_session():
    if "last_stable_id" not in st.session_state:
        st.session_state.last_stable_id = None
    if "recent_history" not in st.session_state:
        st.session_state.recent_history = []


def reset_stabilization_history():
    st.session_state.last_stable_id = None
    st.session_state.recent_history = []


def render_summary(results: list):
    st.markdown("---")
    st.subheader("📋 一括判定サマリー（V83 3モデル統合・3クラス）")

    rows = []
    for r in results:
        if r["error"]:
            rows.append({
                "No.": r["idx"],
                "ファイル名": r["name"],
                "最終判定": "エラー",
                "確信度": "-",
                "正常 (0)": "-",
                "高血糖 (1)": "-",
                "低血糖 (2)": "-",
                "ROIモード": "-",
            })
        else:
            p = r["probs"]
            rows.append({
                "No.": r["idx"],
                "ファイル名": r["name"],
                "最終判定": get_3class_label_text(r["final_id"]),
                "確信度": f"{r['final_conf'] * 100:.1f}%",
                "正常 (0)": f"{p[0] * 100:.1f}%",
                "高血糖 (1)": f"{p[1] * 100:.1f}%",
                "低血糖 (2)": f"{p[2] * 100:.1f}%",
                "ROIモード": r["mode_str"],
            })

    st.table(rows)


def render_details(results: list, show_processed: bool):
    st.markdown("---")
    st.subheader("🖼️ 各画像の詳細結果")

    for r in results:
        with st.expander(f"[{r['idx']}/{len(results)}] {r['name']}", expanded=(len(results) <= 3)):
            if r["error"]:
                st.error(r["error"])
                continue

            if show_processed:
                c1, c2 = st.columns(2)
                c1.image(r["display_rgb"], caption="元画像", use_container_width=True)
                c2.image(
                    r["processed_rgb"],
                    caption=f"前処理後 ({r['mode_str']})",
                    use_container_width=True,
                )
            else:
                st.image(r["display_rgb"], caption=r["name"], use_container_width=True)

            label_text = get_3class_label_text(r["final_id"])
            if r["final_id"] == 0:
                st.success(f"**{label_text}**（確信度: {r['final_conf'] * 100:.1f}%）")
            elif r["final_id"] == 1:
                st.warning(f"**{label_text}**（確信度: {r['final_conf'] * 100:.1f}%）")
            else:
                st.error(f"**{label_text}**（確信度: {r['final_conf'] * 100:.1f}%）")

            st.caption(
                f"ROIモード: {r['mode_str']} / 瞬時ID: {r['raw_id']} → 安定化後ID: {r['final_id']}"
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("正常範囲 (0)", f"{r['probs'][0] * 100:.1f}%")
            m2.metric("高血糖 (1)", f"{r['probs'][1] * 100:.1f}%")
            m3.metric("低血糖 (2)", f"{r['probs'][2] * 100:.1f}%")


def main():
    st.set_page_config(
        page_title="IDDM V83×V93 3クラス血糖値状態判定",
        page_icon="🩺",
        layout="centered",
    )

    init_session()
    models, resolved_paths, all_loaded, errors = load_v83_models()

    st.title("🩺 IDDM 3クラス血糖値状態判定（V83 3モデル統合）")
    st.markdown(
        "リポジトリ内のV83 3モデル（正常・高血糖・低血糖）を統合し、"
        "V93の安定化ロジック（信頼度差 `< 0.25` で直前維持 ＋ 直近5件多数決）で複数枚の静止画を判定します[cite: 8]。"
    )

    # メイン画面上部のコントロール（iPhoneで操作しやすい位置）
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        use_stabilization = st.checkbox("V93 状態安定化を適用", value=True, key="main_use_stab")
    with ctrl_col2:
        if st.button("🔄 安定化履歴をリセット", use_container_width=True, key="main_reset_btn"):
            reset_stabilization_history()
            st.rerun()

    st.caption(
        f"現在の履歴: `{st.session_state.recent_history}` / "
        f"直前確定ID: `{st.session_state.last_stable_id}`"
    )

    with st.sidebar:
        st.header("推論・表示オプション")
        show_processed = st.checkbox("ROIズーム＋CLAHE前処理画像も表示", value=False)
        if st.button("🔄 安定化履歴をリセット", use_container_width=True, key="side_reset_btn"):
            reset_stabilization_history()
            st.rerun()

        st.markdown("---")
        st.caption("V83 モデル配置パス:")
        for key, p in resolved_paths.items():
            st.caption(f"- `{key}`: `{p.relative_to(BASE_DIR)}`")
        st.caption(f"現在の履歴: `{st.session_state.recent_history}`")

    if not all_loaded:
        st.error("一部のV83モデルを読み込めませんでした:\n" + "\n".join(errors))

    uploaded_files = st.file_uploader(
        "静止画を選択（複数選択可: JPG / JPEG / PNG / BMP / WEBP）",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"選択枚数: {len(uploaded_files)} 枚")

        cur_last_id = st.session_state.last_stable_id
        cur_history = list(st.session_state.recent_history)
        results = []

        with st.spinner("V83 3モデルで推論中..."):
            with torch.inference_mode():
                for idx, up_file in enumerate(uploaded_files, 1):
                    res, cur_last_id, cur_history = run_inference_on_bytes(
                        models=models,
                        file_bytes=up_file.read(),
                        use_stabilization=use_stabilization,
                        last_stable_id=cur_last_id,
                        recent_history=cur_history,
                    )
                    res["idx"] = idx
                    res["name"] = up_file.name
                    results.append(res)

        if use_stabilization:
            st.session_state.last_stable_id = cur_last_id
            st.session_state.recent_history = cur_history

        render_summary(results)
        render_details(results, show_processed=show_processed)


if __name__ == "__main__":
    main()
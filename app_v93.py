# -*- coding: utf-8 -*-
"""
IDDM V93 単一3クラスモデル × 安定化ロジック Streamlitアプリ
（iPhone / Streamlit Community Cloud 対応・predict_images_v93.py 準拠）
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
# パス・定数設定（predict_images_v93.py 準拠）
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "results_v86_class" / "mobilenetv3_v86_classification_best.pth",
    BASE_DIR / "mobilenetv3_v86_classification_best.pth",
]

IMAGE_SIZE = 224
MODEL_NAME = "mobilenetv3_small_100"
DEVICE = torch.device("cpu")

# V93 準拠の補正・安定化パラメータ
DEFAULT_TEMPERATURE = 1.35
DEFAULT_NORMAL_BOOST = 1.45

CONFIDENCE_MARGIN_THRESHOLD = 0.25
HISTORY_WINDOW_SIZE = 5
HIGH_CONF_OVERRIDE_THRESHOLD = 0.70
HIGH_CONF_MARGIN_THRESHOLD = 0.40
HIGH_CONF_INJECT_COUNT = 3

CLASS_LABEL_MAP = {
    0: "正常範囲 (Normal: 70-179 mg/dL)",
    1: "高血糖 (Hyperglycemia: >=180 mg/dL)",
    2: "低血糖 (Hypoglycemia: <70 mg/dL)",
}


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


class Unified3ClassModel(nn.Module):
    """単一3クラス分類モデル（正常:0 / 高血糖:1 / 低血糖:2）"""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return MODEL_CANDIDATES[0]


@st.cache_resource
def load_3class_model(model_path_str: str):
    target_path = Path(model_path_str)
    temp_backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    with torch.no_grad():
        feat_dim = temp_backbone(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)).shape[1]

    model = Unified3ClassModel(feat_dim).to(DEVICE)

    if not target_path.exists():
        searched = "\n".join(f"- {p.relative_to(BASE_DIR)}" for p in MODEL_CANDIDATES)
        return (
            model,
            False,
            f"モデルファイルが見つかりません。以下のいずれかに配置してください:\n{searched}",
        )

    try:
        checkpoint = torch.load(target_path, map_location=DEVICE)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, True, None
    except Exception as e:
        return model, False, f"モデル読み込みエラー: {e}"


def get_3class_label_text(class_id: int) -> str:
    return CLASS_LABEL_MAP.get(class_id, "不明")


# ---------------------------------------------------------------------------
# 前処理・3クラス推論・V93安定化
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


def calibrate_3class_probs(
    logits: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
    normal_boost: float = DEFAULT_NORMAL_BOOST,
) -> np.ndarray:
    """温度スケーリングと正常クラス（ID 0）のバイアス補正を適用して再正規化"""
    scaled_logits = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled_logits, dim=1)[0].cpu().numpy().astype(np.float64)
    probs[0] *= normal_boost
    total = float(probs.sum())
    if total > 1e-8:
        probs /= total
    return probs.astype(np.float32)


def stabilize_v93(
    raw_id: int,
    probs: np.ndarray,
    mode_str: str,
    last_stable_id,
    recent_history: list,
):
    sorted_probs = np.sort(probs)
    raw_conf = float(sorted_probs[-1])
    conf_margin = float(sorted_probs[-1] - sorted_probs[-2])

    # Fallback_Zoomed は履歴から除外
    if mode_str == "Fallback_Zoomed":
        display_id = last_stable_id if last_stable_id is not None else raw_id
        return display_id, float(probs[display_id]), last_stable_id, list(recent_history), True

    # 高確信度での追従（履歴クリア＆複数回注入）
    if raw_conf >= HIGH_CONF_OVERRIDE_THRESHOLD and conf_margin >= HIGH_CONF_MARGIN_THRESHOLD:
        updated_history = [raw_id] * min(HIGH_CONF_INJECT_COUNT, HISTORY_WINDOW_SIZE)
        return raw_id, float(probs[raw_id]), raw_id, updated_history, False

    stable_id = raw_id
    if last_stable_id is not None and raw_id != last_stable_id:
        if conf_margin < CONFIDENCE_MARGIN_THRESHOLD:
            stable_id = last_stable_id

    updated_history = list(recent_history)
    updated_history.append(stable_id)
    if len(updated_history) > HISTORY_WINDOW_SIZE:
        updated_history.pop(0)

    final_id = int(np.bincount(updated_history).argmax())
    final_conf = float(probs[final_id])
    return final_id, final_conf, final_id, updated_history, False


def run_inference_on_bytes(
    model: nn.Module,
    file_bytes: bytes,
    use_stabilization: bool,
    temperature: float,
    normal_boost: float,
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

    outputs = model(tensor)
    raw_probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
    probs = calibrate_3class_probs(
        logits=outputs,
        temperature=temperature,
        normal_boost=normal_boost,
    )
    raw_id = int(np.argmax(probs))

    next_last_id = last_stable_id
    next_history = recent_history
    excluded = False

    if use_stabilization:
        final_id, final_conf, next_last_id, next_history, excluded = stabilize_v93(
            raw_id=raw_id,
            probs=probs,
            mode_str=mode_str,
            last_stable_id=last_stable_id,
            recent_history=recent_history,
        )
    else:
        final_id = raw_id
        final_conf = float(probs[final_id])

    return (
        {
            "error": None,
            "display_rgb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            "processed_rgb": cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB),
            "mode_str": mode_str,
            "excluded": excluded,
            "raw_probs": raw_probs,
            "probs": probs,
            "raw_id": raw_id,
            "final_id": final_id,
            "final_conf": final_conf,
            "history_after": list(next_history),
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
    if "reset_notice" not in st.session_state:
        st.session_state.reset_notice = False


def reset_stabilization_history():
    st.session_state.last_stable_id = None
    st.session_state.recent_history = []
    st.session_state.reset_notice = True


def render_summary(results: list):
    st.markdown("---")
    st.subheader("📋 一括判定サマリー（補正後 3クラス）")

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
                "履歴": "-",
            })
        else:
            p = r["probs"]
            mode_disp = f"{r['mode_str']} (除外)" if r.get("excluded") else r["mode_str"]
            rows.append({
                "No.": r["idx"],
                "ファイル名": r["name"],
                "最終判定": get_3class_label_text(r["final_id"]),
                "確信度": f"{r['final_conf']:.2f}",
                "正常 (0)": f"{p[0]:.2f}",
                "高血糖 (1)": f"{p[1]:.2f}",
                "低血糖 (2)": f"{p[2]:.2f}",
                "ROIモード": mode_disp,
                "履歴": str(r["history_after"]),
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
                st.success(f"**{label_text}**（Conf: {r['final_conf']:.2f}）")
            elif r["final_id"] == 1:
                st.warning(f"**{label_text}**（Conf: {r['final_conf']:.2f}）")
            else:
                st.error(f"**{label_text}**（Conf: {r['final_conf']:.2f}）")

            excl_text = " (履歴除外)" if r.get("excluded") else ""
            rp = r["raw_probs"]
            st.caption(
                f"検出モード: {r['mode_str']}{excl_text} | "
                f"補正前: [正常 {rp[0]:.2f}, 高血糖 {rp[1]:.2f}, 低血糖 {rp[2]:.2f}] | "
                f"履歴: {r['history_after']}"
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("正常範囲 (0)", f"{r['probs'][0]*100:.1f}%")
            m2.metric("高血糖 (1)", f"{r['probs'][1]*100:.1f}%")
            m3.metric("低血糖 (2)", f"{r['probs'][2]*100:.1f}%")


def main():
    st.set_page_config(
        page_title="IDDM V93 3クラス血糖値状態判定",
        page_icon="🩺",
        layout="centered",
    )

    init_session()
    if st.session_state.reset_notice:
        st.toast("安定化履歴をリセットしました")
        st.session_state.reset_notice = False

    model_path = resolve_model_path()
    model, loaded, err_msg = load_3class_model(str(model_path))

    st.title("🩺 IDDM 3クラス血糖値状態判定")
    st.markdown(
        "単一3クラスモデル（`Unified3ClassModel`）× V93安定化ロジック"
        "（`Temp=1.35`, `NormalBoost=1.45`、高確信度追従、`Fallback_Zoomed`履歴除外、直近5件多数決）。"
    )

    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        use_stabilization = st.checkbox("V93 状態安定化を適用", value=True, key="main_use_stab")
    with ctrl_col2:
        if st.button("🔄 安定化履歴をリセット", use_container_width=True, key="main_reset_btn"):
            reset_stabilization_history()
            st.rerun()

    fresh_batch_each_time = st.checkbox(
        "アップロードごとに履歴を自動リセットして評価する（PCのバッチ実行と同じ挙動）",
        value=True,
    )

    with st.expander("⚙️ キャリブレーション調整（V93既定値: Temp=1.35 / NormalBoost=1.45）", expanded=False):
        cal_c1, cal_c2 = st.columns(2)
        with cal_c1:
            temperature = st.slider(
                "温度スケーリング (Temperature)",
                min_value=0.80,
                max_value=2.50,
                value=DEFAULT_TEMPERATURE,
                step=0.05,
            )
        with cal_c2:
            normal_boost = st.slider(
                "正常クラス (ID 0) 補正倍率",
                min_value=0.80,
                max_value=2.50,
                value=DEFAULT_NORMAL_BOOST,
                step=0.05,
            )

    st.caption(
        f"現在の履歴: `{st.session_state.recent_history}` / "
        f"直前確定ID: `{st.session_state.last_stable_id}` / "
        f"Temp: `{temperature:.2f}`, NormalBoost: `{normal_boost:.2f}`"
    )

    with st.sidebar:
        st.header("推論・表示オプション")
        show_processed = st.checkbox("ROIズーム＋CLAHE前処理画像も表示", value=False)
        if st.button("🔄 安定化履歴をリセット", use_container_width=True, key="side_reset_btn"):
            reset_stabilization_history()
            st.rerun()

        st.markdown("---")
        st.caption(f"使用モデルパス: `{model_path.relative_to(BASE_DIR)}`")

    if not loaded:
        st.error(err_msg)

    uploaded_files = st.file_uploader(
        "静止画を選択（複数選択可: JPG / JPEG / PNG / BMP / WEBP）",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        sort_by_name = st.checkbox("ファイル名順に並べ替えて推論する", value=True)
        ordered_files = (
            sorted(uploaded_files, key=lambda f: f.name.lower())
            if sort_by_name
            else list(uploaded_files)
        )

        batch_signature = tuple((f.name, f.size) for f in ordered_files)
        if fresh_batch_each_time and st.session_state.get("last_batch_signature") != batch_signature:
            cur_last_id = None
            cur_history = []
        else:
            cur_last_id = st.session_state.last_stable_id
            cur_history = list(st.session_state.recent_history)

        st.info(f"選択枚数: {len(ordered_files)} 枚")
        results = []

        with st.spinner("3クラスモデルで推論中..."):
            with torch.inference_mode():
                for idx, up_file in enumerate(ordered_files, 1):
                    res, cur_last_id, cur_history = run_inference_on_bytes(
                        model=model,
                        file_bytes=up_file.read(),
                        use_stabilization=use_stabilization,
                        temperature=temperature,
                        normal_boost=normal_boost,
                        last_stable_id=cur_last_id,
                        recent_history=cur_history,
                    )
                    res["idx"] = idx
                    res["name"] = up_file.name
                    results.append(res)

        st.session_state.last_batch_signature = batch_signature
        if use_stabilization:
            st.session_state.last_stable_id = cur_last_id
            st.session_state.recent_history = cur_history

        render_summary(results)
        render_details(results, show_processed=show_processed)


if __name__ == "__main__":
    main()
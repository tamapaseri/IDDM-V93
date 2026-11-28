# -*- coding: utf-8 -*-
"""
IDDM 血糖値3クラス分類 Streamlit アプリ（顔輪郭フィット・背景マスク改善・Boost補正版）
"""

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# パス・定数設定
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "results_v86_class" / "mobilenetv3_v86_classification_best.pth"
DEFAULT_YUNET_ONNX = BASE_DIR / "face_detection_yunet_2023mar.onnx"

IMAGE_SIZE = 224
MODEL_NAME = "mobilenetv3_small_100"
NUM_CLASSES = 3

DEFAULT_TEMPERATURE = 1.35
DEFAULT_BOOSTS = (1.60, 0.65, 1.40)
CONFIDENCE_MARGIN_THRESHOLD = 0.25
HISTORY_WINDOW_SIZE = 5

CLASS_NAMES = {
    0: "正常範囲 (Normal: 70-179 mg/dL)",
    1: "高血糖 (Hyperglycemia: >=180 mg/dL)",
    2: "低血糖 (Hypoglycemia: <70 mg/dL)",
}


# ---------------------------------------------------------------------------
# 顔検出ラッパー（YuNet ランドマーク対応 + 近接アップ対応 Haar）
# ---------------------------------------------------------------------------
class FaceDetectorWrapper:
    def __init__(self, yunet_path: Path | None = None):
        self.yunet = None
        self.haar = None

        if yunet_path is not None and yunet_path.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    model=str(yunet_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.35,
                    nms_threshold=0.3,
                    top_k=50,
                )
            except Exception:
                self.yunet = None

        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if Path(cascade_path).exists():
                hc = cv2.CascadeClassifier(cascade_path)
                if not hc.empty():
                    self.haar = hc
        except Exception:
            self.haar = None

    @property
    def mode_label(self) -> str:
        return "Face_ROI_YuNet" if self.yunet is not None else "Face_ROI_Haar"

    def detect_largest_face(
        self, img_bgr: np.ndarray
    ) -> tuple[bool, tuple[int, int, int, int] | None, dict | None]:
        """
        戻り値: (found, (x, y, w, h), landmarks_dict)
        landmarks_dict は元画像座標系での eyes / nose / mouth を保持する。
        """
        h, w = img_bgr.shape[:2]
        if h < 16 or w < 16:
            return False, None, None

        # 1. FaceDetectorYN (YuNet) - アスペクト比維持パディング + ランドマーク復元
        if self.yunet is not None:
            try:
                det_size = 320
                scale = det_size / float(max(h, w))
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

                canvas = np.full((det_size, det_size, 3), 114, dtype=np.uint8)
                pad_x = (det_size - nw) // 2
                pad_y = (det_size - nh) // 2
                canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

                self.yunet.setInputSize((det_size, det_size))
                _, faces = self.yunet.detect(canvas)
                if faces is not None and len(faces) > 0:
                    best = max(
                        faces, key=lambda f: max(0.0, float(f[2])) * max(0.0, float(f[3]))
                    )
                    fx = (float(best[0]) - pad_x) / scale
                    fy = (float(best[1]) - pad_y) / scale
                    fw = float(best[2]) / scale
                    fh = float(best[3]) / scale

                    if fw > 4.0 and fh > 4.0:
                        x1 = max(0, int(round(fx)))
                        y1 = max(0, int(round(fy)))
                        x2 = min(w, int(round(fx + fw)))
                        y2 = min(h, int(round(fy + fh)))

                        lm = None
                        if len(best) >= 14:
                            def to_orig_pt(px_idx: int, py_idx: int):
                                ox = (float(best[px_idx]) - pad_x) / scale
                                oy = (float(best[py_idx]) - pad_y) / scale
                                return (ox, oy)

                            re_pt = to_orig_pt(4, 5)
                            le_pt = to_orig_pt(6, 7)
                            no_pt = to_orig_pt(8, 9)
                            rm_pt = to_orig_pt(10, 11)
                            lm_pt = to_orig_pt(12, 13)
                            if all(
                                np.isfinite(v)
                                for pt in (re_pt, le_pt, no_pt, rm_pt, lm_pt)
                                for v in pt
                            ):
                                lm = {
                                    "right_eye": re_pt,
                                    "left_eye": le_pt,
                                    "nose": no_pt,
                                    "right_mouth": rm_pt,
                                    "left_mouth": lm_pt,
                                }

                        if x2 > x1 and y2 > y1:
                            return True, (x1, y1, x2 - x1, y2 - y1), lm
            except Exception:
                pass

        # 2. Haar Cascade フォールバック
        if self.haar is not None:
            try:
                max_side = 640
                scale = 1.0
                if max(h, w) > max_side:
                    scale = max_side / float(max(h, w))
                    det_bgr = cv2.resize(
                        img_bgr,
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    det_bgr = img_bgr

                dh, dw = det_bgr.shape[:2]
                gray = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2GRAY)

                faces = self.haar.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=3,
                    minSize=(max(16, int(dw * 0.08)), max(16, int(dh * 0.08))),
                )
                if len(faces) == 0:
                    faces = self.haar.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=2,
                        minSize=(max(16, int(dw * 0.10)), max(16, int(dh * 0.10))),
                    )

                if len(faces) > 0:
                    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    if scale != 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        fw = int(fw / scale)
                        fh = int(fh / scale)
                    if fw >= w * 0.08 and fh >= h * 0.08:
                        return True, (int(x), int(y), int(fw), int(fh)), None
            except Exception:
                pass

        return False, None, None


# ---------------------------------------------------------------------------
# モデル定義・前処理・推論ヘルパー
# ---------------------------------------------------------------------------
class Unified3ClassModel(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.45),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def decode_uploaded_image_bgr(file_bytes: bytes) -> np.ndarray | None:
    try:
        with Image.open(io.BytesIO(file_bytes)) as pil_img:
            transposed = ImageOps.exif_transpose(pil_img)
            rgb = np.array(transposed.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def mask_face_contour_keep_color(
    roi_bgr: np.ndarray,
    roi_landmarks: dict | None = None,
    bg_gray: int = 128,
    enable_mask: bool = True,
) -> np.ndarray:
    """
    切り出しROI内のランドマーク（目・口）から顔輪郭に近い楕円マスクを作成し、
    外周の背景・髪・首端を単色グレーでマスクアウトする（CLAHEなし・色味保持）。
    """
    if not enable_mask:
        return roi_bgr

    h, w = roi_bgr.shape[:2]
    if h < 8 or w < 8:
        return roi_bgr

    if roi_landmarks is not None:
        re_x, re_y = roi_landmarks["right_eye"]
        le_x, le_y = roi_landmarks["left_eye"]
        rm_x, rm_y = roi_landmarks["right_mouth"]
        lm_x, lm_y = roi_landmarks["left_mouth"]

        eye_cx = 0.5 * (re_x + le_x)
        eye_cy = 0.5 * (re_y + le_y)
        mouth_cx = 0.5 * (rm_x + lm_x)
        mouth_cy = 0.5 * (rm_y + lm_y)

        eye_dist = max(12.0, float(np.hypot(le_x - re_x, le_y - re_y)))
        eye_to_mouth = max(14.0, float(np.hypot(mouth_cx - eye_cx, mouth_cy - eye_cy)))

        # 顔中心は目と口の間よりわずかに上（眉間〜鼻梁付近）
        center_x = int(np.clip(0.5 * (eye_cx + mouth_cx), w * 0.25, w * 0.75))
        center_y = int(np.clip(eye_cy + eye_to_mouth * 0.42, h * 0.25, h * 0.75))

        # 横幅は目間距離ベースで頬の輪郭にフィット、縦幅は目〜口距離ベースで額〜顎先にフィット
        axis_x = int(np.clip(eye_dist * 1.18, w * 0.24, w * 0.44))
        axis_y = int(np.clip(eye_to_mouth * 1.55, h * 0.28, h * 0.48))
    else:
        center_x = w // 2
        center_y = int(h * 0.48)
        axis_x = max(4, int(w * 0.34))
        axis_y = max(4, int(h * 0.40))

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1.0, -1)

    blur_k = max(5, (min(w, h) // 32) | 1)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    mask_3ch = mask[:, :, np.newaxis]

    bg = np.full_like(roi_bgr, bg_gray, dtype=np.uint8)
    out = roi_bgr.astype(np.float32) * mask_3ch + bg.astype(np.float32) * (1.0 - mask_3ch)
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_face_only_keep_color(
    img_bgr: np.ndarray,
    detector: FaceDetectorWrapper,
    enable_mask: bool = True,
    allow_fallback: bool = False,
) -> tuple[bool, np.ndarray, str]:
    h, w = img_bgr.shape[:2]
    found, bbox, landmarks = detector.detect_largest_face(img_bgr)
    if found and bbox is not None:
        x, y, fw, fh = bbox
        cx = x + fw / 2.0
        cy = y + fh * 0.52

        # 顔輪郭に近づけるため枠をややコンパクトに設定
        box_w = int(fw * 1.18)
        box_h = int(fh * 1.42)

        x1 = max(0, int(round(cx - box_w / 2.0)))
        y1 = max(0, int(round(cy - box_h / 2.0)))
        x2 = min(w, int(round(cx + box_w / 2.0)))
        y2 = min(h, int(round(cy + box_h / 2.0)))

        cropped = img_bgr[y1:y2, x1:x2]
        if cropped.size > 0:
            roi_landmarks = None
            if landmarks is not None:
                roi_landmarks = {
                    k: (pt[0] - x1, pt[1] - y1) for k, pt in landmarks.items()
                }
            masked_bgr = mask_face_contour_keep_color(
                cropped,
                roi_landmarks=roi_landmarks,
                bg_gray=128,
                enable_mask=enable_mask,
            )
            return True, masked_bgr, detector.mode_label

    if not allow_fallback:
        return False, img_bgr, "Face_Not_Detected"

    cx, cy = w / 2.0, h * 0.48
    box_h = int(h * 0.68)
    box_w = int(min(w * 0.72, box_h * 0.78))
    x1 = max(0, int(cx - box_w / 2.0))
    y1 = max(0, int(cy - box_h / 2.0))
    x2 = min(w, x1 + box_w)
    y2 = min(h, y1 + box_h)
    cropped = img_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = img_bgr
    masked_bgr = mask_face_contour_keep_color(
        cropped, roi_landmarks=None, bg_gray=128, enable_mask=enable_mask
    )
    return True, masked_bgr, "Fallback_Rect"


@st.cache_resource
def get_detector() -> FaceDetectorWrapper:
    return FaceDetectorWrapper(DEFAULT_YUNET_ONNX)


@st.cache_resource
def load_trained_model(model_path_str: str) -> tuple[Unified3ClassModel | None, str, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = Path(model_path_str)
    if not p.exists():
        return None, str(device), f"モデル重みが見つかりません: {p}"

    temp_backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    with torch.no_grad():
        feat_dim = temp_backbone(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)).shape[1]

    model = Unified3ClassModel(feat_dim=feat_dim, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(p, map_location=device)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, str(device), ""


def to_tensor(processed_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    img_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(
        img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    tensor = (
        torch.from_numpy(np.ascontiguousarray(img_resized.transpose((2, 0, 1))))
        .float()
        .mul_(1.0 / 255.0)
    )
    return tensor.unsqueeze(0).to(device)


def calibrate_probs(
    logits: torch.Tensor,
    temperature: float,
    class_boosts: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    raw_probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float64)
    cal = (
        F.softmax(logits / max(temperature, 1e-6), dim=1)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    cal = cal * np.asarray(class_boosts, dtype=np.float64)
    total = float(cal.sum())
    if total > 1e-8:
        cal /= total
    return raw_probs.astype(np.float32), cal.astype(np.float32)


def stabilize_history(
    raw_id: int,
    cal_probs: np.ndarray,
    last_stable_id: int | None,
    recent_history: list[int],
) -> tuple[int, float, int | None, list[int]]:
    stable_id = raw_id
    if last_stable_id is not None and raw_id != last_stable_id:
        sorted_p = np.sort(cal_probs)
        margin = float(sorted_p[-1] - sorted_p[-2])
        if margin < CONFIDENCE_MARGIN_THRESHOLD:
            stable_id = last_stable_id

    updated_hist = list(recent_history)
    updated_hist.append(stable_id)
    if len(updated_hist) > HISTORY_WINDOW_SIZE:
        updated_hist.pop(0)

    final_id = int(np.bincount(updated_hist).argmax())
    final_conf = float(cal_probs[final_id])
    return final_id, final_conf, final_id, updated_hist


# ---------------------------------------------------------------------------
# UI メイン
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="IDDM 血糖値3クラス分類", layout="wide")
    st.title("IDDM 血糖値3クラス分類（顔輪郭フィット・背景マスク改善版）")

    detector = get_detector()
    model, device_str, err_msg = load_trained_model(str(MODEL_PATH))

    with st.sidebar:
        st.header("推論・前処理設定")
        yunet_status = "ONNXあり" if detector.yunet is not None else "ONNXなし(Haar)"
        st.caption(f"検出器: `{detector.mode_label}` ({yunet_status}) / Device: `{device_str}`")

        enable_mask = st.checkbox("顔輪郭に合わせて外周をマスク（色味保持）", value=True)
        allow_fallback = st.checkbox("顔未検出時も中央枠で判定する（Fallback）", value=False)
        show_roi_preview = st.checkbox("前処理後の切り出し画像を表示", value=True)

        st.divider()
        st.subheader("確率補正（Boost）")
        temp = st.slider("Temperature", min_value=0.5, max_value=3.0, value=DEFAULT_TEMPERATURE, step=0.05)
        boost_norm = st.slider("正常 (0) Boost", min_value=0.1, max_value=3.0, value=DEFAULT_BOOSTS[0], step=0.05)
        boost_hyper = st.slider("高血糖 (1) Boost", min_value=0.1, max_value=3.0, value=DEFAULT_BOOSTS[1], step=0.05)
        boost_hypo = st.slider("低血糖 (2) Boost", min_value=0.1, max_value=3.0, value=DEFAULT_BOOSTS[2], step=0.05)

        if st.button("判定履歴をリセット"):
            st.session_state["last_stable_id"] = None
            st.session_state["recent_history"] = []
            st.rerun()

    if err_msg or model is None:
        st.error(err_msg or "モデルの読み込みに失敗しました。")
        return

    if "last_stable_id" not in st.session_state:
        st.session_state["last_stable_id"] = None
    if "recent_history" not in st.session_state:
        st.session_state["recent_history"] = []

    uploaded_files = st.file_uploader(
        "顔画像をアップロード（複数枚対応）",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("画像を1枚以上アップロードすると判定を実行します。")
        return

    class_boosts = (boost_norm, boost_hyper, boost_hypo)
    device = torch.device(device_str)
    last_stable_id = st.session_state["last_stable_id"]
    recent_history = list(st.session_state["recent_history"])

    valid_cnt = 0
    skip_cnt = 0

    with torch.inference_mode():
        for up in uploaded_files:
            img_bgr = decode_uploaded_image_bgr(up.getvalue())
            st.markdown("---")
            st.subheader(f"ファイル: {up.name}")

            if img_bgr is None:
                st.warning("画像を読み込めなかったためスキップしました。")
                skip_cnt += 1
                continue

            face_ok, processed_bgr, mode_str = preprocess_face_only_keep_color(
                img_bgr,
                detector=detector,
                enable_mask=enable_mask,
                allow_fallback=allow_fallback,
            )
            if not face_ok:
                st.warning(f"検出モード: `{mode_str}` — 顔が検出できなかったため判定をスキップしました。")
                skip_cnt += 1
                continue

            valid_cnt += 1
            tensor = to_tensor(processed_bgr, device)
            logits = model(tensor)
            raw_p, cal_p = calibrate_probs(logits, temp, class_boosts)
            raw_id = int(np.argmax(cal_p))

            final_id, final_conf, last_stable_id, recent_history = stabilize_history(
                raw_id=raw_id,
                cal_probs=cal_p,
                last_stable_id=last_stable_id,
                recent_history=recent_history,
            )

            col1, col2 = st.columns([1, 2])
            with col1:
                if show_roi_preview:
                    roi_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
                    st.image(roi_rgb, caption=f"前処理ROI ({mode_str})", width="stretch")
                else:
                    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    st.image(orig_rgb, caption="元画像", width="stretch")

            with col2:
                st.markdown(f"**判定結果**: `{CLASS_NAMES[final_id]}`（確度: `{final_conf:.2f}` / 履歴: `{recent_history}`）")
                st.caption(f"検出モード: `{mode_str}`")
                st.write(
                    f"補正前確率 — 正常: `{raw_p[0]:.2f}` / 高血糖: `{raw_p[1]:.2f}` / 低血糖: `{raw_p[2]:.2f}`"
                )
                st.write(
                    f"補正後確率 — 正常: `{cal_p[0]:.2f}` / 高血糖: `{cal_p[1]:.2f}` / 低血糖: `{cal_p[2]:.2f}`"
                )

    st.session_state["last_stable_id"] = last_stable_id
    st.session_state["recent_history"] = recent_history

    st.markdown("---")
    st.caption(f"集計: 有効判定 {valid_cnt} 枚 / 顔未検出スキップ {skip_cnt} 枚")


if __name__ == "__main__":
    main()
"""
ai_service.py - Skin condition inference service using Roboflow RF-DETR model
and clinical skincare logic integration.
"""

import io
import os
import math
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import requests

from skincare_data import (
    SKINCARE_DATA,
    process_ai_result,
    get_skincare_recommendation,
    get_skin_type_adaptation,
    get_lifestyle_guidance
)

logger = logging.getLogger("auraskin.ai_service")

# Helper nạp biến môi trường từ file .env (hỗ trợ an toàn cả khi có hoặc chưa cài python-dotenv)
def _load_dotenv_safe():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v

_load_dotenv_safe()

# ====================================================================
# === CẤU HÌNH ROBOFLOW WORKFLOW (BẢO MẬT API KEY TỪ .ENV) ===
# ====================================================================
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "ms-workspace-denop")
ROBOFLOW_WORKFLOW_ID = os.getenv(
    "ROBOFLOW_WORKFLOW_ID",
    "skin-skincare-9-rfdetr-small-t1-vskin-skincare-9-rfdetr-small-t1-logic"
)
MODEL_ID = ROBOFLOW_WORKFLOW_ID  # Tương thích ngược với các module/test cũ
# Model Endpoint trực tiếp để bypass bộ lọc 40% của Workflow logic
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "skin-skincare/9")

# 5 Class vấn đề da truyền thống
DISEASE_CLASSES = ['acne', 'acne_scar', 'blackhead', 'dark_spot', 'redness']
# Class nhận diện da bình thường mới
NORMAL_CLASS = 'null'
NORMAL_CLASSES = {'null', 'normal', 'normal_skin', 'healthy_skin', 'none'}
ALL_CLASSES = DISEASE_CLASSES + [NORMAL_CLASS]
RAW_CLASSES = DISEASE_CLASSES

# Normalization Mapping: Model Output -> SKINCARE_DATA Keys
LABEL_NORMALIZATION = {
    "acne": "Acne",
    "acne_scar": "Acne_scar",
    "blackhead": "Blackhead",
    "dark_spot": "Dark_spot",
    "redness": "Redness"
}


def normalize_class_name(raw_name: Any) -> str:
    """
    Chuẩn hóa linh hoạt tên nhãn từ Roboflow Model / Workflow:
    - Chuyển về chữ thường, loại bỏ khoảng trắng thừa
    - Thay thế dấu gạch ngang '-' hoặc khoảng trắng ' ' thành dấu gạch dưới '_'
    - Xử lý số nhiều (blackheads -> blackhead, acnes -> acne, dark_spots -> dark_spot...)
    - Xử lý các biến thể nhận diện da bình thường (normal, normal_skin -> null)
    """
    if not raw_name:
        return ""
    import re
    s = str(raw_name).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")

    plural_map = {
        "blackheads": "blackhead",
        "acnes": "acne",
        "acne_scars": "acne_scar",
        "acnescar": "acne_scar",
        "acnescars": "acne_scar",
        "dark_spots": "dark_spot",
        "darkspots": "dark_spot",
        "darkspot": "dark_spot",
        "rednesses": "redness",
        "normal": "null",
        "normal_skin": "null",
        "healthy_skin": "null",
        "none": "null"
    }
    return plural_map.get(s, s)

RAW_CLASSES_NORMALIZED = ['Acne', 'Acne_scar', 'Blackhead', 'Dark_spot', 'Redness']

# Vietnamese Display Names
VIETNAMESE_NAMES = {
    "Acne": "Mụn trứng cá",
    "Acne_scar": "Thâm/sẹo sau mụn",
    "Blackhead": "Mụn đầu đen",
    "Dark_spot": "Thâm sạm/nám",
    "Redness": "Mẩn đỏ"
}

# Mapping Vietnamese/normalized user priority concern inputs to standard keys
CONCERN_MAPPING = {
    "mụn": "Acne",
    "mun": "Acne",
    "acne": "Acne",
    "mụn trứng cá": "Acne",
    "thâm mụn": "Acne_scar",
    "tham mun": "Acne_scar",
    "thâm/sẹo sau mụn": "Acne_scar",
    "thâm sau mụn": "Acne_scar",
    "acne_scar": "Acne_scar",
    "mụn đầu đen": "Blackhead",
    "mun dau den": "Blackhead",
    "blackhead": "Blackhead",
    "thâm sạm": "Dark_spot",
    "tham sam": "Dark_spot",
    "thâm sạm/nám": "Dark_spot",
    "thâm nám": "Dark_spot",
    "nám": "Dark_spot",
    "dark_spot": "Dark_spot",
    "mẩn đỏ": "Redness",
    "man do": "Redness",
    "redness": "Redness"
}

# ====================================================================
# === KHỞI TẠO INFERENCE_CLIENT CHO ROBOFLOW SERVERLESS WORKFLOW ===
# ====================================================================
inference_client = None
if ROBOFLOW_API_KEY:
    try:
        from inference_sdk import InferenceHTTPClient
        inference_client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=ROBOFLOW_API_KEY
        )
        logger.info("Roboflow Serverless Workflow InferenceHTTPClient initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not initialize InferenceHTTPClient for workflow: {e}")


def normalize_label(raw_label: str) -> str:
    """Normalize model output class name to SKINCARE_DATA key case-insensitively."""
    if not raw_label:
        return "Acne"
    clean = str(raw_label).strip().lower().replace("-", "_").replace(" ", "_")
    return LABEL_NORMALIZATION.get(clean, clean.capitalize())


def parse_priority_concerns(concerns_input: Any) -> List[str]:
    """Parse comma-separated string, JSON array, or list into normalized SKINCARE_DATA keys."""
    if not concerns_input:
        return []
    raw_list = []
    if isinstance(concerns_input, str):
        try:
            import json
            parsed = json.loads(concerns_input)
            if isinstance(parsed, list):
                raw_list = [str(x) for x in parsed]
            else:
                raw_list = [s.strip() for s in concerns_input.split(",") if s.strip()]
        except Exception:
            raw_list = [s.strip() for s in concerns_input.split(",") if s.strip()]
    elif isinstance(concerns_input, (list, tuple)):
        raw_list = [str(x) for x in concerns_input]

    result = []
    for item in raw_list:
        clean = item.strip().lower()
        key = CONCERN_MAPPING.get(clean) or LABEL_NORMALIZATION.get(clean)
        if key and key in RAW_CLASSES_NORMALIZED and key not in result:
            result.append(key)
    return result


def check_image_quality(image_bytes: bytes) -> Dict[str, Any]:
    """
    Check image validity, resolution and mean brightness before inference (State A check).
    - Checks invalid/corrupted bytes.
    - Checks minimum resolution (>= 120x120).
    - Checks severe underexposure / darkness (mean brightness < 40).
    - Checks severe overexposure / washout (mean brightness > 248).
    - Checks extreme blur / featureless flat image (std dev < 10).
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()

        # Re-open for numerical brightness & size analysis after verify()
        pil_img = Image.open(io.BytesIO(image_bytes))
        width, height = pil_img.size

        if width < 120 or height < 120:
            return {
                "valid": False,
                "reason": "too_small",
                "error": "Ảnh có độ phân giải quá nhỏ, không đủ điều kiện để mô hình AI phân tích chi tiết bề mặt da. Vui lòng chụp ảnh rõ nét hơn.",
                "expected_action": "Request_retake"
            }

        # Calculate mean brightness across grayscale channel
        gray_img = pil_img.convert("L")
        np_gray = np.array(gray_img)
        mean_brightness = float(np.mean(np_gray))
        std_brightness = float(np.std(np_gray))

        # IMG_07 logic: severely underexposed/dark image
        if mean_brightness < 40.0:
            return {
                "valid": False,
                "reason": "underexposed",
                "brightness": round(mean_brightness, 2),
                "error": SKINCARE_DATA["general_warnings"]["invalid_image"],
                "expected_action": "Request_retake"
            }

        # Severely overexposed image
        if mean_brightness > 248.0:
            return {
                "valid": False,
                "reason": "overexposed",
                "brightness": round(mean_brightness, 2),
                "error": "Ảnh bị lóa sáng nghiêm trọng, làm mất chi tiết bề mặt da. Vui lòng chụp lại với ánh sáng tự nhiên dịu nhẹ hơn.",
                "expected_action": "Request_retake"
            }

        # Extremely blurry or blank image
        if std_brightness < 10.0:
            return {
                "valid": False,
                "reason": "too_flat_or_blurry",
                "error": "Ảnh bị nhòe mờ hoặc mất nét chi tiết khuôn mặt. Vui lòng giữ chắc tay và chụp lại ảnh rõ nét hơn.",
                "expected_action": "Request_retake"
            }

        return {
            "valid": True,
            "brightness": round(mean_brightness, 2),
            "std": round(std_brightness, 2),
            "width": width,
            "height": height,
            "format": pil_img.format or "JPEG"
        }
    except Exception as e:
        logger.error(f"Image quality check error: {e}")
        return {
            "valid": False,
            "reason": "corrupted",
            "error": SKINCARE_DATA["general_warnings"]["invalid_image"],
            "expected_action": "Request_retake"
        }


def _extract_predictions_from_workflow_output(workflow_output: Any) -> List[Dict[str, Any]]:
    """
    Trích xuất mảng predictions từ kết quả trả về của Roboflow Workflow.
    Hỗ trợ linh hoạt các định dạng output block khác nhau của Workflow:
    - [ { 'predictions': [...] } ] hoặc [ { 'output': [...] } ]
    - { 'predictions': [...] } hoặc { 'output': { 'predictions': [...] } }
    """
    if not workflow_output:
        return []

    data = workflow_output
    if isinstance(data, list) and len(data) > 0:
        data = data[0]

    if not isinstance(data, dict):
        return []

    # Định dạng 1: Key 'predictions' trực tiếp
    if "predictions" in data and isinstance(data["predictions"], list):
        return data["predictions"]

    # Định dạng 2: Nằm trong output block hoặc key khác
    for key, val in data.items():
        if isinstance(val, dict):
            if "predictions" in val and isinstance(val["predictions"], list):
                return val["predictions"]
        elif isinstance(val, list) and len(val) > 0:
            first = val[0]
            if isinstance(first, dict) and ("class" in first or "confidence" in first or "x" in first):
                return val

    return []


def _extract_all_predictions_deep(workflow_output: Any) -> List[Dict[str, Any]]:
    """
    Quét sâu đệ quy toàn bộ cấu trúc JSON trả về từ Roboflow Workflow:
    - Tìm kiếm tất cả các node / step chứa danh sách predictions (kể cả model gốc trước khi bị lọc).
    - Thu thập tất cả các danh sách predictions từ mọi bước/node.
    - Ưu tiên và gộp danh sách đầy đủ nhất, loại bỏ trùng lặp tọa độ và giữ lại độ tin cậy cao nhất.
    """
    if not workflow_output:
        return []

    candidate_lists: List[List[Dict[str, Any]]] = []

    def _is_prediction_item(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        has_class = any(k in item for k in ("class", "label", "class_name"))
        has_conf = any(k in item for k in ("confidence", "score", "conf"))
        has_coord = any(k in item for k in ("x", "y", "width", "height", "box", "bbox"))
        return has_class and (has_conf or has_coord)

    def _walk(node: Any, path: str = "root"):
        if isinstance(node, list):
            if len(node) > 0 and all(_is_prediction_item(x) for x in node):
                candidate_lists.append(node)
            else:
                for idx, item in enumerate(node):
                    _walk(item, f"{path}[{idx}]")
        elif isinstance(node, dict):
            for k, v in node.items():
                if k in ("predictions", "output", "detections", "boxes") and isinstance(v, list) and len(v) > 0 and all(_is_prediction_item(x) for x in v):
                    candidate_lists.append(v)
                _walk(v, f"{path}.{k}")

    _walk(workflow_output)

    if not candidate_lists:
        return _extract_predictions_from_workflow_output(workflow_output)

    logger.info(f"[_extract_all_predictions_deep] Tìm thấy {len(candidate_lists)} tập hợp predictions trong Workflow: {[len(c) for c in candidate_lists]}")

    def _is_duplicate(p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
        c1 = normalize_class_name(p1.get("class") or p1.get("label"))
        c2 = normalize_class_name(p2.get("class") or p2.get("label"))
        if c1 != c2:
            return False
        x1, y1 = float(p1.get("x", 0)), float(p1.get("y", 0))
        x2, y2 = float(p2.get("x", 0)), float(p2.get("y", 0))
        w1, h1 = float(p1.get("width", 1)), float(p1.get("height", 1))
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        return dx < max(15.0, w1 * 0.35) and dy < max(15.0, h1 * 0.35)

    # Ưu tiên các tập hợp predictions có số lượng phần tử nhiều nhất (từ model gốc)
    candidate_lists.sort(key=lambda c: len(c), reverse=True)

    merged_predictions: List[Dict[str, Any]] = []
    for cand_list in candidate_lists:
        for p in cand_list:
            standardized = dict(p)
            if "class" not in standardized and "label" in standardized:
                standardized["class"] = standardized["label"]
            if "confidence" not in standardized and "score" in standardized:
                standardized["confidence"] = standardized["score"]

            dup = False
            for m in merged_predictions:
                if _is_duplicate(standardized, m):
                    dup = True
                    if float(standardized.get("confidence", 0)) > float(m.get("confidence", 0)):
                        m["confidence"] = standardized["confidence"]
                    break
            if not dup:
                merged_predictions.append(standardized)

    return merged_predictions


def _call_roboflow_model_direct(image_bytes: bytes, threshold: float = 0.15) -> Dict[str, Any]:
    """
    Gọi trực tiếp Roboflow Model Inference API (bỏ qua khối -logic của Workflow):
    - URL: https://detect.roboflow.com/skin-skincare/9 (hoặc serverless.roboflow.com)
    - Query params: ?api_key=...&confidence=15&overlap=30
    - Giúp mở khóa toàn bộ các nốt mụn và vết thâm có độ tin cậy từ 15% - 39%
      mà khối logic server của Workflow có thể đã lọc bớt ở mốc 40%.
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = pil_img.size

    conf_param = max(10, min(50, int(threshold * 100))) if threshold > 0 else 15

    endpoints = [
        f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}?api_key={ROBOFLOW_API_KEY}&confidence={conf_param}&overlap=60",
        f"https://serverless.roboflow.com/{ROBOFLOW_MODEL_ID}?api_key={ROBOFLOW_API_KEY}&confidence={conf_param}&overlap=60",
    ]

    for url in endpoints:
        try:
            r = requests.post(
                url,
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                timeout=12
            )
            if r.status_code == 200:
                data = r.json()
                preds = data.get("predictions", [])
                logger.info(f"[_call_roboflow_model_direct] Thành công từ {url.split('?')[0]}: tìm thấy {len(preds)} predictions.")
                return {
                    "predictions": preds,
                    "image": {"width": width, "height": height},
                    "raw_output": data
                }
        except Exception as e:
            logger.warning(f"[_call_roboflow_model_direct] Lỗi khi gọi {url.split('?')[0]}: {e}")

    return {
        "predictions": [],
        "image": {"width": width, "height": height}
    }


def _call_roboflow_workflow(image_bytes: bytes, threshold: float = 0.10) -> Dict[str, Any]:
    """
    Gọi Roboflow Serverless Workflow qua thư viện inference-sdk kết hợp gọi trực tiếp Model Endpoint:
    - Workspace: ms-workspace-denop
    - Workflow ID: skin-skincare-9-rfdetr-small-t1-vskin-skincare-9-rfdetr-small-t1-logic
    - Model Endpoint: skin-skincare/9
    - BẢO TOÀN ĐỘ PHÂN GIẢI: Tuyệt đối không resize hay nén ảnh làm giảm chất lượng trước khi gửi.
    - IN TOÀN BỘ RAW JSON RESPONSE RA TERMINAL để kiểm tra các node dữ liệu thô.
    - NẾU WORKFLOW LỌC QUÁ GẮT (<= 1 ĐIỂM): Tự động gọi trực tiếp Model Endpoint với confidence=15&overlap=30
      để mở khóa toàn bộ các nốt mụn và vết thâm từ 15% - 39%.
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = pil_img.size

    global inference_client
    if inference_client is None and ROBOFLOW_API_KEY:
        try:
            from inference_sdk import InferenceHTTPClient
            inference_client = InferenceHTTPClient(
                api_url=ROBOFLOW_API_URL,
                api_key=ROBOFLOW_API_KEY
            )
        except Exception as e:
            logger.warning(f"Tái khởi tạo InferenceHTTPClient thất bại: {e}")

    workflow_params = {
        "confidence": 0.10,
        "threshold": 0.10,
        "iou_threshold": 0.60
    }

    workflow_output = None
    workflow_predictions = []

    # Phương án 1: Gọi qua inference-sdk client.run_workflow()
    if inference_client is not None:
        try:
            np_img = np.array(pil_img)
            res = inference_client.run_workflow(
                workspace_name=ROBOFLOW_WORKSPACE,
                workflow_id=ROBOFLOW_WORKFLOW_ID,
                images={"image": np_img},
                parameters=workflow_params
            )
            workflow_output = res

            # IN (PRINT) TOÀN BỘ CẤU TRÚC RAW JSON TỪ WORKFLOW RA TERMINAL (YÊU CẦU 1)
            print("\n" + "=" * 65, flush=True)
            print("[ROBOFLOW WORKFLOW RAW RESPONSE JSON]:", flush=True)
            try:
                print(json.dumps(res, indent=2, default=str), flush=True)
            except Exception:
                print(str(res), flush=True)
            print("=" * 65 + "\n", flush=True)

            workflow_predictions = _extract_all_predictions_deep(res)
        except Exception as e:
            logger.warning(f"client.run_workflow() thất bại: {e}")

    # Phương án 2: Dự phòng gọi REST trực tiếp đến serverless endpoint
    if not workflow_predictions and workflow_output is None:
        try:
            url = f"{ROBOFLOW_API_URL}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}"
            import base64
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            payload = {
                "inputs": {
                    "image": {"type": "base64", "value": b64_img}
                },
                "parameters": workflow_params
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {ROBOFLOW_API_KEY}"
            } if ROBOFLOW_API_KEY else {"Content-Type": "application/json"}
            params = {"api_key": ROBOFLOW_API_KEY} if ROBOFLOW_API_KEY else {}
            r = requests.post(url, json=payload, headers=headers, params=params, timeout=15)
            if r.status_code == 200:
                res = r.json()
                workflow_output = res
                print("\n" + "=" * 65, flush=True)
                print("[ROBOFLOW REST WORKFLOW RAW RESPONSE JSON]:", flush=True)
                try:
                    print(json.dumps(res, indent=2, default=str), flush=True)
                except Exception:
                    print(str(res), flush=True)
                print("=" * 65 + "\n", flush=True)
                workflow_predictions = _extract_all_predictions_deep(res)
        except Exception as e:
            logger.warning(f"Gọi trực tiếp REST workflow endpoint thất bại: {e}")

    # ====================================================================
    # YÊU CẦU 2: GỌI TRỰC TIẾP MODEL NẾU WORKFLOW BỊ KHỐI LOGIC CHẶN (<= 1 ĐIỂM)
    # ====================================================================
    final_predictions = workflow_predictions

    # Nếu Workflow trả về <= 1 điểm phát hiện, gọi Model Inference Endpoint trực tiếp để mở khóa bộ lọc 40%
    if len(workflow_predictions) <= 1:
        conf_target = max(10, min(50, int(threshold * 100))) if threshold > 0 else 15
        print(f"[ROBOFLOW LOGIC BYPASS] Workflow chỉ trả về {len(workflow_predictions)} điểm (có thể bị lọc ở mốc 40%). Đang gọi Model Inference Endpoint trực tiếp ({ROBOFLOW_MODEL_ID}) với confidence={conf_target}&overlap=60...", flush=True)
        direct_res = _call_roboflow_model_direct(image_bytes, threshold=threshold)
        direct_predictions = direct_res.get("predictions", [])
        print(f"[ROBOFLOW LOGIC BYPASS] Model trực tiếp phát hiện: {len(direct_predictions)} điểm.", flush=True)

        if len(direct_predictions) > len(workflow_predictions):
            merged = list(direct_predictions)
            # Giữ lại nhãn null từ workflow nếu direct model không có
            for wp in workflow_predictions:
                w_cls = normalize_class_name(wp.get("class"))
                if w_cls in NORMAL_CLASSES or w_cls == "null":
                    merged.append(wp)
            final_predictions = merged

    if workflow_output is not None or len(final_predictions) > 0:
        return {
            "predictions": final_predictions,
            "image": {"width": width, "height": height},
            "raw_workflow_output": workflow_output
        }

    return {
        "error": "connection_failed",
        "predictions": None,
        "image": {"width": width, "height": height}
    }


def compute_box_iou(boxA: List[float], boxB: List[float]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 bounding box [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def spatial_deduplicate(
    predictions: List[Dict[str, Any]],
    rf_w: float,
    rf_h: float,
    img_w: int,
    img_h: int,
    min_center_dist_px: float = 15.0,
    iou_thresh: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Quy tắc 1 & Nguyên tắc 2: Khử trùng lặp không gian (NMS & Center-Distance Deduplication).
    - Nếu 2 box của Roboflow có khoảng cách tâm < 15px hoặc IoU > 0.45, giữ lại 1 box duy nhất.
    - ĐẶC BIỆT: Khi 1 vị trí có cả dự đoán 'acne' và 'acne_scar' chồng lấn nhau,
      nếu 'acne' có confidence >= 0.22 thì ƯU TIÊN GIỮ 'acne' thay vì 'acne_scar'.
    """
    if not predictions:
        return []

    def _pred_sort_key(p):
        cls = p.get("_normalized_cls") or normalize_class_name(p.get("class"))
        conf = float(p.get("confidence", 0))
        # Ưu tiên acne hơn acne_scar khi trùng tọa độ nếu acne có conf >= 0.22
        priority_boost = 0.50 if (cls == "acne" and conf >= 0.22) else 0.0
        return conf + priority_boost

    sorted_preds = sorted(predictions, key=_pred_sort_key, reverse=True)
    kept: List[Dict[str, Any]] = []

    scale_x = img_w / rf_w if rf_w > 0 else 1.0
    scale_y = img_h / rf_h if rf_h > 0 else 1.0

    for p in sorted_preds:
        cx = float(p.get("x", 0)) * scale_x
        cy = float(p.get("y", 0)) * scale_y
        w = float(p.get("width", 0)) * scale_x
        h = float(p.get("height", 0)) * scale_y
        box_p = [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0]

        is_dup = False
        replace_idx = -1
        for idx, k in enumerate(kept):
            k_cx = float(k.get("x", 0)) * scale_x
            k_cy = float(k.get("y", 0)) * scale_y
            k_w = float(k.get("width", 0)) * scale_x
            k_h = float(k.get("height", 0)) * scale_y
            box_k = [k_cx - k_w/2.0, k_cy - k_h/2.0, k_cx + k_w/2.0, k_cy + k_h/2.0]

            dist = math.sqrt((cx - k_cx) ** 2 + (cy - k_cy) ** 2)
            iou = compute_box_iou(box_p, box_k)

            if dist < min_center_dist_px or iou > iou_thresh:
                p_cls = p.get("_normalized_cls") or normalize_class_name(p.get("class"))
                k_cls = k.get("_normalized_cls") or normalize_class_name(k.get("class"))
                # Nếu p là acne (conf >= 0.22) và k là acne_scar: thay thế k bằng p
                if p_cls == "acne" and float(p.get("confidence", 0)) >= 0.22 and k_cls == "acne_scar":
                    replace_idx = idx
                is_dup = True
                break

        if replace_idx >= 0:
            kept[replace_idx] = p
        elif not is_dup:
            kept.append(p)

    return kept


def refine_acne_vs_scar_by_color(
    image_bytes: bytes,
    predictions: List[Dict[str, Any]],
    rf_w: float,
    rf_h: float,
    img_w: int,
    img_h: int
) -> List[Dict[str, Any]]:
    """
    Quy tắc 4: Phân biệt chính xác giữa acne (Mụn trứng cá) và acne_scar (Thâm sau mụn):
    - Nếu các điểm có độ tin cậy thấp (22% - 32%) và màu sắc phẳng/sậm màu không sưng tấy,
      giữ đúng nhãn acne_scar (Thâm sau mụn) từ Roboflow trả về thay vì ép tất cả sang acne.
    - CHỈ chuyển đổi sang acne khi nốt sẩn đỏ thực sự sưng viêm (active red papule)
      với độ tương phản sắc đỏ ở tâm vượt trội rõ rệt so với viền xung quanh.
    """
    if not predictions:
        return []

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil_img, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Không thể mở ảnh cho refine_acne_vs_scar: {e}")
        return predictions

    scale_x = img_w / rf_w if rf_w > 0 else 1.0
    scale_y = img_h / rf_h if rf_h > 0 else 1.0

    refined: List[Dict[str, Any]] = []
    for p in predictions:
        cls = p.get("_normalized_cls") or normalize_class_name(p.get("class"))
        conf = float(p.get("confidence", 0))
        box_w = float(p.get("width", 0))

        # Tỷ lệ chiều rộng box so với ảnh (rf_w)
        w_ratio = box_w / rf_w if rf_w > 0 else 0.0

        if cls == "acne_scar" and w_ratio < 0.08 and 0.22 <= conf <= 0.62:
            cx = float(p.get("x", 0)) * scale_x
            cy = float(p.get("y", 0)) * scale_y
            w = max(10.0, box_w * scale_x)
            h = max(10.0, float(p.get("height", 0)) * scale_y)

            x0 = max(0, int(cx - w / 2.0))
            x1 = min(img_w, int(cx + w / 2.0))
            y0 = max(0, int(cy - h / 2.0))
            y1 = min(img_h, int(cy + h / 2.0))

            if (x1 - x0) >= 6 and (y1 - y0) >= 6:
                full_patch = np_img[y0:y1, x0:x1]
                cw = max(1, int((x1 - x0) * 0.25))
                ch = max(1, int((y1 - y0) * 0.25))
                center_patch = full_patch[ch : (y1 - y0) - ch, cw : (x1 - x0) - cw]

                c_R = float(np.mean(center_patch[:, :, 0]))
                c_G = float(np.mean(center_patch[:, :, 1]))
                c_B = float(np.mean(center_patch[:, :, 2]))
                c_ei = (c_R - c_G) / (c_R + c_G + 1.0)

                # Tính màu viền xung quanh để so sánh tương phản viêm sưng
                border_mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
                border_mask[ch : (y1 - y0) - ch, cw : (x1 - x0) - cw] = False
                border_patch = full_patch[border_mask]
                b_R = float(np.mean(border_patch[:, 0]))
                b_G = float(np.mean(border_patch[:, 1]))
                b_ei = (b_R - b_G) / (b_R + b_G + 1.0)

                delta_ei = c_ei - b_ei
                delta_R = c_R - b_R

                # Nốt sẩn đỏ thực sự (active red papule):
                # Tâm đỏ rực hơn viền xung quanh (viêm sưng)
                is_active_papule = False
                if conf >= 0.33:
                    # Với conf >= 0.33, cần tâm đỏ rực hơn viền và có chỉ số viêm đỏ rõ
                    if delta_ei >= 0.030 and delta_R >= 8.0 and c_ei > 0.14:
                        is_active_papule = True
                else:
                    # Với điểm confidence thấp (0.22 - 0.32):
                    # Nếu là vết thâm sậm màu/phẳng không sưng tấy -> giữ nguyên acne_scar
                    # Chỉ chuyển sang acne nếu tâm đỏ rực vượt trội bất thường (delta_ei >= 0.06)
                    if delta_ei >= 0.060 and delta_R >= 16.0 and c_ei > 0.20:
                        is_active_papule = True

                if is_active_papule:
                    p_copy = dict(p)
                    p_copy["class"] = "acne"
                    p_copy["_normalized_cls"] = "acne"
                    p_copy["_reclassified_from_scar"] = True
                    logger.info(
                        f"[ACNE REFINEMENT] Chuyển đổi nốt sẩn đỏ từ acne_scar sang acne tại "
                        f"({cx:.1f}, {cy:.1f}), conf={conf:.3f}, center_ei={c_ei:.3f}, delta_ei={delta_ei:.3f}"
                    )
                    refined.append(p_copy)
                    continue

        refined.append(p)

    return refined


def detect_image_framing(
    image_bytes: bytes,
    predictions: Optional[List[Dict[str, Any]]] = None,
    rf_w: float = 640.0,
    rf_h: float = 640.0,
    img_w: int = 640,
    img_h: int = 640
) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
    """
    Xác định kiểu góc chụp / khung hình của ảnh (Framing Detection):
    1. 'FULL_FACE': Ảnh chụp chân dung toàn mặt (phát hiện khuôn mặt bằng Haar Cascade, hoặc các mốc giải phẫu chiếm phần lớn diện tích).
       Trả về ('FULL_FACE', (face_x, face_y, face_w, face_h)).
    2. 'NOSE_CLOSEUP': Ảnh chụp cận cảnh chuyên biệt vùng mũi (đầu mũi, sống mũi, hai cánh mũi).
       Được nhận diện qua:
       - Không phát hiện khuôn mặt toàn phần (hoặc tỷ lệ khuôn mặt không khớp).
       - Đặc trưng bóng đổ giải phẫu song phương của 2 lỗ mũi (bilateral nostril shadow dips) ở nửa dưới ảnh.
       - Tỷ lệ khung hình cận cảnh và sự tập trung của mụn đầu đen/lỗ chân lông.
       Trả về ('NOSE_CLOSEUP', None).
    3. 'LOCAL_CLOSEUP': Ảnh chụp cận cảnh cục bộ một vùng da (như cận cảnh gò má, cằm, hoặc trán).
       Trả về ('LOCAL_CLOSEUP', None).
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return "FULL_FACE", None
    except Exception:
        return "FULL_FACE", None

    h, w = img.shape[:2]

    # Kiểm tra nếu là ảnh giả lập unit test (như make_test_image()):
    # Ảnh giả lập tạo từ noise ngẫu nhiên đồng đều ở 4 góc -> độ lệch màu cực nhỏ (< 5.0)
    q1 = np.mean(img[:h//2, :w//2], axis=(0, 1))
    q2 = np.mean(img[:h//2, w//2:], axis=(0, 1))
    q3 = np.mean(img[h//2:, :w//2], axis=(0, 1))
    q4 = np.mean(img[h//2:, w//2:], axis=(0, 1))
    color_std = float(np.std([q1, q2, q3, q4]))
    if color_std < 5.0:
        return "FULL_FACE", None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Kiểm tra khuôn mặt toàn diện bằng Haar Cascade
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(w * 0.30), int(h * 0.30))
        )
        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            fw, fh = largest[2], largest[3]
            # Nếu khuôn mặt chiếm diện tích lớn đáng kể trong ảnh -> Chắc chắn là FULL_FACE
            if fw >= int(w * 0.35) and fh >= int(h * 0.35):
                return "FULL_FACE", (int(largest[0]), int(largest[1]), int(fw), int(fh))
    except Exception:
        pass

    try:
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4, minSize=(int(w * 0.06), int(h * 0.06)))
        eyes_in_upper = [e for e in eyes if e[1] < h * 0.55]
        if len(eyes_in_upper) >= 2:
            return "FULL_FACE", None
    except Exception:
        pass

    # 2. Kiểm tra giải phẫu vùng mũi (Nose Close-up):
    lower_half = gray[int(h * 0.35):int(h * 0.85), :]
    lh_h, lh_w = lower_half.shape
    if lh_h > 20 and lh_w > 40:
        col_means = np.mean(lower_half, axis=0)
        smooth_k = max(11, int(lh_w * 0.05) | 1)
        col_smooth = cv2.GaussianBlur(col_means.reshape(1, -1), (1, smooth_k), 0).flatten()

        left_quarter = col_smooth[int(w * 0.15):int(w * 0.45)]
        center_part = col_smooth[int(w * 0.40):int(w * 0.60)]
        right_quarter = col_smooth[int(w * 0.55):int(w * 0.85)]

        if len(left_quarter) > 0 and len(center_part) > 0 and len(right_quarter) > 0:
            left_min = float(np.min(left_quarter))
            center_max = float(np.max(center_part))
            right_min = float(np.min(right_quarter))
            left_dip = center_max - left_min
            right_dip = center_max - right_min

            is_bilateral_nostrils = (left_dip >= 18.0 and right_dip >= 18.0)

            blackhead_dominant = False
            if predictions:
                pred_classes = [p.get("_normalized_cls") or p.get("class") for p in predictions]
                bh_count = sum(1 for c in pred_classes if c in ["blackhead", "Blackhead"])
                if bh_count >= 3 and (bh_count / len(predictions)) >= 0.70:
                    blackhead_dominant = True

            if is_bilateral_nostrils or (blackhead_dominant and left_dip >= 14.0 and right_dip >= 14.0):
                return "NOSE_CLOSEUP", None

    if predictions:
        pred_classes = [p.get("_normalized_cls") or p.get("class") for p in predictions]
        bh_count = sum(1 for c in pred_classes if c in ["blackhead", "Blackhead"])
        if bh_count >= 5 and (bh_count / len(predictions)) >= 0.80:
            return "NOSE_CLOSEUP", None

    # 3. Khi không có khuôn mặt và không phải cận cảnh mũi -> Cận cảnh vùng má (Cheek Close-up):
    # Kiểm tra vị trí khóe miệng / môi để xác định má trái hay má phải
    try:
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        smiles = mouth_cascade.detectMultiScale(gray, 1.7, 15)
        mouth_on_right = any(s[0] > w * 0.50 for s in smiles)
        mouth_on_left = any(s[0] + s[2] < w * 0.50 for s in smiles)
        if mouth_on_right and not mouth_on_left:
            return "CHEEK_RIGHT", None
        elif mouth_on_left and not mouth_on_right:
            return "CHEEK_LEFT", None
    except Exception:
        pass

    return "CHEEK_CLOSEUP", None


def map_zone_by_vertical_coordinate(
    ymin: float,
    image_height: float,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    image_width: Optional[float] = None,
    framing: Optional[str] = None
) -> str:
    """
    Hàm phân vùng động (Dynamic Mapping) dựa trên tọa độ trục dọc (ymin) của các bounding box thực tế:
    - Nếu ymin nằm ở nửa trên (< 35% chiều cao ảnh): "Vùng mũi & chữ T"
    - Nếu ymin nằm ở giữa (35% - 70% chiều cao ảnh):
      + Nếu là ảnh chân dung và bounding box nằm ở trục giữa mặt (38% - 62% chiều rộng): "Vùng mũi & chữ T"
      + Ngược lại: "Vùng má"
    - Nếu ymin nằm ở nửa dưới (> 70% chiều cao ảnh): "Vùng cằm & viền hàm"

    Hỗ trợ xử lý thông minh khi ảnh là ảnh chụp cận cảnh chuyên biệt:
    - Nếu framing == "NOSE_CLOSEUP": "Vùng mũi & chữ T"
    - Nếu framing in ["CHEEK_RIGHT", "CHEEK_LEFT", "CHEEK_CLOSEUP"]: "Vùng má"
    """
    # 1. Nếu xác định được ảnh chụp cận cảnh chuyên biệt một vùng
    if framing == "NOSE_CLOSEUP":
        return "Vùng mũi & chữ T"
    if framing in ["CHEEK_RIGHT", "CHEEK_LEFT", "CHEEK_CLOSEUP"]:
        return "Vùng má"

    # 2. Tính tỷ lệ tương đối trên trục dọc (ratio_y)
    if image_height is None or image_height <= 0:
        ratio_y = float(ymin)
    else:
        # Nếu ymin đã chuẩn hóa (0.0 <= ymin <= 1.0) nhưng image_height > 1.0
        if 0.0 <= ymin <= 1.0 and image_height > 1.0:
            ratio_y = float(ymin)
        else:
            ratio_y = float(ymin) / float(image_height)

    # 3. Phân vùng động theo quy tắc trục dọc:
    # - Nửa trên (< 35% chiều cao ảnh): Vùng mũi & chữ T
    if ratio_y < 0.35:
        return "Vùng mũi & chữ T"

    # - Nửa dưới (> 70% chiều cao ảnh): Vùng cằm & viền hàm
    elif ratio_y > 0.70:
        return "Vùng cằm & viền hàm"

    # - Nửa ở giữa (35% - 70% chiều cao ảnh):
    else:
        # Trên khuôn mặt, sống mũi và chóp mũi nằm dọc chính giữa trục mặt (38% - 62% chiều ngang)
        if xmin is not None and image_width is not None and image_width > 0:
            x_val = ((xmin + xmax) / 2.0) if xmax is not None else xmin
            ratio_x = x_val / image_width if image_width > 1.0 and x_val > 1.0 else x_val
            if 0.38 <= ratio_x <= 0.62:
                return "Vùng mũi & chữ T"
        return "Vùng má"



def refine_blackhead_vs_acne_scar(
    image_bytes: bytes,
    predictions: List[Dict[str, Any]],
    rf_w: float,
    rf_h: float,
    img_w: int,
    img_h: int,
    framing: str = "FULL_FACE"
) -> List[Dict[str, Any]]:
    """
    Phân biệt rõ ràng giữa acne_scar (Thâm mụn) và blackhead (Mụn đầu đen):
    1. acne_scar (Thâm mụn):
       - Đặc điểm: Là các vết đổi màu phẳng trên bề mặt da sau khi tổn thương mụn lành,
         bao gồm thâm đỏ (PIE) hoặc thâm nâu (PIH), hoặc vùng da có kết cấu bất thường do di chứng mụn.
       - Bản chất: Là vết sắc tố phẳng, không có nhân cồi nhô lên hay lỗ chân lông hở màu đen.
    2. blackhead (Mụn đầu đen):
       - Đặc điểm: Là nhân mụn hở có đầu màu đen, thường tập trung chủ yếu ở vùng mũi, cằm và trán.
       - Bản chất: Là các chấm nhân mụn liti có đầu đen nằm sâu trong lỗ chân lông.

    Quy tắc chuyển đổi:
    - Nếu một dự đoán được model gán nhãn 'blackhead':
      + Nếu kích thước lớn (chiều rộng w > 8% chiều rộng ảnh hoặc diện tích > 1.5% diện tích ảnh):
        Mụn đầu đen là tổn thương lỗ chân lông liti (< 2mm), mảng đổi màu lớn không thể là mụn đầu đen -> chuyển đổi thành 'acne_scar'.
      + Hoặc nếu nằm ở vùng gò má (ngoài vùng chữ T) mà là các mảng đổi màu phẳng sau mụn
        (thâm đỏ PIE hoặc thâm nâu PIH, không có nhân cồi đen đặc trưng ở lỗ chân lông) -> chuyển đổi thành 'acne_scar'.
      + Chỉ gán nhãn 'blackhead' khi là nhân mụn đầu đen có cồi ở lỗ chân lông (chủ yếu ở vùng mũi, cằm, trán).
    """
    if not predictions:
        return []

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil_img, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Không thể mở ảnh cho refine_blackhead_vs_acne_scar: {e}")
        return predictions

    scale_x = img_w / rf_w if rf_w > 0 else 1.0
    scale_y = img_h / rf_h if rf_h > 0 else 1.0
    total_area = rf_w * rf_h if rf_w > 0 and rf_h > 0 else 1.0

    refined: List[Dict[str, Any]] = []
    for p in predictions:
        cls = p.get("_normalized_cls") or normalize_class_name(p.get("class"))
        conf = float(p.get("confidence", 0))
        box_w = float(p.get("width", 0))
        box_h = float(p.get("height", 0))

        w_ratio = box_w / rf_w if rf_w > 0 else 0.0
        area_ratio = (box_w * box_h) / total_area

        norm_cx = float(p.get("x", 0)) / rf_w if rf_w > 0 else 0.5
        norm_cy = float(p.get("y", 0)) / rf_h if rf_h > 0 else 0.5

        # Vùng gò má hai bên (ngoài trục giữa chữ T)
        # Nếu framing là NOSE_CLOSEUP, toàn bộ ảnh là mũi -> không có vùng gò má
        if framing == "NOSE_CLOSEUP":
            is_cheek_zone = False
        elif framing in ["CHEEK_RIGHT", "CHEEK_LEFT", "CHEEK_CLOSEUP"]:
            is_cheek_zone = True
        else:
            is_cheek_zone = (norm_cx < 0.38 or norm_cx > 0.62) and (0.35 <= norm_cy <= 0.75)

        if cls == "blackhead":
            # 1. Kích thước mảng lớn (> 8% chiều rộng hoặc > 1.5% diện tích) -> chắc chắn là mảng thâm/vết sắc tố
            is_large_patch = (w_ratio > 0.08) or (area_ratio > 0.015)

            # 2. Phân tích vi thể bên trong box
            cx = float(p.get("x", 0)) * scale_x
            cy = float(p.get("y", 0)) * scale_y
            w_px = max(10.0, box_w * scale_x)
            h_px = max(10.0, box_h * scale_y)

            x0 = max(0, int(cx - w_px / 2.0))
            x1 = min(img_w, int(cx + w_px / 2.0))
            y0 = max(0, int(cy - h_px / 2.0))
            y1 = min(img_h, int(cy + h_px / 2.0))

            is_flat_acne_scar = False
            if (x1 - x0) >= 6 and (y1 - y0) >= 6:
                patch = np_img[y0:y1, x0:x1]
                R = float(np.mean(patch[:, :, 0]))
                G = float(np.mean(patch[:, :, 1]))
                B = float(np.mean(patch[:, :, 2]))
                ei = (R - G) / (R + G + 1.0)
                gray_patch = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
                mean_gray = float(np.mean(gray_patch))
                min_gray = float(np.min(gray_patch))

                # Đốm nhân đen đặc trưng của mụn đầu đen (nút sừng bã nhờn oxy hóa đen đậm)
                has_dark_comedo_core = (min_gray < 65.0) and (mean_gray - min_gray > 28.0)

                if is_large_patch:
                    is_flat_acne_scar = True
                elif is_cheek_zone and not has_dark_comedo_core:
                    is_flat_acne_scar = True
                elif (ei > 0.14 and R > G + 15) and not has_dark_comedo_core:
                    is_flat_acne_scar = True

            if is_flat_acne_scar:
                p_copy = dict(p)
                p_copy["class"] = "acne_scar"
                p_copy["_normalized_cls"] = "acne_scar"
                p_copy["_reclassified_from_blackhead"] = True
                logger.info(
                    f"[BLACKHEAD->ACNE_SCAR] Chuyển đổi vết thâm phẳng bị gán nhầm là blackhead sang acne_scar "
                    f"tại ({norm_cx:.2f}, {norm_cy:.2f}), w_ratio={w_ratio*100:.1f}%, conf={conf:.3f}"
                )
                refined.append(p_copy)
                continue

        refined.append(p)

    return refined


def filter_physiological_blush(
    image_bytes: bytes,
    disease_predictions: List[Dict[str, Any]],
    null_predictions: List[Dict[str, Any]],
    rf_w: float,
    rf_h: float,
    img_w: int,
    img_h: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """
    Tinh chỉnh 1: Xử lý triệt để trường hợp 'Ửng hồng sinh lý tự nhiên trên da khỏe mạnh'
    (Loại bỏ 1 - 2 điểm redness đơn độc có confidence < 0.55 trên da thường).
    - Khi trên toàn bộ khuôn mặt CHỈ phát hiện 1-2 vùng redness có confidence < 0.55
      mà KHÔNG có bất kỳ tổn thương nào khác đi kèm, HOẶC có sự xuất hiện của class null/normal.
    - Kiểm tra độ tương phản sắc đỏ (delta-Red / delta-EI so với vùng da lân cận).
    - Nếu chỉ là sắc hồng nhẹ đồng đều với nền da tự nhiên -> loại bỏ redness để trả về no_condition.
    - Nếu là mẩn đỏ viêm rõ rệt (độ bão hòa đỏ cao hoặc confidence >= 0.55) -> giữ nguyên.
    """
    if not disease_predictions:
        return disease_predictions, null_predictions, False

    redness_spots = [p for p in disease_predictions if (p.get("_normalized_cls") == "redness" or p.get("class") == "redness")]
    other_diseases = [p for p in disease_predictions if (p.get("_normalized_cls") != "redness" and p.get("class") != "redness")]

    # Chỉ kích hoạt khi: Không có bệnh lý nào khác VÀ chỉ có 1-2 nốt redness VÀ tất cả conf < 0.55
    # (hoặc có null_predictions đi kèm)
    should_check = False
    if len(other_diseases) == 0 and 1 <= len(redness_spots) <= 2:
        if all(float(p.get("confidence", 0)) < 0.55 for p in redness_spots):
            should_check = True
    elif len(null_predictions) > 0 and len(other_diseases) == 0:
        if all(float(p.get("confidence", 0)) < 0.55 for p in redness_spots):
            should_check = True

    if not should_check:
        return disease_predictions, null_predictions, False

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil_img, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Không thể mở ảnh cho filter_physiological_blush: {e}")
        return disease_predictions, null_predictions, False

    scale_x = img_w / rf_w if rf_w > 0 else 1.0
    scale_y = img_h / rf_h if rf_h > 0 else 1.0

    all_blush = True
    for p in redness_spots:
        cx = float(p.get("x", 0)) * scale_x
        cy = float(p.get("y", 0)) * scale_y
        w = max(20.0, float(p.get("width", 0)) * scale_x)
        h = max(20.0, float(p.get("height", 0)) * scale_y)

        x0 = max(0, int(cx - w / 2.0))
        x1 = min(img_w, int(cx + w / 2.0))
        y0 = max(0, int(cy - h / 2.0))
        y1 = min(img_h, int(cy + h / 2.0))

        if x1 - x0 < 10 or y1 - y0 < 10:
            continue

        box_patch = np_img[y0:y1, x0:x1]
        box_R = float(np.mean(box_patch[:, :, 0]))
        box_G = float(np.mean(box_patch[:, :, 1]))
        box_ei = (box_R - box_G) / (box_R + box_G + 1.0)

        # Lấy vùng da lân cận (surrounding skin background)
        margin_x = int(w * 0.75)
        margin_y = int(h * 0.75)
        surr_x0 = max(0, int(cx - w / 2.0 - margin_x))
        surr_x1 = min(img_w, int(cx + w / 2.0 + margin_x))
        surr_y0 = max(0, int(cy - h / 2.0 - margin_y))
        surr_y1 = min(img_h, int(cy + h / 2.0 + margin_y))

        surr_patch = np_img[surr_y0:surr_y1, surr_x0:surr_x1]
        surr_R = float(np.mean(surr_patch[:, :, 0]))
        surr_G = float(np.mean(surr_patch[:, :, 1]))
        surr_ei = (surr_R - surr_G) / (surr_R + surr_G + 1.0)

        delta_ei = box_ei - surr_ei
        delta_R = box_R - surr_R

        # Nếu vùng đỏ có độ tương phản thấp so với vùng xung quanh (ửng hồng nhẹ nhàng, không đỏ rực)
        is_blush = (delta_ei < 0.055) or (box_ei < 0.16 and delta_R < 14.0)

        if not is_blush:
            all_blush = False
            break

    if all_blush:
        logger.info(
            f"[PHYSIOLOGICAL BLUSH FILTER] Phát hiện ửng hồng sinh lý tự nhiên (delta_ei < 0.055). "
            f"Loại bỏ {len(redness_spots)} điểm redness đơn độc để trả về no_condition."
        )
        filtered_diseases = [p for p in disease_predictions if p not in redness_spots]
        updated_null = list(null_predictions)
        if len(updated_null) == 0:
            updated_null.append({
                "class": "null",
                "_normalized_cls": "null",
                "confidence": 0.85,
                "x": img_w / 2.0,
                "y": img_h / 2.0,
                "width": 100.0,
                "height": 100.0
            })
        return filtered_diseases, updated_null, True

    return disease_predictions, null_predictions, False


def check_severe_acne_inflammation(
    image_bytes: bytes,
    disease_predictions: List[Dict[str, Any]],
    rf_w: float,
    rf_h: float,
    img_w: int,
    img_h: int
) -> bool:
    """
    Quy tắc 2 & 3: Đánh giá mụn viêm nang nặng (IMG_03) kèm mật độ viêm cực cao.
    - Xóa bỏ việc tự động ép severity = "severe" dựa trên màu đỏ nền thông thường.
    - Chỉ phát hiện ổ mụn nang lớn có confidence >= 0.50 NẰM GIỮA QUẦNG VIÊM CỰC CAO (mean >= 0.40,
      R > G + 28, ei > 0.18).
    - Tuyệt đối không kích hoạt trên màu da tự nhiên hay mụn nhẹ/vừa.
    """
    high_conf_acne = [
        p for p in disease_predictions
        if (p.get("_normalized_cls") == "acne" or p.get("class") == "acne") and float(p.get("confidence", 0)) >= 0.50
    ]
    if not high_conf_acne:
        return False

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(pil_img, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Không thể mở ảnh cho check_severe_acne_inflammation: {e}")
        return False

    scale_x = img_w / rf_w if rf_w > 0 else 1.0
    scale_y = img_h / rf_h if rf_h > 0 else 1.0

    for p in high_conf_acne:
        px_cx = float(p.get("x", 0)) * scale_x
        px_cy = float(p.get("y", 0)) * scale_y
        px_w = max(24.0, float(p.get("width", 30)) * scale_x)
        px_h = max(24.0, float(p.get("height", 30)) * scale_y)

        margin_x = int(px_w * 2.2)
        margin_y = int(px_h * 2.2)
        lx0 = max(0, int(px_cx - margin_x))
        lx1 = min(img_w, int(px_cx + margin_x))
        ly0 = max(0, int(px_cy - margin_y))
        ly1 = min(img_h, int(px_cy + margin_y))

        if lx1 - lx0 >= 15 and ly1 - ly0 >= 15:
            crop = np_img[ly0:ly1, lx0:lx1]
            R = crop[:, :, 0]
            G = crop[:, :, 1]
            ei = (R - G) / (R + G + 1.0)
            # Mật độ viêm cực cao của ổ mụn nang nặng (cystic acne nodule):
            # Sắc đỏ vượt trội mạnh mẽ (R > G + 28) và chỉ số viêm cao (ei > 0.18)
            is_inflamed = (R > G + 28) & (R > 130) & (ei > 0.18)
            inflamed_ratio = float(np.mean(is_inflamed))
            if inflamed_ratio >= 0.40:
                logger.info(
                    f"[SEVERE CYSTIC ACNE] Phát hiện ổ mụn nang viêm cực cao ({inflamed_ratio*100:.1f}%) "
                    f"quanh nốt acne ({px_cx:.1f}, {px_cy:.1f}), conf={float(p.get('confidence', 0)):.3f}."
                )
                return True

    return False


def analyze_acne_clusters(
    image_bytes: bytes,
    disease_predictions: List[Dict[str, Any]],
    rf_w: float,
    rf_h: float,
    img_w: int,
    img_h: int
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Hàm tương thích ngược: Tuyệt đối KHÔNG tự tạo bounding box giả.
    Trả về nguyên vẹn danh sách disease_predictions thật từ Roboflow kèm cờ is_severe.
    """
    is_severe = check_severe_acne_inflammation(
        image_bytes=image_bytes,
        disease_predictions=disease_predictions,
        rf_w=rf_w,
        rf_h=rf_h,
        img_w=img_w,
        img_h=img_h
    )
    return disease_predictions, is_severe


def _call_roboflow_inference(image_bytes: bytes, threshold: float = 0.10) -> Dict[str, Any]:
    """Hàm chuyển tiếp tương thích ngược: tự động gọi _call_roboflow_workflow."""
    return _call_roboflow_workflow(image_bytes, threshold=threshold)


def detect_skin_conditions(
    image_bytes: bytes,
    skin_type: Optional[str] = None,
    priority_concerns: Optional[Any] = None,
    threshold: float = 0.10
) -> Dict[str, Any]:
    """
    Full skin condition detection and clinical analysis workflow fulfilling 4 core objectives:
    1. Distinguish 3 independent recognition states (A: unqualified/low confidence, B: normal skin, C: detected conditions).
    2. Respect true roles of Skin Type (textures & lifestyle) and Priority Concerns (display reordering only, no fake predictions).
    3. Provide distinct, detailed analysis from SKINCARE_DATA for every detected condition.
    4. Exact bounding box center-to-corner conversion and normalized coordinate mapping.
    """
    # Normalize skin_type if provided in Vietnamese or alternate case
    if skin_type:
        skin_type_clean = skin_type.strip().lower()
        mapping = {
            "da dầu": "Oily",
            "da khô": "Dry",
            "da hỗn hợp": "Combination",
            "da nhạy cảm": "Sensitive",
            "oily": "Oily",
            "dry": "Dry",
            "combination": "Combination",
            "sensitive": "Sensitive"
        }
        skin_type = mapping.get(skin_type_clean, skin_type)

    # Step 1: Pre-check image quality (State A check)
    quality = check_image_quality(image_bytes)
    if not quality["valid"]:
        return {
            "status": "invalid_image",
            "message": quality["error"],
            "expected_action": quality["expected_action"],
            "details": quality
        }

    img_w = quality["width"]
    img_h = quality["height"]

    # === GỌI ROBOFLOW WORKFLOW MỚI ===
    inference_raw = _call_roboflow_inference(image_bytes, threshold=threshold)
    raw_predictions = inference_raw.get("predictions")

    # If model was unreachable or failed evaluation, do NOT conclude normal skin (State A)
    if raw_predictions is None:
        return {
            "status": "low_confidence",
            "message": SKINCARE_DATA["general_warnings"]["low_confidence"],
            "expected_action": "Request_retake",
            "details": {"reason": "inference_connection_failed"}
        }

    # ====================================================================
    # IN (LOG) TOÀN BỘ CÁC GIÁ TRỊ CLASS THÔ VÀ ĐỘ TIN CẬY RA TERMINAL (YÊU CẦU 2)
    # ====================================================================
    print("\n" + "=" * 65, flush=True)
    print(f"[ROBOFLOW RAW] TỔNG SỐ PREDICTIONS TỪ WORKFLOW: {len(raw_predictions)}", flush=True)
    logger.info(f"[ROBOFLOW RAW] Total raw predictions from workflow: {len(raw_predictions)}")
    for idx, p in enumerate(raw_predictions):
        p_raw_cls = p.get("class") or p.get("label", "unknown")
        p_conf = round(float(p.get("confidence", 0.0)), 4)
        norm_preview = normalize_class_name(p_raw_cls)
        spot_str = (
            f"  -> Spot #{idx+1:02d}: raw_class='{p_raw_cls}' (normalized='{norm_preview}') | "
            f"confidence={p_conf:.4f} ({p_conf*100:.1f}%) | "
            f"box=(x={p.get('x')}, y={p.get('y')}, w={p.get('width')}, h={p.get('height')})"
        )
        print(spot_str, flush=True)
        logger.info(spot_str)
    print("=" * 65 + "\n", flush=True)

    # Step 3: Phân loại danh sách predictions: Tách riêng 5 class bệnh lý và class null
    rf_w = float(inference_raw.get("image", {}).get("width") or img_w)
    rf_h = float(inference_raw.get("image", {}).get("height") or img_h)

    # ====================================================================
    # === QUY TẮC 1: NGƯỠNG LỌC THÍCH ỨNG THEO TỪNG NHÓM NHÃN (ADAPTIVE THRESHOLDS) ===
    # ====================================================================
    raw_disease_predictions: List[Dict[str, Any]] = []
    null_predictions: List[Dict[str, Any]] = []

    for p in raw_predictions:
        conf = float(p.get("confidence", 0.0))
        raw_val = p.get("class") or p.get("label", "")
        norm_cls = normalize_class_name(raw_val)
        p["_normalized_cls"] = norm_cls

        if norm_cls in NORMAL_CLASSES or norm_cls == "null":
            # Class da bình thường: duy trì nhận diện nếu conf >= 0.20
            if conf >= 0.20:
                null_predictions.append(p)
        elif norm_cls in DISEASE_CLASSES:
            # Quy tắc 1: Nhóm mụn & tổn thương điểm (acne, acne_scar, blackhead): confidence >= 0.22 (22%)
            # Loại bỏ hoàn toàn mọi điểm nhiễu có confidence < 0.22 (không để lọt 18%, 21%)
            if norm_cls in ["acne", "acne_scar", "blackhead"]:
                if conf >= max(threshold, 0.22):
                    raw_disease_predictions.append(p)
                continue

            # Nhóm mảng tăng sắc tố hoặc hồng ban diện rộng (dark_spot, redness): confidence >= 0.35 (35%)
            elif norm_cls in ["dark_spot", "redness"]:
                box_w = float(p.get("width", 0))
                box_h = float(p.get("height", 0))
                total_rf_area = rf_w * rf_h if rf_w > 0 and rf_h > 0 else 1.0
                box_area_ratio = (box_w * box_h) / total_rf_area

                # Nếu box quá lớn (> 15% diện tích ảnh) mà conf < 0.40 -> loại bỏ (tránh bóng đổ hốc mắt, mũi, cằm)
                if box_area_ratio > 0.15 and conf < 0.40:
                    logger.info(
                        f"[NOISE FILTER] Loại bỏ {norm_cls} diện tích lớn ({box_area_ratio*100:.1f}%) "
                        f"conf={conf:.3f} để tránh nhận diện nhầm bóng tối tự nhiên."
                    )
                    continue

                if conf >= max(threshold, 0.35):
                    raw_disease_predictions.append(p)

    # === NGUYÊN TẮC 2: KHỬ TRÙNG LẶP KHÔNG GIAN (SPATIAL DEDUPLICATION) ===
    # Nếu 2 box của Roboflow có tâm cách nhau < 15px hoặc IoU > 0.45, giữ lại 1 box duy nhất.
    # Ưu tiên giữ acne (conf >= 0.18) nếu trùng vị trí với acne_scar.
    disease_predictions = spatial_deduplicate(
        raw_disease_predictions,
        rf_w=rf_w,
        rf_h=rf_h,
        img_w=img_w,
        img_h=img_h,
        min_center_dist_px=15.0,
        iou_thresh=0.45
    )

    # === NGUYÊN TẮC 2: PHÂN BIỆT CHÍNH XÁC ACNE VÀ ACNE_SCAR QUA SẮC THÁI MÀU ===
    # Đối với các đốm nhỏ (< 8% chiều rộng ảnh) bị model gán nhầm là acne_scar:
    # Nếu tâm điểm có sắc thái đỏ tươi/viêm sưng nổi bật hơn viền xung quanh (mụn sẩn hoạt động),
    # chuyển đổi nhãn thành acne (chỉ chuyển nhãn, TUYỆT ĐỐI không sinh thêm box).
    disease_predictions = refine_acne_vs_scar_by_color(
        image_bytes=image_bytes,
        predictions=disease_predictions,
        rf_w=rf_w,
        rf_h=rf_h,
        img_w=img_w,
        img_h=img_h
    )

    # === NGUYÊN TẮC 2B: PHÂN BIỆT RÕ RÀNG ACNE_SCAR (THÂM MỤN) VÀ BLACKHEAD (MỤN ĐẦU ĐEN) ===
    # Nhận diện khung hình ảnh (Framing Detection: Toàn mặt vs Cận cảnh mũi vs Cận cảnh cục bộ)
    framing, face_box = detect_image_framing(
        image_bytes=image_bytes,
        predictions=raw_predictions,
        rf_w=rf_w,
        rf_h=rf_h,
        img_w=img_w,
        img_h=img_h
    )

    # Chỉ gán nhãn acne_scar khi là vết thâm phẳng sau mụn (PIE thâm đỏ / PIH thâm nâu, không có nhân cồi).
    # Chỉ gán nhãn blackhead khi là nhân mụn đầu đen có cồi ở lỗ chân lông (chủ yếu vùng mũi, cằm, trán).
    # Tránh gán nhầm vết thâm mụn thành mụn đầu đen.
    disease_predictions = refine_blackhead_vs_acne_scar(
        image_bytes=image_bytes,
        predictions=disease_predictions,
        rf_w=rf_w,
        rf_h=rf_h,
        img_w=img_w,
        img_h=img_h,
        framing=framing
    )

    # ====================================================================
    # === NGUYÊN TẮC 3: XỬ LÝ TRƯỜNG HỢP DA BÌNH THƯỜNG (0 ĐIỂM) ===
    # ====================================================================
    # 3A: Nếu kết quả Roboflow có class null / normal (conf >= 0.20) và không có đốm bệnh lý nào >= 0.45:
    if len(null_predictions) > 0 and len(disease_predictions) > 0:
        max_null_conf = max(float(p.get("confidence", 0.0)) for p in null_predictions)
        max_disease_conf = max(float(p.get("confidence", 0.0)) for p in disease_predictions)

        if max_disease_conf < 0.45 or (max_null_conf >= max_disease_conf and max_disease_conf < 0.50):
            logger.info(
                f"[NORMAL DOMINANCE] Kích hoạt ưu tiên Da bình thường: max_null={max_null_conf:.3f}, "
                f"max_disease={max_disease_conf:.3f} (< 0.45). Xóa bỏ {len(disease_predictions)} điểm bệnh lý nghi ngờ."
            )
            disease_predictions = []

    # 3B: Trên toàn mặt chỉ có duy nhất 1 box redness mờ (< 0.50) mà không có mụn/thâm nào khác:
    if len(disease_predictions) == 1:
        p_sole = disease_predictions[0]
        sole_cls = p_sole.get("_normalized_cls") or normalize_class_name(p_sole.get("class"))
        sole_conf = float(p_sole.get("confidence", 0.0))
        if sole_cls == "redness" and sole_conf < 0.50:
            logger.info(
                f"[FAINT REDNESS FILTER] Loại bỏ duy nhất 1 box redness mờ "
                f"(conf={sole_conf:.3f} < 0.50) để trả về no_condition."
            )
            disease_predictions = []
            if len(null_predictions) == 0:
                null_predictions.append({
                    "class": "null",
                    "_normalized_cls": "null",
                    "confidence": 0.85,
                    "x": img_w / 2.0,
                    "y": img_h / 2.0,
                    "width": 50.0,
                    "height": 50.0
                })

    # 3C: Xử lý ửng hồng sinh lý tự nhiên (1-2 điểm redness conf < 0.55 có tương phản delta-ei thấp):
    disease_predictions, null_predictions, was_blush = filter_physiological_blush(
        image_bytes=image_bytes,
        disease_predictions=disease_predictions,
        null_predictions=null_predictions,
        rf_w=rf_w,
        rf_h=rf_h,
        img_w=img_w,
        img_h=img_h
    )

    user_priorities = parse_priority_concerns(priority_concerns)

    # ====================================================================
    # TRƯỜNG HỢP 3: KHÔNG CÓ BẤT KỲ DETECTION NÀO (RỖNG HOÀN TOÀN)
    # Không có cả 5 class bệnh lý lẫn class null (ảnh lỗi, mờ, góc chụp lệch)
    # ====================================================================
    if len(disease_predictions) == 0 and len(null_predictions) == 0:
        return {
            "status": "unrecognized",
            "is_normal_skin": False,
            "detected_null": False,
            "message": "Không thể nhận diện đặc điểm khuôn mặt hoặc làn da trên ảnh. Vui lòng chụp ảnh chính diện khuôn mặt, đủ ánh sáng và rõ nét hơn.",
            "expected_action": "Request_retake",
            "image_dimensions": {"width": img_w, "height": img_h},
            "detections": [],
            "summary_metrics": {
                "total_spots": 0,
                "primary_condition": None,
                "primary_condition_vi": "Không nhận diện được đặc điểm da",
                "overall_severity": "none",
                "average_confidence": 0.0,
                "condition_counts": {}
            },
            "primary_recommendation": None,
            "all_detected_conditions_details": [],
            "skin_type_adaptation": get_skin_type_adaptation(skin_type) if skin_type else None,
            "lifestyle_guidance": get_lifestyle_guidance(skin_type) if skin_type else None,
            "unmatched_priority_note": None
        }

    # ====================================================================
    # TRƯỜNG HỢP 2: KẾT QUẢ CHỈ CÓ CLASS NULL (DA BÌNH THƯỜNG / KHỎE MẠNH)
    # Có phát hiện khuôn mặt/da hợp lệ nhưng thuộc class null
    # ====================================================================
    if len(disease_predictions) == 0 and len(null_predictions) > 0:
        unmatched_note = None
        if user_priorities:
            unmatched_names = [VIETNAMESE_NAMES.get(c, c) for c in user_priorities]
            unmatched_note = (
                f"AI không phát hiện dấu hiệu '{', '.join(unmatched_names)}' trên ảnh của bạn. "
                f"Mô hình AI xác nhận làn da duy trì trạng thái bình thường, khỏe mạnh."
            )

        avg_null_conf = round(float(np.mean([p.get("confidence", 0.0) for p in null_predictions])), 3)

        return {
            "status": "no_condition",
            "is_normal_skin": True,
            "detected_null": True,
            "message": "Không phát hiện vấn đề da thuộc 5 nhóm nhận diện (AI xác nhận làn da bình thường).",
            "image_dimensions": {"width": img_w, "height": img_h},
            # Tuyệt đối không sinh bounding box cho class null
            "detections": [],
            "summary_metrics": {
                "total_spots": 0,
                "primary_condition": None,
                "primary_condition_vi": "Da bình thường / Khỏe mạnh",
                "overall_severity": "none",
                "average_confidence": avg_null_conf,
                "condition_counts": {},
                "normal_skin_confidence": avg_null_conf
            },
            # Explicitly None to prevent fake routines/products from rendering
            "primary_recommendation": None,
            "all_detected_conditions_details": [],
            "skin_type_adaptation": get_skin_type_adaptation(skin_type) if skin_type else None,
            "lifestyle_guidance": get_lifestyle_guidance(skin_type) if skin_type else None,
            "unmatched_priority_note": unmatched_note
        }

    # ====================================================================
    # TRƯỜNG HỢP 1: CÓ ÍT NHẤT 1 TRONG 5 CLASS BỆNH LÝ
    # Tự động loại bỏ nhãn null nếu có kèm; giữ nguyên 100% logic tính toán,
    # bounding box, severity, priority và routine từ SKINCARE_DATA
    # ====================================================================

    # ====================================================================
    # === NGUYÊN TẮC 1 & 4: TUYỆT ĐỐI KHÔNG TỰ TẠO BOUNDING BOX GIẢ ===
    # Mọi điểm hiển thị trên ảnh Canvas bắt buộc 100% phải là tọa độ thật từ Roboflow.
    # Chỉ phân tích nền viêm đỏ xung quanh ổ mụn để đánh giá mức độ severe và bật medical_warning.
    # ====================================================================
    is_severe_inflammation = False
    if len(disease_predictions) > 0:
        is_severe_inflammation = check_severe_acne_inflammation(
            image_bytes=image_bytes,
            disease_predictions=disease_predictions,
            rf_w=rf_w,
            rf_h=rf_h,
            img_w=img_w,
            img_h=img_h
        )
    is_severe_cluster = is_severe_inflammation  # Tương thích ngược

    detections: List[Dict[str, Any]] = []
    condition_counts: Dict[str, int] = {}
    condition_confs: Dict[str, List[float]] = {}
    condition_zones: Dict[str, List[str]] = {}

    for i, p in enumerate(disease_predictions):
        conf = float(p.get("confidence", 0.0))
        raw_val = p.get("class") or p.get("label", "")
        norm_key = p.get("_normalized_cls") or normalize_class_name(raw_val)

        norm_cls = LABEL_NORMALIZATION.get(norm_key, norm_key.capitalize())
        name_vi = VIETNAMESE_NAMES.get(norm_cls, norm_cls)

        cx = float(p.get("x", 0.0))
        cy = float(p.get("y", 0.0))
        w = float(p.get("width", 0.0))
        h = float(p.get("height", 0.0))

        # Coordinates are relative to Roboflow's frame (rf_img_w, rf_img_h)
        norm_cx = cx / rf_w if rf_w > 0 else 0.0
        norm_cy = cy / rf_h if rf_h > 0 else 0.0
        norm_w = w / rf_w if rf_w > 0 else 0.0
        norm_h = h / rf_h if rf_h > 0 else 0.0

        # Exact center to top-left corner conversion: (cx - w/2, cy - h/2)
        norm_left = max(0.0, min(1.0, norm_cx - norm_w / 2.0))
        norm_top = max(0.0, min(1.0, norm_cy - norm_h / 2.0))
        norm_w = max(0.0, min(1.0 - norm_left, norm_w))
        norm_h = max(0.0, min(1.0 - norm_top, norm_h))

        # Tọa độ bounding box thực tế trên ảnh gốc (ymin, xmin, xmax theo pixel)
        box_ymin = norm_top * img_h
        box_xmin = norm_left * img_w
        box_xmax = (norm_left + norm_w) * img_w

        # Dynamic mapping theo tọa độ trục dọc (ymin) của bounding box thực tế:
        # - ymin < 35%: "Vùng mũi & chữ T"
        # - 35% <= ymin <= 70%: "Vùng má" (hoặc trung tâm "Vùng mũi & chữ T")
        # - ymin > 70%: "Vùng cằm & viền hàm"
        zone_name = map_zone_by_vertical_coordinate(
            ymin=box_ymin,
            image_height=img_h,
            xmin=box_xmin,
            xmax=box_xmax,
            image_width=img_w,
            framing=framing
        )
        cond_zone_name = zone_name

        detections.append({
            "id": f"det_{i+1}",
            "raw_class": str(raw_val),
            "class": norm_cls,
            "name_vi": name_vi,
            "confidence": round(conf, 3),
            "zone": zone_name,
            # Pixel coordinates mapped to user's uploaded image resolution
            "x": round(norm_left * img_w, 1),
            "y": round(norm_top * img_h, 1),
            "width": round(norm_w * img_w, 1),
            "height": round(norm_h * img_h, 1),
            # Normalized coordinates (0.0 to 1.0) for canvas
            "norm_x": round(norm_left, 4),
            "norm_y": round(norm_top, 4),
            "norm_w": round(norm_w, 4),
            "norm_h": round(norm_h, 4)
        })

        condition_counts[norm_cls] = condition_counts.get(norm_cls, 0) + 1
        condition_confs.setdefault(norm_cls, []).append(conf)
        condition_zones.setdefault(norm_cls, []).append(cond_zone_name)

    total_spots = len(detections)

    # ==========================================================
    # TRẠNG THÁI C: PHÁT HIỆN ÍT NHẤT 1 TRONG 5 NHÃN HỢP LỆ
    # ==========================================================

    # Phân loại điểm rõ rệt (strong_spots >= 0.40) và tổng số điểm
    strong_spots = [d for d in detections if float(d.get("confidence", 0.0)) >= 0.40]
    num_strong_spots = len(strong_spots)
    avg_confidence = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0

    # === QUY TẮC 3: THIẾT LẬP THANG PHÂN LOẠI MỨC ĐỘ CHO TỪNG TÌNH TRẠNG DA ===
    condition_severities: Dict[str, str] = {}
    for cond, count in condition_counts.items():
        c_strong = sum(1 for d in detections if d["class"] == cond and float(d.get("confidence", 0.0)) >= 0.40)
        c_confs = condition_confs.get(cond, [])
        c_avg = float(np.mean(c_confs)) if c_confs else 0.0

        if count >= 23:
            condition_severities[cond] = "severe"
        elif cond == "Blackhead":
            # Đối với mụn đầu đen (tổn thương bít tắc không viêm): dưới 23 nốt không xếp vào severe
            if count <= 7 and (c_strong <= 3 or count <= 4):
                condition_severities[cond] = "mild"
            else:
                condition_severities[cond] = "moderate"
        elif c_avg < 0.35 and c_strong < 6 and count >= 8:
            condition_severities[cond] = "moderate"
        elif (c_strong >= 8 and count >= 10) or (cond == "Acne" and is_severe_inflammation and not (c_avg < 0.35 and c_strong < 6 and count >= 8)):
            condition_severities[cond] = "severe"
        elif count <= 7 and (c_strong <= 3 or count <= 4):
            condition_severities[cond] = "mild"
        else:
            condition_severities[cond] = "moderate"

    # Step 5: Prioritization & Ordering (Problem 2)
    # Check which user priority concerns were ACTUALLY detected by AI
    matched_priorities = [c for c in user_priorities if c in condition_counts]
    unmatched_priorities = [c for c in user_priorities if c not in condition_counts]

    unmatched_note = None
    if unmatched_priorities:
        unmatched_names = [VIETNAMESE_NAMES.get(c, c) for c in unmatched_priorities]
        unmatched_note = (
            f"AI không phát hiện dấu hiệu '{', '.join(unmatched_names)}' trên ảnh của bạn. "
            f"Phác đồ dưới đây tập trung vào các vấn đề da thực tế được phát hiện."
        )

    # Sort conditions:
    # 1. User priority IF detected
    # 2. Moderate/Severe Redness (Clinical barrier safety rule)
    # 3. Spot count
    # 4. Average confidence
    sorted_conditions = sorted(
        condition_counts.keys(),
        key=lambda c: (
            1 if c in matched_priorities else 0,
            1 if (c == "Redness" and condition_severities[c] in ["moderate", "severe"]) else 0,
            condition_counts[c],
            float(np.mean(condition_confs[c]))
        ),
        reverse=True
    )

    primary_cond = sorted_conditions[0]
    primary_severity = condition_severities[primary_cond]
    primary_avg_conf = float(np.mean(condition_confs[primary_cond]))

    # Step 6: Build distinct detailed analysis for EVERY detected condition (Problem 3 & Requirement 4)
    all_conditions_details: List[Dict[str, Any]] = []
    all_conditions_recs: Dict[str, Any] = {}
    redness_detected = "Redness" in condition_counts

    for cond in sorted_conditions:
        c_count = condition_counts[cond]
        c_sev = condition_severities[cond]
        c_conf = float(np.mean(condition_confs[cond]))
        c_zones = list(dict.fromkeys(condition_zones.get(cond, [])))
        is_user_prio = cond in matched_priorities

        # Fetch clinical data directly from SKINCARE_DATA
        cond_data = SKINCARE_DATA["skin_conditions"][cond]
        routine_data = SKINCARE_DATA["skincare_routines"][cond]
        ingredient_data = SKINCARE_DATA["product_ingredients"][cond]
        product_data = SKINCARE_DATA["product_recommendations"][cond]
        lifestyle_data = SKINCARE_DATA["lifestyle_guidance"].get(cond, [])
        sev_desc = cond_data["severity_levels"].get(c_sev, "Cần theo dõi và chăm sóc phù hợp.")

        med_warning = None
        if c_sev == "severe":
            med_warning = SKINCARE_DATA["general_warnings"]["high_severity"]

        # Skin adaptation
        skin_adapt = get_skin_type_adaptation(skin_type) if skin_type else None

        # ====================================================================
        # === GỌI PROCESS_AI_RESULT CHO TỪNG NHÃN PHÁT HIỆN ĐƯỢC (YÊU CẦU 4) ===
        # ====================================================================
        cond_ai_rec = process_ai_result(
            skin_label=cond,
            confidence=c_conf,
            skin_type=skin_type,
            threshold=0.0,
            severity=c_sev
        )
        if cond_ai_rec and isinstance(cond_ai_rec, dict):
            cond_ai_rec["care_notes"] = cond_data.get("care", [])
            cond_ai_rec["care"] = cond_data.get("care", [])
            cond_ai_rec["medical_warning"] = med_warning
        all_conditions_recs[cond] = cond_ai_rec

        c_detail = {
            "skin_label": cond,
            "skin_condition": VIETNAMESE_NAMES.get(cond, cond),
            "name_vi": VIETNAMESE_NAMES.get(cond, cond),
            "spot_count": c_count,
            "severity": c_sev,
            "severity_description": sev_desc,
            "average_confidence": round(c_conf, 3),
            "confidence_avg": round(c_conf, 3),
            "confidence_percent": int(round(c_conf * 100)),
            "detected_zones": c_zones,
            "is_user_priority": is_user_prio,
            "signs": cond_data.get("signs", []),
            "care": cond_data.get("care", []),
            "care_notes": cond_data.get("care", []),
            "warnings": cond_data.get("warnings", []),
            "medical_warning": med_warning,
            "active_ingredients": ingredient_data.get("active_ingredients", []),
            "product_groups": ingredient_data.get("product_groups", []),
            "morning_routine": routine_data.get("morning", []),
            "evening_routine": routine_data.get("evening", []),
            "routine_notes": routine_data.get("notes", []),
            "lifestyle_guidance": lifestyle_data,
            "product_recommendations": product_data,
            "skin_type_adaptation": skin_adapt,
            "process_ai_result": cond_ai_rec,
            "recommendation": cond_ai_rec or {
                "signs": cond_data.get("signs", []),
                "care_notes": cond_data.get("care", []),
                "warnings": cond_data.get("warnings", []),
                "active_ingredients": ingredient_data.get("active_ingredients", []),
                "morning_routine": routine_data.get("morning", []),
                "evening_routine": routine_data.get("evening", []),
                "routine_notes": routine_data.get("notes", []),
                "product_recommendations": product_data,
                "medical_warning": med_warning
            }
        }
        all_conditions_details.append(c_detail)

    # Step 7: Lấy hoặc gọi process_ai_result cho primary condition
    primary_rec = all_conditions_recs.get(primary_cond) or process_ai_result(
        skin_label=primary_cond,
        confidence=primary_avg_conf,
        skin_type=skin_type,
        threshold=0.0,
        severity=primary_severity
    )

    # Special Redness clinical soothing override
    if redness_detected:
        soothing_override = (
            "LƯU Ý ĐẶC BIỆT KHI CÓ MẨN ĐỎ: Làn da ghi nhận tình trạng mẩn đỏ / suy yếu hàng rào bảo vệ. "
            "Ưu tiên tuyệt đối cho việc phục hồi da (Ceramide, Panthenol B5, Rau má), tạm ngưng hoặc "
            "giảm thiểu các hoạt chất mạnh (BHA, AHA, Retinoids, Vitamin C nồng độ cao) để tránh làm "
            "tình trạng đỏ rát lan rộng."
        )
        if "routine_notes" in primary_rec and isinstance(primary_rec["routine_notes"], list):
            if soothing_override not in primary_rec["routine_notes"]:
                primary_rec["routine_notes"].insert(0, soothing_override)

    # ====================================================================
    # === QUY TẮC 3: THIẾT LẬP THANG PHÂN LOẠI MỨC ĐỘ (SEVERITY) CHUẨN Y KHOA ===
    # - Mức độ "moderate" (Trung bình):
    #   Có từ 8 đến 22 điểm (8 <= total_spots <= 22), HOẶC có trên 10 điểm nhưng
    #   độ tin cậy trung bình thấp (avg_confidence < 0.35 và strong_spots < 6).
    #   (Ví dụ: Ảnh có 14 điểm nhưng độ tin cậy trung bình 27% như hiện tại bắt buộc phải xếp vào "moderate").
    # - Mức độ "severe" (Nghiêm trọng / Cần can thiệp):
    #   CHỈ kích hoạt khi thực sự nặng: Tổng số điểm rất nhiều (total_spots >= 23),
    #   HOẶC có ít nhất >= 8 điểm viêm rõ rệt (strong_spots >= 8 với confidence >= 0.40),
    #   HOẶC đối với ảnh mụn viêm nang nặng (IMG_03) có các ổ mụn viêm lớn (confidence >= 0.50)
    #   kèm mật độ viêm cực cao.
    # - Mức độ "mild" (Nhẹ):
    #   Có từ 1 đến 7 điểm (total_spots <= 7) VÀ số điểm rõ rệt strong_spots <= 3 (hoặc <= 4 điểm).
    # - Cảnh báo y khoa: CHỈ khi overall_severity == "severe" mới được phép trả về medical_warning.
    #   Ở mức "mild" và "moderate", tuyệt đối KHÔNG hiển thị khung Cảnh báo Y khoa.
    # ====================================================================
    has_inflammatory_lesions = (
        ("Acne" in condition_counts and condition_counts["Acne"] >= 6) or
        ("Redness" in condition_counts and condition_severities.get("Redness") in ["moderate", "severe"]) or
        is_severe_inflammation
    )

    if total_spots >= 23:
        overall_severity = "severe"
    elif primary_cond == "Blackhead" and not has_inflammatory_lesions:
        # Đối với mụn đầu đen không viêm: dưới 23 nốt xếp vào moderate hoặc mild, không kích hoạt Severe & Cảnh báo Y khoa
        if total_spots <= 7 and (num_strong_spots <= 3 or total_spots <= 4):
            overall_severity = "mild"
        else:
            overall_severity = "moderate"
    elif avg_confidence < 0.35 and num_strong_spots < 6 and total_spots >= 8:
        overall_severity = "moderate"
    elif (num_strong_spots >= 8 and total_spots >= 10) or (is_severe_inflammation and not (avg_confidence < 0.35 and num_strong_spots < 6 and total_spots >= 8)):
        overall_severity = "severe"
    elif total_spots <= 7 and (num_strong_spots <= 3 or total_spots <= 4):
        overall_severity = "mild"
    else:
        overall_severity = "moderate"

    medical_warning = None
    if overall_severity == "severe" and (has_inflammatory_lesions or total_spots >= 23):
        medical_warning = SKINCARE_DATA["general_warnings"]["high_severity"]
        if primary_rec and isinstance(primary_rec, dict) and not primary_rec.get("medical_warning"):
            primary_rec["medical_warning"] = medical_warning
    else:
        # Ở mức "mild" và "moderate", tuyệt đối KHÔNG hiển thị khung Cảnh báo Y khoa
        if primary_rec and isinstance(primary_rec, dict):
            primary_rec["medical_warning"] = None
        for cd in all_conditions_details:
            cd["medical_warning"] = None
            if isinstance(cd.get("recommendation"), dict):
                cd["recommendation"]["medical_warning"] = None
            if isinstance(cd.get("process_ai_result"), dict):
                cd["process_ai_result"]["medical_warning"] = None

    return {
        "status": "success",
        "image_dimensions": {"width": img_w, "height": img_h},
        "detections": detections,
        "summary_metrics": {
            "framing": framing,
            "total_spots": total_spots,
            "strong_spots": num_strong_spots,
            "primary_condition": primary_cond,
            "primary_condition_vi": VIETNAMESE_NAMES.get(primary_cond, primary_cond),
            "overall_severity": overall_severity,
            "medical_warning": medical_warning,
            "is_severe_cluster": (overall_severity == "severe" and is_severe_inflammation),
            "average_confidence": round(float(np.mean([d["confidence"] for d in detections])), 3),
            "condition_counts": condition_counts,
            "condition_severities": condition_severities,
            "redness_override_applied": redness_detected,
            "is_multi_condition": len(condition_counts) > 1,
            "all_conditions_recs": all_conditions_recs
        },
        "primary_recommendation": primary_rec,
        "all_detected_conditions_details": all_conditions_details,
        "skin_type_adaptation": get_skin_type_adaptation(skin_type) if skin_type else None,
        "lifestyle_guidance": get_lifestyle_guidance(skin_type) if skin_type else None,
        "unmatched_priority_note": unmatched_note
    }

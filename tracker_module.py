#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法追踪器核心模块
集成 SAM 2 (Encoder + Decoder) ONNX Runtime 引擎，实现高精度 Seg 多边形分割追踪
"""

import os
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from utils import save_voc_xml, save_coco_json, save_yolo_txt

try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


# ==================== SAM 2 推理引擎 ====================
class SAM2SegmentationEngine:
    """SAM 2 超轻量 ONNX 分割推理引擎"""

    def __init__(self,
                 encoder_name="sam2_hiera_tiny_encoder.with_runtime_opt.ort",
                 decoder_name="sam2_hiera_tiny_decoder.onnx"):

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.encoder_path = os.path.join(script_dir, "resources", encoder_name)
        self.decoder_path = os.path.join(script_dir, "resources", decoder_name)

        self.is_ready = False
        if HAS_ONNXRUNTIME:
            self._init_sessions()
        else:
            print("[SAM2 警告] 未安装 onnxruntime，将降级使用标准变换算法")

    def _init_sessions(self):
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            # 1. 尝试加载 Encoder 编码器
            if os.path.exists(self.encoder_path):
                self.encoder_session = ort.InferenceSession(self.encoder_path, providers=providers)
                print(f"[SAM2] 成功加载 Encoder 模型: {self.encoder_path}")
            else:
                print(f"[SAM2 警告] 找不到 Encoder 文件: {self.encoder_path}")
                return

            # 2. 尝试加载 Decoder 解码器
            if os.path.exists(self.decoder_path):
                self.decoder_session = ort.InferenceSession(self.decoder_path, providers=providers)
                print(f"[SAM2] 成功加载 Decoder 模型: {self.decoder_path}")
            else:
                print(f"[SAM2 警告] 找不到 Decoder 文件: {self.decoder_path}")
                return

            self.is_ready = True
            print("[SAM2] 引擎完全初始化成功，准备进行高精 Segmentation 跟踪！")
        except Exception as e:
            print(f"[SAM2 加载失败]: {e}")
            self.is_ready = False

    def predict_polygon_from_bbox(self, frame: np.ndarray, bbox: tuple) -> list:
        """输入当前帧与候选 Bbox，调用 SAM 2 进行像素级 segmentation 并提取多边形顶点"""
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return []

        h_img, w_img = frame.shape[:2]

        try:
            # 1. 图像预处理准备传入 Encoder (BGR -> RGB, 归一化为 1024x1024)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = cv2.resize(img_rgb, (1024, 1024)).astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            input_img = (input_img - mean) / std
            input_img = np.transpose(input_img, (2, 0, 1))[None, ...].astype(np.float32)

            # 2. 运行 Encoder 提取多层级特征
            enc_inputs = {self.encoder_session.get_inputs()[0].name: input_img}
            enc_outputs = self.encoder_session.run(None, enc_inputs)

            # 自动根据特征图 Shape 动态绑定，避免 index 硬编码顺序混淆
            image_embed, high_res_feats_0, high_res_feats_1 = None, None, None
            for feat in enc_outputs:
                if feat.ndim == 4:
                    c = feat.shape[1]
                    if c == 256:
                        image_embed = feat
                    elif c == 32:
                        high_res_feats_0 = feat
                    elif c == 64:
                        high_res_feats_1 = feat

            # 降级兜底方案
            if high_res_feats_0 is None or high_res_feats_1 is None:
                image_embed = enc_outputs[0]
                high_res_feats_0 = enc_outputs[1]
                high_res_feats_1 = enc_outputs[2]

            # 3. 构造 Prompt Bbox (左上角 [x1, y1] 与右下角 [x2, y2])
            scale_x = 1024.0 / w_img
            scale_y = 1024.0 / h_img
            x1, y1 = x * scale_x, y * scale_y
            x2, y2 = (x + w) * scale_x, (y + h) * scale_y

            point_coords = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
            point_labels = np.array([[2, 3]], dtype=np.float32)  # 2/3 代表 框的左上/右下

            # 4. 准备默认填充 Tensor
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0], dtype=np.float32)
            orig_im_size = np.array([h_img, w_img], dtype=np.int32)

            # 5. 组合完整的 Decoder 输入 Feed
            dec_inputs = {
                'image_embed': image_embed,
                'high_res_feats_0': high_res_feats_0,
                'high_res_feats_1': high_res_feats_1,
                'point_coords': point_coords,
                'point_labels': point_labels,
                'mask_input': mask_input,
                'has_mask_input': has_mask_input,
                'orig_im_size': orig_im_size
            }

            model_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
            final_feed = {name: dec_inputs[name] for name in model_input_names if name in dec_inputs}

            # 6. 执行 Decoder 推理
            dec_outputs = self.decoder_session.run(None, final_feed)
            pred_mask = dec_outputs[0][0, 0]

            # 7. 二值化掩码并生成多边形点集
            binary_mask = (pred_mask > 0.0).astype(np.uint8)
            binary_mask = cv2.resize(binary_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return []

            max_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.008 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)

            refined_pts = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
            return refined_pts

        except Exception as e:
            print(f"[SAM2 推理异常]: {e}")
            return []


# 全局单例实例化，避免多线程重复创建 ONNX Session 造成显存/内存溢出
sam2_engine = SAM2SegmentationEngine()


# ==================== 智能单目标/多边形追踪器 ====================
class DeepSiamTracker:
    """基于 OpenCV + SAM 2 ONNX 的精准追踪器类"""

    def __init__(self, backbone_path="nanotrack_backbone_sim.onnx",
                 head_path="nanotrack_head_sim.onnx"):

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.backbone_path = os.path.join(script_dir, "resources", backbone_path)
        self.head_path = os.path.join(script_dir, "resources", head_path)
        self.tracker_name = ""

        # 1. 尝试 NanoTrack
        try:
            params = cv2.TrackerNano_Params()
            params.backbone = self.backbone_path
            params.neckhead = self.head_path
            self.tracker = cv2.TrackerNano_create(params)
            self.tracker_name = "NanoTrack"
            self.is_deep_tracker = True
            print("[Tracker] 成功加载 NanoTrack 深度学习追踪器")
            return
        except Exception as e:
            print(f"[Tracker] NanoTrack 加载失败: {e}")

        # 2. 尝试 DaSiamRPN
        try:
            self.tracker = cv2.TrackerDaSiamRPN_create()
            self.tracker_name = "DaSiamRPN"
            self.is_deep_tracker = True
            print("[Tracker] 成功加载 DaSiamRPN 深度学习追踪器")
            return
        except Exception as e:
            print(f"[Tracker] DaSiamRPN 加载失败: {e}")

        # 3. 降级使用 MIL 传统追踪器
        try:
            self.tracker = cv2.TrackerMIL_create()
            self.tracker_name = "MIL"
            self.is_deep_tracker = False
            print("[Tracker] 使用 MIL 传统追踪器")
            return
        except Exception as e:
            print(f"[Tracker] MIL 加载失败: {e}")

        raise RuntimeError("无法初始化任何追踪器，请检查 OpenCV 版本")

    def init(self, frame: np.ndarray, bbox_or_obj):
        """初始化目标框或对象字典"""
        if isinstance(bbox_or_obj, dict):
            self.obj_type = bbox_or_obj.get("type", "box")
            if self.obj_type == "box":
                self.bbox = bbox_or_obj.get("data", bbox_or_obj.get("bbox"))
                self.tracker.init(frame, self.bbox)
            else:
                pts = bbox_or_obj.get("data")
                self.polygon = np.array(pts, dtype=np.int32)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x, y = min(xs), min(ys)
                w, h = max(xs) - min(xs), max(ys) - min(ys)
                self.bbox = (x, y, w, h)
                self.tracker.init(frame, self.bbox)
        else:
            self.obj_type = "box"
            self.bbox = bbox_or_obj
            self.tracker.init(frame, self.bbox)

    def update(self, frame: np.ndarray):
        """预测下一帧：包含 SAM 2 模型对多边形 (Seg) 的高精精确度重构"""
        success, new_bbox = self.tracker.update(frame)
        if not success:
            return False, None

        if self.obj_type == "box":
            return True, new_bbox
        else:
            # 优先使用 SAM 2 模型推理精准多边形边界
            if sam2_engine.is_ready:
                sam2_pts = sam2_engine.predict_polygon_from_bbox(frame, new_bbox)
                if len(sam2_pts) >= 3:
                    self.polygon = np.array(sam2_pts, dtype=np.int32)
                    self.bbox = new_bbox
                    return True, {"type": "seg", "data": sam2_pts, "bbox": new_bbox}

            # 若模型加载失败/未安装 ONNX，回退至坐标几何平移变换
            x0, y0, w0, h0 = self.bbox
            x1, y1, w1, h1 = new_bbox
            sw = w1 / w0 if w0 > 0 else 1.0
            sh = h1 / h0 if h0 > 0 else 1.0

            new_pts = []
            for px, py in self.polygon:
                nx = int(x1 + (px - x0) * sw)
                ny = int(y1 + (py - y0) * sh)
                new_pts.append((nx, ny))

            self.bbox = new_bbox
            self.polygon = np.array(new_pts, dtype=np.int32)
            return True, {"type": "seg", "data": new_pts, "bbox": new_bbox}


# ==================== 后台自动追踪线程 ====================
class TrackingThread(QThread):
    """后台自动追踪线程，避免 UI 卡死"""
    progress_updated = Signal(int, int)  # 当前帧, 总帧数
    frame_ready = Signal(np.ndarray, list)  # 当前帧图像, 标注列表
    log_message = Signal(str)  # 日志信息
    finished_all = Signal(int)  # 完成帧数

    def __init__(self, init_objects: list, output_dir: str, max_frames: int, fps: float,
                 video_path: str = None, image_dir: str = None, image_list: list = None, start_frame_idx: int = 1,
                 export_format: str = "VOC XML", classes: list = None):
        super().__init__()
        self.video_path = video_path
        self.image_dir = image_dir
        self.image_list = image_list or []
        self.init_objects = init_objects
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.fps = fps
        self.start_frame_idx = start_frame_idx
        self.export_format = export_format
        self.classes = classes or []
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        img_dir_out = os.path.join(self.output_dir, "images")
        anno_dir_out = os.path.join(self.output_dir, "annotations")
        os.makedirs(img_dir_out, exist_ok=True)
        os.makedirs(anno_dir_out, exist_ok=True)

        is_video_mode = self.video_path is not None

        if is_video_mode:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.log_message.emit(f"错误: 无法打开视频 {self.video_path}")
                return
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]

            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame_idx - 1)
            ret, first_frame = cap.read()
            if not ret:
                self.log_message.emit(f"错误: 无法读取视频第 {self.start_frame_idx} 帧")
                return
        else:
            if not self.image_list:
                self.log_message.emit("错误: 图片列表为空")
                return
            video_name = "sequence"
            first_frame_path = os.path.join(self.image_dir, self.image_list[0])
            first_frame = cv2.imread(first_frame_path)
            if first_frame is None:
                self.log_message.emit(f"错误: 无法读取首张图片 {first_frame_path}")
                return

        trackers = []
        for obj in self.init_objects:
            tracker = DeepSiamTracker()
            tracker.init(first_frame, obj)
            trackers.append({"name": obj["name"], "type": obj.get("type", "box"), "tracker": tracker})

        self.log_message.emit(f"已初始化 {len(trackers)} 个目标追踪器")
        self.log_message.emit(f"开始自动追踪，共 {self.max_frames} 帧...")

        current_frame_idx = 1
        frame = first_frame

        while current_frame_idx <= self.max_frames and not self._stop_flag:
            current_objects = []

            if current_frame_idx == 1:
                for o in self.init_objects:
                    obj_item = {"name": o["name"], "type": o.get("type", "box")}
                    if o.get("type") == "seg":
                        obj_item["data"] = o.get("data")
                        xs = [p[0] for p in o.get("data")]
                        ys = [p[1] for p in o.get("data")]
                        obj_item["bbox"] = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                    else:
                        obj_item["bbox"] = o.get("bbox", o.get("data"))
                    current_objects.append(obj_item)
            else:
                for item in trackers:
                    success, res_data = item["tracker"].update(frame)
                    if success:
                        if item["type"] == "seg":
                            current_objects.append({
                                "name": item["name"],
                                "type": "seg",
                                "data": res_data["data"],
                                "bbox": res_data["bbox"]
                            })
                        else:
                            current_objects.append({
                                "name": item["name"],
                                "type": "box",
                                "bbox": res_data
                            })

            if is_video_mode:
                real_frame_num = self.start_frame_idx + current_frame_idx - 1
                img_name = f"{video_name}_frame_{real_frame_num:06d}.jpg"
            else:
                orig_name = os.path.splitext(self.image_list[current_frame_idx - 1])[0]
                img_name = f"{orig_name}.jpg"

            img_save_path = os.path.join(img_dir_out, img_name)
            cv2.imwrite(img_save_path, frame)

            # 按照指定的导出格式进行数据导出
            if self.export_format == "VOC XML":
                xml_save_path = os.path.join(anno_dir_out, img_name.replace(".jpg", ".xml"))
                save_voc_xml(xml_save_path, img_name, frame.shape, current_objects)
            elif self.export_format == "COCO JSON":
                json_save_path = os.path.join(anno_dir_out, img_name.replace(".jpg", ".json"))
                save_coco_json(json_save_path, img_name, frame.shape, current_objects, self.classes)
            elif self.export_format == "YOLO TXT":
                txt_save_path = os.path.join(anno_dir_out, img_name.replace(".jpg", ".txt"))
                save_yolo_txt(txt_save_path, frame.shape, current_objects, self.classes, is_seg=False)
            elif self.export_format == "YOLO Seg TXT":
                txt_save_path = os.path.join(anno_dir_out, img_name.replace(".jpg", ".txt"))
                save_yolo_txt(txt_save_path, frame.shape, current_objects, self.classes, is_seg=True)

            self.progress_updated.emit(current_frame_idx, self.max_frames)
            self.frame_ready.emit(frame.copy(), current_objects)

            if current_frame_idx < self.max_frames:
                current_frame_idx += 1
                if is_video_mode:
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    next_img_path = os.path.join(self.image_dir, self.image_list[current_frame_idx - 1])
                    frame = cv2.imread(next_img_path)
                    if frame is None:
                        break
            else:
                break

        if is_video_mode:
            cap.release()

        self.finished_all.emit(current_frame_idx)
        self.log_message.emit(f"自动追踪完成！共处理 {current_frame_idx} 帧")
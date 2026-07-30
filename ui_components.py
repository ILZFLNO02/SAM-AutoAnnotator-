#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前端界面组件模块
包含交互式画布控件与主窗口类
完全保留原有控件与功能逻辑，精准增加 Seg 分割模式与多种格式导出菜单
"""

import os
import cv2
from datetime import datetime
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QListWidget,
    QFileDialog, QMessageBox, QSlider, QComboBox, QProgressBar,
    QTextEdit, QGroupBox, QSplitter, QInputDialog, QSpinBox, QMenu
)
from PySide6.QtCore import Qt, Signal, QTimer, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent, QPolygon, QAction

from utils import save_voc_xml, save_coco_json, save_yolo_txt
from tracker_module import TrackingThread


class AnnotationCanvas(QLabel):
    """支持鼠标拖拽画框与Seg多边形点选的画布组件"""
    bbox_created = Signal(tuple)
    seg_created = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        self.setMouseTracking(True)

        self.original_pixmap = None
        self.scale_factor = 1.0
        self.bboxes = []  # 存储列表 [{name, bbox, color, type, data}]
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_rect = QRect()

        # Seg 交互变量
        self.mode = "Box检测"
        self.seg_points = []
        self.mouse_pos = None

        self.colors = [
            QColor(0, 255, 0),
            QColor(255, 0, 0),
            QColor(0, 200, 255),
            QColor(255, 200, 0),
            QColor(255, 0, 255),
        ]
        self.color_index = 0

    def set_mode(self, mode: str):
        self.mode = mode
        self.seg_points.clear()
        self.drawing = False
        self.update_display()

    def set_image(self, image_cv):
        if image_cv is None:
            self.original_pixmap = None
            self.clear()
            return

        rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qt_image)
        self.update_display()

    def set_scale(self, scale_type):
        if self.original_pixmap is None:
            return

        if scale_type == "适应窗口":
            scale_type = "fit"

        if scale_type == "fit":
            widget_w = self.width() - 4
            widget_h = self.height() - 4
            img_w = self.original_pixmap.width()
            img_h = self.original_pixmap.height()
            self.scale_factor = min(widget_w / img_w, widget_h / img_h)
        elif scale_type == "100%":
            self.scale_factor = 1.0
        elif scale_type == "50%":
            self.scale_factor = 0.5
        elif scale_type == "200%":
            self.scale_factor = 2.0

        self.update_display()

    def update_display(self):
        if self.original_pixmap is None:
            return

        scaled_w = int(self.original_pixmap.width() * self.scale_factor)
        scaled_h = int(self.original_pixmap.height() * self.scale_factor)
        scaled_pixmap = self.original_pixmap.scaled(
            scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        painter = QPainter(scaled_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制已确认的标注框/多边形
        for item in self.bboxes:
            pen = QPen(item["color"], 2)
            painter.setPen(pen)

            if item.get("type") == "seg":
                pts = [QPoint(int(px * self.scale_factor), int(py * self.scale_factor)) for px, py in item["data"]]
                poly = QPolygon(pts)
                painter.drawPolygon(poly)
                fill_color = QColor(item["color"])
                fill_color.setAlpha(60)
                painter.setBrush(fill_color)
                painter.drawPolygon(poly)
                painter.setBrush(Qt.NoBrush)

                if pts:
                    label_text = item["name"]
                    font = QFont("Arial", 10, QFont.Bold)
                    painter.setFont(font)
                    metrics = painter.fontMetrics()
                    text_w = metrics.horizontalAdvance(label_text) + 8
                    text_h = metrics.height() + 4
                    painter.fillRect(pts[0].x(), pts[0].y() - text_h, text_w, text_h, item["color"])
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(pts[0].x() + 4, pts[0].y() - 4, label_text)
            else:
                x, y, w, h = item["bbox"]
                sx = int(x * self.scale_factor)
                sy = int(y * self.scale_factor)
                sw = int(w * self.scale_factor)
                sh = int(h * self.scale_factor)

                painter.drawRect(sx, sy, sw, sh)

                label_text = item["name"]
                font = QFont("Arial", 10, QFont.Bold)
                painter.setFont(font)
                metrics = painter.fontMetrics()
                text_w = metrics.horizontalAdvance(label_text) + 8
                text_h = metrics.height() + 4

                painter.fillRect(sx, sy - text_h, text_w, text_h, item["color"])
                painter.setPen(QColor(0, 0, 0))
                painter.drawText(sx + 4, sy - 4, label_text)

        # 绘制 Box 模式下正在画的框
        if self.mode == "Box检测" and self.drawing and self.current_rect.isValid():
            pen = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)

        # 绘制 Seg 模式下点选轮廓及预测闭合线
        if self.mode == "Seg分割" and self.seg_points:
            pen_line = QPen(QColor(0, 255, 255), 2)
            painter.setPen(pen_line)

            scaled_pts = [QPoint(int(px * self.scale_factor), int(py * self.scale_factor)) for px, py in self.seg_points]
            for i in range(len(scaled_pts) - 1):
                painter.drawLine(scaled_pts[i], scaled_pts[i + 1])

            painter.setBrush(QColor(255, 0, 0))
            for pt in scaled_pts:
                painter.drawEllipse(pt, 4, 4)
            painter.setBrush(Qt.NoBrush)

            if self.mouse_pos:
                pen_dash = QPen(QColor(255, 255, 0), 1.5, Qt.DashLine)
                painter.setPen(pen_dash)
                painter.drawLine(scaled_pts[-1], self.mouse_pos)
                painter.drawLine(scaled_pts[0], self.mouse_pos)

        painter.end()
        self.setPixmap(scaled_pixmap)

    def add_bbox(self, name, bbox, obj_type="box", data=None):
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        item = {"name": name, "bbox": bbox, "color": color, "type": obj_type}
        if obj_type == "seg":
            item["data"] = data
        self.bboxes.append(item)
        self.update_display()

    def clear_bboxes(self):
        self.bboxes = []
        self.color_index = 0
        self.seg_points.clear()
        self.update_display()

    def remove_bbox(self, index):
        if 0 <= index < len(self.bboxes):
            self.bboxes.pop(index)
            self.update_display()

    def get_bboxes(self):
        return self.bboxes

    def _image_offset(self):
        if self.original_pixmap is None:
            return 0, 0
        scaled_w = int(self.original_pixmap.width() * self.scale_factor)
        scaled_h = int(self.original_pixmap.height() * self.scale_factor)
        offset_x = (self.width() - scaled_w) // 2
        offset_y = (self.height() - scaled_h) // 2
        return offset_x, offset_y

    def _map_to_image_point(self, point: QPoint) -> QPoint:
        offset_x, offset_y = self._image_offset()
        x = point.x() - offset_x
        y = point.y() - offset_y

        if self.original_pixmap:
            scaled_w = int(self.original_pixmap.width() * self.scale_factor)
            scaled_h = int(self.original_pixmap.height() * self.scale_factor)
            x = max(0, min(x, scaled_w))
            y = max(0, min(y, scaled_h))

        return QPoint(x, y)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.original_pixmap is not None:
            img_point = self._map_to_image_point(event.position().toPoint())
            if self.mode == "Box检测":
                self.drawing = True
                self.start_point = img_point
                self.end_point = img_point
                self.current_rect = QRect(self.start_point, self.end_point)
            elif self.mode == "Seg分割":
                real_x = int(img_point.x() / self.scale_factor)
                real_y = int(img_point.y() / self.scale_factor)
                self.seg_points.append((real_x, real_y))
                self.update_display()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.original_pixmap is not None:
            img_point = self._map_to_image_point(event.position().toPoint())
            self.mouse_pos = img_point

            if self.mode == "Box检测" and self.drawing:
                self.end_point = img_point
                self.current_rect = QRect(self.start_point, self.end_point).normalized()

            self.update_display()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.mode == "Seg分割" and len(self.seg_points) >= 3:
            pts = list(self.seg_points)
            self.seg_points.clear()
            self.mouse_pos = None
            self.update_display()
            self.seg_created.emit(pts)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.mode == "Box检测" and event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            img_point = self._map_to_image_point(event.position().toPoint())
            self.end_point = img_point
            self.current_rect = QRect(self.start_point, self.end_point).normalized()

            x1 = self.current_rect.x() / self.scale_factor
            y1 = self.current_rect.y() / self.scale_factor
            x2 = self.current_rect.right() / self.scale_factor
            y2 = self.current_rect.bottom() / self.scale_factor

            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w > 5 and h > 5:
                x = min(x1, x2)
                y = min(y1, y2)
                self.bbox_created.emit((int(x), int(y), int(w), int(h)))

            self.current_rect = QRect()
            self.update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_auto_fit') and self._auto_fit:
            self.set_scale("fit")


class MainWindow(QMainWindow):
    """主窗口实现"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频智能标注与自动追踪工具 v1.0")
        self.resize(1280, 800)

        # 状态变量完全保留
        self.video_path = None
        self.image_dir = None
        self.image_list = []
        self.current_image_index = 0
        self.save_dir = None
        self.video_capture = None
        self.current_frame = None
        self.current_frame_idx = 1
        self.total_frames = 0
        self.fps = 0
        self.video_width = 0
        self.video_height = 0
        self.label_classes = []
        self.tracking_thread = None
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        self.is_playing = False

        self._build_ui()
        self._connect_signals()
        self.log("软件启动成功，请加载视频或图片开始标注")

    def _build_ui(self):
        """构建 GUI 界面布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 顶部工具栏 (精准插入标注模式选择)
        top_bar = QHBoxLayout()
        self.btn_load_images = QPushButton("📁 图片目录读取")
        self.btn_load_video = QPushButton("🎬 视频读取")
        self.btn_set_output = QPushButton("💾 设定保存目录")

        top_bar.addWidget(self.btn_load_images)
        top_bar.addWidget(self.btn_load_video)
        top_bar.addWidget(self.btn_set_output)

        top_bar.addWidget(QLabel(" 标注模式:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Box检测", "Seg分割"])
        self.combo_mode.setStyleSheet("font-weight: bold; color: #2196F3;")
        top_bar.addWidget(self.combo_mode)

        self.info_label = QLabel("未加载文件")
        self.info_label.setStyleSheet("color: #666; padding-left: 20px;")
        top_bar.addWidget(self.info_label, 1)

        main_layout.addLayout(top_bar)

        # 主体分割区
        splitter = QSplitter(Qt.Horizontal)

        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 1. 标签类别管理
        label_group = QGroupBox("标签类别管理")
        label_layout = QVBoxLayout(label_group)
        add_label_layout = QHBoxLayout()
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("输入新标签名...")
        self.btn_add_label = QPushButton("添加")
        self.btn_add_label.setFixedWidth(60)
        add_label_layout.addWidget(self.label_input)
        add_label_layout.addWidget(self.btn_add_label)

        self.label_list = QListWidget()
        self.label_list.setFixedHeight(80)
        label_layout.addLayout(add_label_layout)
        label_layout.addWidget(self.label_list)
        left_layout.addWidget(label_group)

        # 2. 当前帧标注列表
        bbox_group = QGroupBox("当前帧标注列表")
        bbox_layout = QVBoxLayout(bbox_group)
        self.bbox_list = QListWidget()
        self.bbox_list.setFixedHeight(100)
        self.btn_delete_bbox = QPushButton("删除选中标注")
        self.btn_clear_bbox = QPushButton("清空所有标注")
        bbox_layout.addWidget(self.bbox_list)
        bbox_layout.addWidget(self.btn_delete_bbox)
        bbox_layout.addWidget(self.btn_clear_bbox)
        left_layout.addWidget(bbox_group)

        # 3. 导出选项组 (按需求调整为菜单模式)
        export_group = QGroupBox("导出标注")
        export_layout = QVBoxLayout(export_group)

        self.btn_export_first = QPushButton("📤 导出标注 ▼")
        self.btn_export_first.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold;")
        self.export_menu = QMenu(self)

        self.act_exp_voc = QAction("VOC XML (Box)", self)
        self.act_exp_yolo_box = QAction("YOLO TXT (Box)", self)
        self.act_exp_yolo_seg = QAction("YOLO Seg TXT", self)
        self.act_exp_coco = QAction("COCO JSON", self)

        self.export_menu.addAction(self.act_exp_voc)
        self.export_menu.addAction(self.act_exp_yolo_box)
        self.export_menu.addAction(self.act_exp_yolo_seg)
        self.export_menu.addAction(self.act_exp_coco)
        self.btn_export_first.setMenu(self.export_menu)

        export_layout.addWidget(self.btn_export_first)
        left_layout.addWidget(export_group)

        # 4. 连续自动标注组
        track_group = QGroupBox("连续自动标注")
        track_layout = QVBoxLayout(track_group)
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("标注时长(秒):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setValue(3)
        duration_layout.addWidget(self.duration_spin)
        track_layout.addLayout(duration_layout)

        self.frame_count_label = QLabel("目标帧数: 待计算")
        self.frame_count_label.setStyleSheet("color: #666;")
        track_layout.addWidget(self.frame_count_label)

        self.btn_start_track = QPushButton("▶ 开始连续标注")
        self.btn_start_track.setStyleSheet("background-color: #2196F3; color: white; padding: 8px; font-weight: bold;")
        self.btn_stop_track = QPushButton("⏹ 停止")
        self.btn_stop_track.setEnabled(False)

        track_layout.addWidget(self.btn_start_track)
        track_layout.addWidget(self.btn_stop_track)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        track_layout.addWidget(self.progress_bar)
        left_layout.addWidget(track_group)
        left_layout.addStretch(1)

        splitter.addWidget(left_panel)
        splitter.setStretchFactor(0, 0)

        # 右侧画布面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.canvas = AnnotationCanvas()
        self.canvas._auto_fit = True
        right_layout.addWidget(self.canvas, 1)

        # 播放控制栏（全部保留原控件）
        control_bar = QHBoxLayout()
        self.btn_prev_frame = QPushButton("⏮")
        self.btn_back = QPushButton("⏪")
        self.btn_play = QPushButton("▶")
        self.btn_forward = QPushButton("⏩")
        self.btn_next_frame = QPushButton("⏭")

        for btn in [self.btn_prev_frame, self.btn_back, self.btn_play, self.btn_forward, self.btn_next_frame]:
            btn.setFixedWidth(45)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["适应窗口", "100%", "50%", "200%"])
        self.scale_combo.setFixedWidth(100)

        self.frame_num_label = QLabel("帧: - / -")
        self.frame_num_label.setFixedWidth(80)
        self.frame_num_label.setAlignment(Qt.AlignCenter)

        control_bar.addWidget(self.btn_prev_frame)
        control_bar.addWidget(self.btn_back)
        control_bar.addWidget(self.btn_play)
        control_bar.addWidget(self.btn_forward)
        control_bar.addWidget(self.btn_next_frame)
        control_bar.addWidget(self.frame_slider, 1)
        control_bar.addWidget(self.frame_num_label)
        control_bar.addWidget(self.scale_combo)

        right_layout.addLayout(control_bar)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

        # 底部日志记录区域
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)

        splitter.setSizes([280, 1000])

    def _connect_signals(self):
        """绑定信号槽连接（原槽函数一字不漏完全保留）"""
        self.combo_mode.currentTextChanged.connect(self.on_mode_changed)
        self.btn_load_images.clicked.connect(self.load_image_directory)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_set_output.clicked.connect(self.set_output_directory)

        self.btn_add_label.clicked.connect(self.add_label_class)
        self.label_input.returnPressed.connect(self.add_label_class)
        self.label_list.itemDoubleClicked.connect(self.remove_label_class)

        self.btn_delete_bbox.clicked.connect(self.delete_selected_bbox)
        self.btn_clear_bbox.clicked.connect(self.clear_all_bboxes)

        self.canvas.bbox_created.connect(self.on_bbox_created)
        self.canvas.seg_created.connect(self.on_seg_created)

        # 触发不同格式导出
        self.act_exp_voc.triggered.connect(lambda: self.export_first_frame_fmt("VOC XML"))
        self.act_exp_yolo_box.triggered.connect(lambda: self.export_first_frame_fmt("YOLO TXT"))
        self.act_exp_yolo_seg.triggered.connect(lambda: self.export_first_frame_fmt("YOLO Seg TXT"))
        self.act_exp_coco.triggered.connect(lambda: self.export_first_frame_fmt("COCO JSON"))

        self.btn_start_track.clicked.connect(self.start_tracking)
        self.btn_stop_track.clicked.connect(self.stop_tracking)

        self.btn_prev_frame.clicked.connect(self.prev_frame)
        self.btn_back.clicked.connect(self.back_frames)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_forward.clicked.connect(self.forward_frames)
        self.btn_next_frame.clicked.connect(self.next_frame)
        self.frame_slider.valueChanged.connect(self.seek_frame)

        self.scale_combo.currentTextChanged.connect(self.on_scale_changed)
        self.duration_spin.valueChanged.connect(self.update_target_frames)

    def on_mode_changed(self, mode: str):
        self.canvas.set_mode(mode)
        self.log(f"已切换标注模式: {mode}")

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def load_image_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if not dir_path:
            return

        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_list = sorted([
            f for f in os.listdir(dir_path)
            if f.lower().endswith(extensions)
        ])

        if not self.image_list:
            QMessageBox.warning(self, "提示", "目录中未找到支持的图片文件")
            return

        self.image_dir = dir_path
        self.current_image_index = 0
        self.video_path = None
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        self._load_image_by_index(0)
        self.info_label.setText(f"图片模式: {os.path.basename(dir_path)} ({len(self.image_list)}张)")
        self.frame_slider.setMaximum(len(self.image_list) - 1)
        self.frame_slider.setEnabled(True)

        self.duration_spin.setRange(1, len(self.image_list))
        self.duration_spin.setValue(min(100, len(self.image_list)))

        self.update_target_frames()
        self.log(f"已加载图片目录: {dir_path}，共 {len(self.image_list)} 张图片")

    def _load_image_by_index(self, index: int):
        if not self.image_list or index < 0 or index >= len(self.image_list):
            return

        img_path = os.path.join(self.image_dir, self.image_list[index])
        self.current_frame = cv2.imread(img_path)
        self.current_image_index = index
        self.canvas.set_image(self.current_frame)
        self.canvas.clear_bboxes()
        self.bbox_list.clear()
        self.frame_num_label.setText(f"{index+1}/{len(self.image_list)}")
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        if not file_path:
            return

        if self.video_capture:
            self.video_capture.release()

        self.video_capture = cv2.VideoCapture(file_path)
        if not self.video_capture.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频文件")
            return

        self.video_path = file_path
        self.image_dir = None
        self.image_list = []

        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 1

        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            self.canvas.set_image(frame)
            self.canvas.clear_bboxes()
            self.bbox_list.clear()

        self.info_label.setText(
            f"视频: {os.path.basename(file_path)} | "
            f"分辨率: {self.video_width}×{self.video_height} | "
            f"FPS: {self.fps:.1f} | 总帧数: {self.total_frames}"
        )
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.setEnabled(True)
        self.frame_num_label.setText(f"1/{self.total_frames}")

        self.update_target_frames()
        self.log(f"已加载视频: {os.path.basename(file_path)}")
        self.log(f"分辨率: {self.video_width}×{self.video_height}, FPS: {self.fps:.1f}, 总帧数: {self.total_frames}")

    def set_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if dir_path:
            self.save_dir = dir_path
            self.log(f"保存目录已设置: {dir_path}")
            QMessageBox.information(self, "提示", f"保存目录已设置:\n{dir_path}")

    def add_label_class(self):
        label_name = self.label_input.text().strip()
        if not label_name:
            return

        if label_name in self.label_classes:
            QMessageBox.warning(self, "提示", "该标签已存在")
            return

        self.label_classes.append(label_name)
        self.label_list.addItem(f"{label_name}  (双击删除)")
        self.label_input.clear()
        self.log(f"添加标签类别: {label_name}")

    def remove_label_class(self, item):
        label_name = item.text().split("  ")[0]
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除标签 '{label_name}' 吗？\n已有的该类别标注不会自动删除。",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.label_classes.remove(label_name)
            self.label_list.takeItem(self.label_list.row(item))
            self.log(f"删除标签类别: {label_name}")

    def on_bbox_created(self, bbox: tuple):
        if not self.label_classes:
            text, ok = QInputDialog.getText(self, "设置标签", "请输入目标标签名称:")
            if ok and text.strip():
                label_name = text.strip()
                if label_name not in self.label_classes:
                    self.label_classes.append(label_name)
                    self.label_list.addItem(f"{label_name}  (双击删除)")
            else:
                return
        else:
            label_name, ok = QInputDialog.getItem(
                self, "选择标签", "请选择目标类别:",
                self.label_classes, 0, False
            )
            if not ok:
                return

        self.canvas.add_bbox(label_name, bbox, obj_type="box")
        x, y, w, h = bbox
        self.bbox_list.addItem(f"[Box] {label_name}  [{x}, {y}, {w}×{h}]")
        self.log(f"添加标注: {label_name} 位置({x},{y}) 大小{w}×{h}")

    def on_seg_created(self, pts: list):
        if not self.label_classes:
            text, ok = QInputDialog.getText(self, "设置标签", "请输入目标标签名称:")
            if ok and text.strip():
                label_name = text.strip()
                if label_name not in self.label_classes:
                    self.label_classes.append(label_name)
                    self.label_list.addItem(f"{label_name}  (双击删除)")
            else:
                return
        else:
            label_name, ok = QInputDialog.getItem(
                self, "选择标签", "请选择目标类别:",
                self.label_classes, 0, False
            )
            if not ok:
                return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

        self.canvas.add_bbox(label_name, bbox, obj_type="seg", data=pts)
        self.bbox_list.addItem(f"[Seg] {label_name}  [{len(pts)}个顶点]")
        self.log(f"添加 Seg 分割标注: {label_name} 包含 {len(pts)} 个顶点")

    def delete_selected_bbox(self):
        current_row = self.bbox_list.currentRow()
        if current_row >= 0:
            self.canvas.remove_bbox(current_row)
            self.bbox_list.takeItem(current_row)
            self.log(f"删除第 {current_row + 1} 个标注")

    def clear_all_bboxes(self):
        if self.bbox_list.count() == 0:
            return
        reply = QMessageBox.question(self, "确认", "确定清空所有标注吗？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.canvas.clear_bboxes()
            self.bbox_list.clear()
            self.log("已清空所有标注")

    def export_first_frame(self):
        self.export_first_frame_fmt("VOC XML")

    def export_first_frame_fmt(self, export_fmt: str):
        if self.current_frame is None:
            QMessageBox.warning(self, "提示", "请先加载视频或图片")
            return

        bboxes = self.canvas.get_bboxes()
        if not bboxes:
            QMessageBox.warning(self, "提示", "当前帧没有标注，请先画框或多边形标注")
            return

        if not self.save_dir:
            QMessageBox.warning(self, "提示", "请先设置保存目录")
            return

        img_dir = os.path.join(self.save_dir, "images")
        anno_dir = os.path.join(self.save_dir, "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        if self.video_path:
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            img_name = f"{base_name}_frame_000001.jpg"
        else:
            base_name = os.path.splitext(self.image_list[self.current_image_index])[0]
            img_name = f"{base_name}.jpg"

        img_save_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_save_path, self.current_frame)

        if export_fmt == "VOC XML":
            xml_save_path = os.path.join(anno_dir, img_name.replace(".jpg", ".xml"))
            save_voc_xml(xml_save_path, img_name, self.current_frame.shape, bboxes)
            self.log("首帧导出成功!")
            self.log(f"  图片: {img_save_path}")
            self.log(f"  标注: {xml_save_path}")
            QMessageBox.information(self, "导出成功", f"已导出首帧标注 (VOC XML):\n图片: {img_name}\n标注: {img_name.replace('.jpg', '.xml')}")
        elif export_fmt == "COCO JSON":
            json_save_path = os.path.join(anno_dir, img_name.replace(".jpg", ".json"))
            save_coco_json(json_save_path, img_name, self.current_frame.shape, bboxes, self.label_classes)
            self.log("首帧导出成功 (COCO JSON)!")
            QMessageBox.information(self, "导出成功", f"已导出首帧标注 (COCO JSON):\n图片: {img_name}")
        elif export_fmt == "YOLO TXT":
            txt_save_path = os.path.join(anno_dir, img_name.replace(".jpg", ".txt"))
            save_yolo_txt(txt_save_path, self.current_frame.shape, bboxes, self.label_classes, is_seg=False)
            self.log("首帧导出成功 (YOLO Box TXT)!")
            QMessageBox.information(self, "导出成功", f"已导出首帧标注 (YOLO TXT):\n图片: {img_name}")
        elif export_fmt == "YOLO Seg TXT":
            txt_save_path = os.path.join(anno_dir, img_name.replace(".jpg", ".txt"))
            save_yolo_txt(txt_save_path, self.current_frame.shape, bboxes, self.label_classes, is_seg=True)
            self.log("首帧导出成功 (YOLO Seg TXT)!")
            QMessageBox.information(self, "导出成功", f"已导出首帧标注 (YOLO Seg TXT):\n图片: {img_name}")

    def update_target_frames(self):
        if self.video_path and self.fps > 0:
            target = int(self.fps * self.duration_spin.value())
            self.frame_count_label.setText(f"目标生成: {target} 帧 (基于时间计算)")
        elif self.image_list:
            target = self.duration_spin.value()
            remaining_images = len(self.image_list) - self.current_image_index
            if target > remaining_images:
                target = remaining_images
            self.frame_count_label.setText(f"目标生成: {target} 帧 (从当前位置向后)")
        else:
            self.frame_count_label.setText("目标帧数: 待计算")

    def start_tracking(self):
        if not self.video_path and not self.image_list:
            QMessageBox.warning(self, "提示", "请先加载视频文件或图片目录！")
            return

        bboxes = self.canvas.get_bboxes()
        if not bboxes:
            QMessageBox.warning(self, "提示", "请先在当前帧标注目标")
            return

        if not self.save_dir:
            QMessageBox.warning(self, "提示", "请先设置保存目录")
            return

        user_set_val = self.duration_spin.value()

        if self.video_path:
            remaining_frames = self.total_frames - self.current_frame_idx + 1
            target_frames = int(self.fps * user_set_val)
            target_frames = min(target_frames, remaining_frames)

            start_frame_idx = self.current_frame_idx
            sub_image_list = []
            image_dir = None
        else:
            remaining_images = len(self.image_list) - self.current_image_index
            target_frames = min(user_set_val, remaining_images)

            start_frame_idx = 1
            start_idx = self.current_image_index
            sub_image_list = self.image_list[start_idx: start_idx + target_frames]
            image_dir = self.image_dir

        if target_frames <= 0:
            QMessageBox.warning(self, "提示", "已到达序列末尾，无更多帧可供标注！")
            return

        # ------------------- 关键修复点 -------------------
        # 根据当前标注模式自动决定连续追踪的默认导出格式，或提供默认
        # 如果当前是 Seg 模式，默认导出 COCO JSON 或 YOLO Seg TXT
        current_mode = self.combo_mode.currentText()
        if current_mode == "Seg分割":
            export_fmt = "COCO JSON"  # 或 "YOLO Seg TXT"
        else:
            export_fmt = "VOC XML"

        self.tracking_thread = TrackingThread(
            init_objects=bboxes,
            output_dir=self.save_dir,
            max_frames=target_frames,
            fps=self.fps if self.fps > 0 else 25,
            video_path=self.video_path,
            image_dir=image_dir,
            image_list=sub_image_list,
            start_frame_idx=start_frame_idx,
            export_format=export_fmt,  # 👈 修复：正确传入 Seg/Box 对应的格式！
            classes=self.label_classes
        )
        # --------------------------------------------------

        self.tracking_thread.progress_updated.connect(self.on_tracking_progress)
        self.tracking_thread.frame_ready.connect(self.on_tracking_frame)
        self.tracking_thread.log_message.connect(self.log)
        self.tracking_thread.finished_all.connect(self.on_tracking_finished)

        self.btn_start_track.setEnabled(False)
        self.btn_stop_track.setEnabled(True)
        self.progress_bar.setValue(0)

        self.tracking_thread.start()
        self.log(f"自动追踪线程已启动，从第 {start_frame_idx} 帧开始，目标处理 {target_frames} 帧，导出格式: {export_fmt}")

    def stop_tracking(self):
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            self.log("正在停止追踪...")

    def on_tracking_progress(self, current: int, total: int):
        progress = int(current / total * 100)
        self.progress_bar.setValue(progress)
        self.frame_num_label.setText(f"{current}/{total}")

    def on_tracking_frame(self, frame, objects: list):
        self.current_frame = frame
        self.canvas.bboxes = []
        for i, obj in enumerate(objects):
            color = self.canvas.colors[i % len(self.canvas.colors)]
            item = {
                "name": obj["name"],
                "color": color,
                "type": obj.get("type", "box"),
                "bbox": obj.get("bbox")
            }
            if obj.get("type") == "seg":
                item["data"] = obj.get("data")
            self.canvas.bboxes.append(item)

        self.canvas.color_index = len(objects)
        self.canvas.set_image(frame)

        self.bbox_list.clear()
        for obj in objects:
            if obj.get("type") == "seg":
                pts = obj.get("data", [])
                self.bbox_list.addItem(f"[Seg] {obj['name']}  [{len(pts)}顶点]")
            else:
                x, y, w, h = obj["bbox"]
                self.bbox_list.addItem(f"[Box] {obj['name']}  [{int(x)}, {int(y)}, {int(w)}×{int(h)}]")

    def on_tracking_finished(self, count: int):
        self.btn_start_track.setEnabled(True)
        self.btn_stop_track.setEnabled(False)
        self.log(f"自动追踪全部完成，共生成 {count} 帧标注数据")
        QMessageBox.information(self, "完成", f"自动追踪完成！\n共生成 {count} 帧标注数据\n保存在: {self.save_dir}")

    def prev_frame(self):
        if self.video_capture:
            new_idx = max(1, self.current_frame_idx - 1)
            self._seek_to_frame(new_idx)
        elif self.image_list:
            new_idx = max(0, self.current_image_index - 1)
            self._load_image_by_index(new_idx)

    def back_frames(self):
        if self.video_capture:
            new_idx = max(1, self.current_frame_idx - 5)
            self._seek_to_frame(new_idx)
        elif self.image_list:
            new_idx = max(0, self.current_image_index - 5)
            self._load_image_by_index(new_idx)

    def next_frame(self):
        if self.video_capture:
            new_idx = min(self.total_frames, self.current_frame_idx + 1)
            self._seek_to_frame(new_idx)
        elif self.image_list:
            new_idx = min(len(self.image_list) - 1, self.current_image_index + 1)
            self._load_image_by_index(new_idx)

    def forward_frames(self):
        if self.video_capture:
            new_idx = min(self.total_frames, self.current_frame_idx + 5)
            self._seek_to_frame(new_idx)
        elif self.image_list:
            new_idx = min(len(self.image_list) - 1, self.current_image_index + 5)
            self._load_image_by_index(new_idx)

    def toggle_play(self):
        if self.is_playing:
            self.play_timer.stop()
            self.btn_play.setText("▶")
            self.is_playing = False
        else:
            interval = int(1000 / self.fps) if self.fps > 0 else 40
            self.play_timer.start(interval)
            self.btn_play.setText("⏸")
            self.is_playing = True

    def seek_frame(self, value: int):
        if self.video_capture:
            self._seek_to_frame(value + 1)
        elif self.image_list:
            self._load_image_by_index(value)

    def _seek_to_frame(self, frame_idx: int):
        if not self.video_capture:
            return
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            self.current_frame_idx = frame_idx
            self.canvas.set_image(frame)
            self.canvas.clear_bboxes()
            self.bbox_list.clear()
            self.frame_num_label.setText(f"{frame_idx}/{self.total_frames}")
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_idx - 1)
            self.frame_slider.blockSignals(False)

    def on_scale_changed(self, text: str):
        scale_map = {
            "适应窗口": "fit",
            "100%": "100%",
            "50%": "50%",
            "200%": "200%"
        }
        self.canvas._auto_fit = (text == "适应窗口")
        self.canvas.set_scale(scale_map.get(text, "fit"))

    def closeEvent(self, event):
        if self.tracking_thread and self.tracking_thread.isRunning():
            self.tracking_thread.stop()
            self.tracking_thread.wait()

        if self.play_timer.isActive():
            self.play_timer.stop()

        if self.video_capture:
            self.video_capture.release()

        event.accept()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理与导出工具模块
包含 Pascal VOC XML 格式及标准的 Labelme JSON、YOLO Box/Seg TXT 导出函数
"""

import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom


def save_voc_xml(save_path: str, img_name: str, img_shape: tuple, tracked_objects: list):
    """
    生成标准 Pascal VOC XML 标注文件（仅导出 Box 矩形框）
    """
    h_img, w_img, depth = img_shape
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = img_name
    ET.SubElement(annotation, "path").text = save_path

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w_img)
    ET.SubElement(size, "height").text = str(h_img)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(annotation, "segmented").text = "0"

    for obj in tracked_objects:
        obj_type = obj.get("type", "box")
        # 如果是 seg，导出时跳过（根据规则：XML仅导出Box坐标）
        if obj_type == "seg":
            continue

        name = obj["name"]
        x, y, w, h = obj["bbox"] if "bbox" in obj else obj["data"]
        xmin = max(0, min(w_img - 1, int(x)))
        ymin = max(0, min(h_img - 1, int(y)))
        xmax = max(0, min(w_img, int(x + w)))
        ymax = max(0, min(h_img, int(y + h)))

        obj_item = ET.SubElement(annotation, "object")
        ET.SubElement(obj_item, "name").text = name
        ET.SubElement(obj_item, "pose").text = "Unspecified"
        ET.SubElement(obj_item, "truncated").text = "0"
        ET.SubElement(obj_item, "difficult").text = "0"

        bndbox = ET.SubElement(obj_item, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
    lines = [line for line in xml_str.split("\n") if line.strip()]
    formatted_xml = "\n".join(lines)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(formatted_xml)


def save_coco_json(save_path: str, img_name: str, img_shape: tuple, tracked_objects: list, classes: list = None):
    """
    保存为 Labelme 规范格式的 JSON 文件（支持多边形 Seg 坐标）
    """
    h_img, w_img, _ = img_shape

    shapes = []
    for obj in tracked_objects:
        obj_type = obj.get("type", "box")
        # 按照约定：JSON 导出仅导出 Seg 多边形坐标
        if obj_type == "seg":
            pts = obj.get("data", [])
            # 格式化点集坐标为浮点数 [[x1, y1], [x2, y2], ...]
            points_list = [[float(pt[0]), float(pt[1])] for pt in pts]

            shapes.append({
                "label": obj["name"],
                "points": points_list,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

    # 构建标准 JSON 根节点结构
    json_data = {
        "version": "4.5.13",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"..\\images\\{img_name}",
        "imageData": None,
        "imageHeight": int(h_img),
        "imageWidth": int(w_img)
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def save_yolo_txt(save_path: str, img_shape: tuple, tracked_objects: list, classes: list, is_seg: bool = False):
    """导出 YOLO Box 或 YOLO Seg TXT 格式"""
    h_img, w_img, _ = img_shape
    class_map = {name: i for i, name in enumerate(classes)}
    lines = []

    for obj in tracked_objects:
        cls_id = class_map.get(obj["name"], 0)
        obj_type = obj.get("type", "box")

        if is_seg and obj_type == "seg":
            pts = obj.get("data", [])
            norm_pts = []
            for px, py in pts:
                norm_pts.append(f"{px / w_img:.6f}")
                norm_pts.append(f"{py / h_img:.6f}")
            lines.append(f"{cls_id} " + " ".join(norm_pts))

        elif not is_seg and obj_type == "box":
            x, y, w, h = obj["bbox"] if "bbox" in obj else obj["data"]
            cx = (x + w / 2) / w_img
            cy = (y + h / 2) / h_img
            nw = w / w_img
            nh = h / h_img
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
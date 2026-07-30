# 🎯 SAM-AutoAnnotator

> 🚀 **SAM-AutoAnnotator** 是一款基于 **SAM 2 (Segment Anything Model 2)** 与深度视觉追踪算法（NanoTrack / DaSiamRPN）打造的高效视频/序列图像智能标注工具。
> 
> 💡 **“首帧标注，全程自动”** —— 告别逐帧标注的枯燥繁琐！支持目标检测框（Bounding Box）与像素级多边形分割（Polygon Segmentation）的高精度自动追踪，一键导出 VOC、COCO、YOLO 等主流数据集格式。

---

## ✨ 核心特性 (Key Features)

* **⚡ SAM 2 引擎驱动**：集成超轻量 ONNX Runtime 引擎（`sam2_hiera_tiny`），实现毫秒级像素分割与顶点拟合。
* **🎯 混合追踪算法**：优先加载 **NanoTrack / DaSiamRPN** 深度学习追踪器，若环境受限可自动平滑降级至传统 MIL 算法。
* **🎨 双标注模式**：
  * **Box 检测模式**：快速拉框标注目标检测矩形框。
  * **Seg 分割模式**：鼠标点选关键轮廓，双击自动拟合多边形边界。
* **📤 丰富数据导出**：支持 **Pascal VOC XML**、**COCO JSON (Labelme Standard)**、**YOLO TXT (Box/Seg)** 快速导出。
* **💻 现代化 PySide6 交互界面**：实时帧拖拽、多比例缩放、自动化进度条显示及操作日志反馈。

---

## 🛠️ 环境要求与安装 (Installation)

### 1. 简易环境准备

项目支持 **Python 3.8+** 环境：

```bash
# 1. 克隆本仓库
git clone [https://github.com/ILZFLNO02/SAM-AutoAnnotator.git](https://github.com/ILZFLNO02/SAM-AutoAnnotator.git)
cd SAM-AutoAnnotator

# 2. 安装基础依赖
pip install -r requirements.txt


📖 使用步骤 (Step-by-Step Guide)
📥 加载数据：

点击顶部 📁 图片目录读取 或 🎬 视频读取 载入素材。

💾 设置输出路径：

点击 💾 设定保存目录 指定图像与标注结果的导出目录。

🏷️ 管理标签与选择模式：

在左侧 标签类别管理 中添加目标类别（如 person, car）。

在顶部下拉框中切换 Box检测 或 Seg分割 模式。

✏️ 绘制首帧标注：

Box 模式：在画面中按住鼠标左键拖拽画框。

Seg 模式：沿目标边缘依次点击关键点，双击 完成封闭多边形构建。

⚡ 启动连续自动追踪：

设置 标注时长(秒)，点击 ▶ 开始连续标注。后台线程将自动逐帧追踪、生成标注并自动导出数据！

📁 目录结构 (Directory Structure)
Plaintext


SAM-AutoAnnotator/
├── resources/           # 模型权重文件 (ONNX / ORT)
├── main.py              # 应用主程序入口
├── ui_components.py     # GUI 界面与核心交互画布控件
├── tracker_module.py    # SAM 2 推理引擎与 Tracking 线程算法
├── utils.py             # VOC XML / COCO JSON / YOLO TXT 数据导出工具
└── requirements.txt     # 项目依赖列表
🤝 贡献与反馈 (Contributing & Feedback)
欢迎提交 Issue 或 Pull Request 为本项目改进做贡献！如果这个项目帮到了你，请点个 🌟 Star 给予鼓励支持！

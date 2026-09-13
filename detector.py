# -*- coding: utf-8 -*-
"""
detector.py —— 垃圾溢出检测的 YOLO 封装与结果统计

本模块只负责“模型 + 统计”，不涉及任何 PyQt 界面代码，方便单独测试。

类别定义（与训练时 data.yaml 一致）：
    0 -> 溢出的垃圾
    1 -> 垃圾
    2 -> 未溢出的垃圾

统一检测置信度：conf = 0.60
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------

#: 训练好的模型权重（固定路径，模块内可直接被 main.py 引用）
MODEL_PATH = r"C:\Users\李佳钰\PycharmProjects\stair_obstacle_project\runs\trash_overflow_800\weights\best.pt"

#: 统一检测置信度
CONF_THRESHOLD = 0.40

#: 推理尺寸
IMG_SIZE = 640

#: 监控站点编号（界面上 label_station 显示的就是它）
STATION_ID = "S001"

#: 类别编号 -> 中文名称
CLASS_NAMES = {
    0: "溢出的垃圾",
    1: "垃圾",
    2: "未溢出的垃圾",
}

#: 代表“垃圾溢出”的类别编号
OVERFLOW_CLASS_ID = 0


class DetectorError(Exception):
    """模型加载 / 推理过程中的可预期错误，便于界面层统一提示。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """单个检测框。"""

    class_id: int
    class_name: str
    confidence: float


@dataclass
class FrameResult:
    """一帧（或一张图片）的完整检测结果。"""

    annotated_frame: Optional[object] = None  # 已画好框的 BGR 图像（numpy.ndarray）
    detections: List[Detection] = field(default_factory=list)
    device: str = "cpu"


@dataclass
class DetectionStats:
    """对一帧检测结果的统计，界面文字全部由它派生。"""

    total: int = 0                 # 当前画面所有检测目标数量
    overflow_count: int = 0        # class 0（溢出的垃圾）数量
    class_counts: dict = field(default_factory=dict)  # 每个类别各有多少个
    max_confidence: float = 0.0    # 所有目标中的最高置信度

    @property
    def has_overflow(self) -> bool:
        """是否存在 class 0。"""
        return self.overflow_count > 0

    @property
    def max_confidence_percent(self) -> str:
        """最高置信度的百分比文本。"""
        return "{:.0f}%".format(self.max_confidence * 100)

    # ---- 界面文字（统一在这里生成，避免各处手写字符串不一致） ----

    @property
    def target_text(self) -> str:
        """label_target 的取值。"""
        if self.has_overflow:
            return "溢出的垃圾"
        if self.total > 0:
            return "垃圾"
        return "无"

    @property
    def status_text(self) -> str:
        """label_status 的取值（不含“当前状态：”前缀）。"""
        return "检测到垃圾溢出" if self.has_overflow else "未发现垃圾溢出"

    @property
    def municipal_text(self) -> str:
        """label_municipal 的取值（不含“城市管理状态：”前缀，属于模拟上报）。"""
        return "模拟上报城市管理平台" if self.has_overflow else "无需上报"

    # ---- 直接拼好的完整句子 ----

    @property
    def label_target(self) -> str:
        return "检测目标：{}".format(self.target_text)

    @property
    def label_count(self) -> str:
        return "检测数量：{}".format(self.total)

    @property
    def label_status(self) -> str:
        return "当前状态：{}".format(self.status_text)

    @property
    def label_conf(self) -> str:
        return "最高置信度：{}".format(self.max_confidence_percent)

    @property
    def label_municipal(self) -> str:
        return "城市管理状态：{}".format(self.municipal_text)


def build_stats(detections: Sequence[Detection]) -> DetectionStats:
    """把检测框列表统计成 DetectionStats。"""
    stats = DetectionStats()
    best = 0.0

    for det in detections:
        stats.total += 1
        stats.class_counts[det.class_id] = stats.class_counts.get(det.class_id, 0) + 1
        if det.class_id == OVERFLOW_CLASS_ID:
            stats.overflow_count += 1
        if det.confidence > best:
            best = det.confidence

    stats.max_confidence = best
    return stats


# ---------------------------------------------------------------------------
# 模型封装
# ---------------------------------------------------------------------------


def _resolve_device_preference() -> str:
    """优先使用 GPU；torch 不可用时退回 CPU。"""
    try:
        import torch  # noqa: WPS433 (延迟导入，避免无 torch 时直接报错)

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


class TrashOverflowDetector:
    """垃圾溢出检测器（懒加载 YOLO 模型，可重复使用）。"""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf: float = CONF_THRESHOLD,
        imgsz: int = IMG_SIZE,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz

        self._model = None
        self._device_preference = _resolve_device_preference()
        self._device_in_use = "cpu"
        self.names = dict(CLASS_NAMES)

    # ---- 属性 ----

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_in_use(self) -> str:
        return self._device_in_use

    @property
    def model_exists(self) -> bool:
        return os.path.isfile(self.model_path)

    # ---- 加载 ----

    def load(self):
        """加载模型；失败时抛 DetectorError。"""
        if self._model is not None:
            return self._model

        if not self.model_exists:
            raise DetectorError(
                "模型文件不存在：\n{}\n\n请确认已训练完成，或修改 detector.MODEL_PATH。".format(
                    self.model_path
                )
            )

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - 环境问题
            raise DetectorError(
                "无法导入 ultralytics，请先安装：pip install ultralytics\n原始错误：{}".format(exc)
            )

        try:
            self._model = YOLO(self.model_path)
        except Exception as exc:
            raise DetectorError(
                "加载 YOLO 模型失败：\n{}\n\n原始错误：{}".format(self.model_path, exc)
            )

        # 用模型自带的类别名覆盖默认表（保证与训练时完全一致）
        model_names = getattr(self._model, "names", None)
        if isinstance(model_names, dict):
            self.names = {int(k): str(v) for k, v in model_names.items()}
        elif isinstance(model_names, (list, tuple)):
            self.names = {i: str(v) for i, v in enumerate(model_names)}

        self._device_in_use = self._device_preference
        return self._model

    # ---- 推理 ----

    def _predict_once(self, source, device: str, verbose: bool = False):
        model = self.load()
        return model.predict(
            source=source,
            imgsz=self.imgsz,
            conf=self.conf,
            device=device,
            verbose=verbose,
        )

    def detect(self, image, verbose: bool = False) -> FrameResult:
        """
        对一张图片 / 一帧画面做检测。

        :param image: numpy.ndarray(BGR) 或图片路径
        :return: FrameResult（含画好框的图像与检测列表）
        """
        device = self._device_in_use
        try:
            results = self._predict_once(image, device, verbose)
        except DetectorError:
            raise
        except Exception as exc:
            # GPU 不可用 / 显存不足等 -> 退回 CPU 再试一次
            if device != "cpu":
                try:
                    results = self._predict_once(image, "cpu", verbose)
                    self._device_in_use = "cpu"
                except Exception as exc2:
                    raise DetectorError("YOLO 推理失败：{}".format(exc2))
            else:
                raise DetectorError("YOLO 推理失败：{}".format(exc))

        if not results:
            raise DetectorError("YOLO 没有返回任何结果。")

        result = results[0]

        detections: List[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            try:
                classes = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
            except Exception:
                classes = [int(c) for c in boxes.cls]
                confs = [float(c) for c in boxes.conf]

            for cls_raw, conf_raw in zip(classes, confs):
                class_id = int(cls_raw)
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=self.names.get(class_id, str(class_id)),
                        confidence=float(conf_raw),
                    )
                )

        try:
            annotated = result.plot()
        except Exception:
            annotated = image if hasattr(image, "shape") else None

        return FrameResult(
            annotated_frame=annotated,
            detections=detections,
            device=self._device_in_use,
        )

    def detect_and_stats(self, image, verbose: bool = False):
        """检测并顺便统计，返回 (FrameResult, DetectionStats)。"""
        frame_result = self.detect(image, verbose=verbose)
        return frame_result, build_stats(frame_result.detections)

# -*- coding: utf-8 -*-
"""桌面入口：界面与检测流程；MySQL 写入由 persistence.py 处理。"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFontDatabase
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QSizePolicy

from config import MODEL_PATH, CONF_THRESHOLD, STATION_ID, FROZEN, RESOURCE_ROOT
from database import utc_now
from detector import DetectorError, TrashOverflowDetector
from main_window import Ui_MainWindow
from persistence import Persistence
from theme import apply_theme, update_database_badge
from history_dialog import HistoryDialog
from database_settings import DatabaseSettingsDialog

IMAGE_FILTER = "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*.*)"
VIDEO_FILTER = "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;所有文件 (*.*)"
STREAM_INTERVAL_MS = 1
WARMUP_SIZE = 320
DISPLAY_MAX_HEIGHT_RATIO = 0.60
DISPLAY_MIN_LIMIT = 240


class ModelWarmupWorker(QThread):
    ready = pyqtSignal(str, bool, str)

    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector

    def run(self):
        try:
            self.detector.load()
            self.detector.detect(np.zeros((WARMUP_SIZE, WARMUP_SIZE, 3), dtype=np.uint8))
            self.ready.emit(self.detector.device_in_use, True, "")
        except Exception as exc:
            self.ready.emit("", False, str(exc))


class DatabaseSignals(QObject):
    status = pyqtSignal(bool, str)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        if FROZEN:
            QFontDatabase.addApplicationFont(str(RESOURCE_ROOT / 'fonts/NotoSansCJKsc-Regular.otf'))
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.history_dialog = None
        history_action = self.menuBar().addAction('历史记录')
        history_action.triggered.connect(self.show_history)
        settings_action = self.menuBar().addAction('数据库设置')
        settings_action.triggered.connect(self.show_database_settings)
        apply_theme(self)
        self.last_frame_rgb = None
        self._rendering = False
        self.setup_display_area()
        self.detector = TrashOverflowDetector(model_path=MODEL_PATH, conf=CONF_THRESHOLD)
        self.video_capture = None
        self.mode = "idle"
        self.is_open = False
        self.read_failures = 0
        self.frame_index = 0
        self.task_id = None
        self.model_ready = False
        self.warmup_worker = None
        self.stream_timer = QTimer(self)
        self.stream_timer.setSingleShot(True)
        self.stream_timer.timeout.connect(self._next_stream_frame)
        self.ui.btn_image.clicked.connect(self.on_btn_image)
        self.ui.btn_video.clicked.connect(self.on_btn_video)
        self.ui.btn_camera.clicked.connect(self.on_btn_camera)
        self.ui.btn_stop.clicked.connect(self.on_btn_stop)
        self.db_message = "MySQL 正在连接"
        self.db_signals = DatabaseSignals(self)
        self.db_signals.status.connect(self.on_database_status)
        self.persistence = Persistence(self.db_signals.status.emit)
        self.reset_result_labels()
        self.ui.label_station.setText(f"监控站点：{STATION_ID}")
        self.ui.label_display.setText("检测画面")
        if self.detector.model_exists:
            self.start_model_warmup()
        else:
            QMessageBox.warning(self, "模型文件未找到", f"检测功能暂不可用：\n{MODEL_PATH}")
            self.show_status("模型未找到")

    def show_status(self, message):
        self.statusBar().showMessage(message)

    def show_history(self):
        if self.history_dialog is None:
            self.history_dialog = HistoryDialog(self)
        self.history_dialog.show()
        self.history_dialog.raise_()
        self.history_dialog.refresh()

    def show_database_settings(self):
        dialog = DatabaseSettingsDialog(self)
        if dialog.exec_() != dialog.Accepted:
            return
        self.stop_detection(silent=True)
        if not self.persistence.close():
            self.show_status('旧连接仍在收尾，请重启程序使设置生效')
            return
        self.persistence = Persistence(self.db_signals.status.emit)
        if self.history_dialog is not None and not self.history_dialog.is_loading():
            self.history_dialog.refresh()

    def on_database_status(self, success, message):
        self.db_message = message
        update_database_badge(self, success, message)
        self.statusBar().showMessage(message)

    def start_model_warmup(self):
        self.set_buttons_running(True)
        self.ui.label_status.setText("当前状态：正在加载模型…")
        self.show_status("正在后台加载 YOLO 模型，请稍候…")
        self.warmup_worker = ModelWarmupWorker(self.detector, self)
        self.warmup_worker.ready.connect(self.on_model_warmup_finished)
        self.warmup_worker.start()

    def on_model_warmup_finished(self, device, success, message):
        self.model_ready = success
        self.set_buttons_running(False)
        if self.ui.label_status.text() == "当前状态：正在加载模型…":
            self.ui.label_status.setText("当前状态：未检测")
        if success:
            self.show_status(f"模型就绪 | 设备 {device} | conf={CONF_THRESHOLD:.2f} | 站点 {STATION_ID}")
        else:
            self.show_status("模型加载失败，检测功能不可用")
            QMessageBox.critical(self, "模型加载失败", message)

    def setup_display_area(self) -> None:
        try:
            container = self.ui.displayContainer
            label = self.ui.label_display

            container.setMinimumSize(0, 0)
            label.setMinimumSize(1, 1)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            main_layout = self.ui.centralwidget.layout()
            if main_layout is not None:
                # 只让中间区域吸收多余高度，标题/按钮/状态保持自然高度
                main_layout.setStretch(1, 1)

            self.apply_display_height_limit()
        except Exception:
            # 布局微调失败不影响任何检测功能
            pass

    def apply_display_height_limit(self) -> None:
        """按窗口高度动态限制画面区高度，避免检测画面过分膨胀。"""
        try:
            limit = max(DISPLAY_MIN_LIMIT, int(self.height() * DISPLAY_MAX_HEIGHT_RATIO))
            self.ui.label_display.setMaximumHeight(limit)
            self.ui.displayContainer.setMaximumHeight(limit)
        except Exception:
            pass

    def reset_result_labels(self) -> None:
        self.ui.label_result.setText("检测结果")
        self.ui.label_target.setText("检测目标：--")
        self.ui.label_count.setText("检测数量：--")
        self.ui.label_status.setText("当前状态：未检测")
        self.ui.label_conf.setText("最高置信度：--")
        self.ui.label_municipal.setText("城市管理状态：未上报")

    def update_result_labels(self, stats) -> None:
        """用一帧的统计结果刷新右侧信息区。"""
        self.ui.label_target.setText(stats.label_target)
        self.ui.label_count.setText(stats.label_count)
        self.ui.label_status.setText(stats.label_status)
        self.ui.label_conf.setText(stats.label_conf)
        self.ui.label_municipal.setText(stats.label_municipal)

    def show_frame(self, frame_bgr) -> None:
        if frame_bgr is None:
            return

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return

        self.last_frame_rgb = frame_rgb
        self._render_last_frame()

    def _render_last_frame(self) -> None:
        """把最近一帧按 label_display 当前大小缩放显示。"""
        if self.last_frame_rgb is None:
            return

        frame = self.last_frame_rgb
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        q_image = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format_RGB888
        )

        label = self.ui.label_display
        target_size = label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = label.minimumSize()

        pixmap = QPixmap.fromImage(q_image)
        if target_size.width() > 0 and target_size.height() > 0:
            pixmap = pixmap.scaled(
                target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        self._rendering = True
        try:
            label.setPixmap(pixmap)
        finally:
            self._rendering = False

    def resizeEvent(self, event):  # noqa: N802 (Qt 命名风格)
        """窗口缩放时：调整画面区高度上限，并把最近一帧重新缩放显示。"""
        super().resizeEvent(event)
        self.apply_display_height_limit()
        if not self._rendering and self.last_frame_rgb is not None:
            self._render_last_frame()

    def set_buttons_running(self, running: bool) -> None:
        """运行中禁止重复点检测按钮，停止按钮始终可用。"""
        for button in (self.ui.btn_image, self.ui.btn_video, self.ui.btn_camera):
            button.setEnabled(not running)
        self.ui.btn_stop.setEnabled(True)

    def ensure_model_ready(self) -> bool:
        if self.model_ready:
            return True
        QMessageBox.information(
            self,
            "模型正在加载",
            "YOLO 模型仍在后台加载中，请稍等几秒再点击检测按钮。",
        )
        return False

    def begin_task(self, mode, source):
        self.mode = mode
        self.frame_index = 0
        self.task_id = str(uuid4())
        self.persistence.submit("start", self.task_id, mode, str(source), self.detector.model_path,
                                self.detector.conf, self.detector.imgsz, utc_now())
        self.reset_result_labels()
        self.set_buttons_running(True)

    def process_frame(self, frame):
        """三个入口共用检测、显示和持久化逻辑。"""
        try:
            result, stats = self.detector.detect_and_stats(frame)
        except Exception as exc:
            self.stop_detection(silent=True, status="failed", error=str(exc))
            self.ui.label_status.setText("当前状态：检测失败")
            self.show_status("检测失败")
            title = "检测失败" if isinstance(exc, DetectorError) else "检测异常"
            QMessageBox.critical(self, title, str(exc))
            return False
        self.show_frame(result.annotated_frame if result.annotated_frame is not None else frame)
        self.update_result_labels(stats)
        media_ms = None
        if self.mode == "video":
            media_ms = max(0.0, self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
        clock = media_ms / 1000 if media_ms is not None else time.monotonic()
        objects = [(obj.class_id, obj.class_name, obj.confidence) for obj in result.detections]
        snapshot = None
        if stats.has_overflow:
            try:
                picture = result.annotated_frame if result.annotated_frame is not None else frame
                ok, encoded = cv2.imencode('.jpg', picture)
                if ok:
                    snapshot = encoded.tobytes()
            except Exception:
                pass
        self.persistence.submit("frame", self.task_id, self.frame_index, media_ms, utc_now(),
                                clock, result.device, objects, stats, snapshot)
        self.frame_index += 1
        name = {"image": "图片检测完成", "video": "视频检测中", "camera": "摄像头检测中"}[self.mode]
        self.show_status(f"{name} | 目标 {stats.total} 个 | 溢出 {stats.overflow_count} 个 | 设备 {result.device}")
        return True

    def on_btn_image(self):
        if not self.ensure_model_ready():
            return
        self.stop_detection(silent=True)
        path, _ = QFileDialog.getOpenFileName(self, "选择要检测的图片", "", IMAGE_FILTER)
        if not path:
            return
        try:
            frame = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = None
        if frame is None:
            QMessageBox.critical(self, "图片读取失败", f"无法读取该图片：\n{path}")
            return
        self.begin_task("image", path)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.process_frame(frame):
                self.stop_detection(keep_status=True, silent=True, status="completed")
        finally:
            QApplication.restoreOverrideCursor()

    def on_btn_video(self):
        if not self.ensure_model_ready():
            return
        self.stop_detection(silent=True)
        path, _ = QFileDialog.getOpenFileName(self, "选择要检测的视频", "", VIDEO_FILTER)
        if path:
            self.start_stream("video", path)

    def on_btn_camera(self):
        if self.ensure_model_ready():
            self.stop_detection(silent=True)
            self.start_stream("camera", 0)

    def start_stream(self, mode, source):
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            name = "视频" if mode == "video" else "摄像头"
            QMessageBox.critical(self, f"{name}打开失败", f"无法打开{name}：{source}")
            return
        self.video_capture = capture
        self.is_open = True
        self.read_failures = 0
        self.begin_task(mode, source)
        self._schedule_stream_frame()

    def _schedule_stream_frame(self):
        if self.is_open and self.mode in ("video", "camera"):
            self.stream_timer.start(STREAM_INTERVAL_MS)

    def _next_stream_frame(self):
        if not self.is_open or self.video_capture is None:
            return
        ok, frame = self.video_capture.read()
        if not ok or frame is None:
            if self.mode == "video":
                self.stop_detection(silent=True, status="completed")
                self.ui.label_status.setText("当前状态：视频检测结束")
                self.show_status("视频检测结束，资源已释放")
                QMessageBox.information(self, "视频检测结束", "视频已播放并检测完毕。")
            else:
                self.read_failures += 1
                if self.read_failures >= 30:
                    self.stop_detection(silent=True, status="failed", error="摄像头连续读帧失败")
                    QMessageBox.critical(self, "摄像头读取失败", "连续多次无法读取摄像头画面，已停止检测。")
                else:
                    self._schedule_stream_frame()
            return
        self.read_failures = 0
        if self.process_frame(frame):
            self._schedule_stream_frame()

    def stop_detection(self, keep_status=False, silent=False, status="stopped", error=None):
        self.stream_timer.stop()
        capture, self.video_capture = self.video_capture, None
        if capture is not None:
            capture.release()
        if self.task_id is not None:
            self.persistence.submit("finish", self.task_id, status, utc_now(), error)
            self.task_id = None
        self.mode = "idle"
        self.is_open = False
        self.set_buttons_running(False)
        if not silent:
            if not keep_status:
                self.reset_result_labels()
            self.ui.label_status.setText("当前状态：已停止")
            self.show_status("检测已停止，资源已释放")

    def on_btn_stop(self):
        self.stop_detection()

    def closeEvent(self, event):
        if self.history_dialog is not None and self.history_dialog.is_loading():
            event.ignore()
            self.show_status('历史查询正在完成，请稍后关闭')
            return
        self.stop_detection(silent=True)
        if self.warmup_worker is not None and self.warmup_worker.isRunning():
            if not self.warmup_worker.wait(5000):
                event.ignore()
                self.show_status("模型仍在加载，请稍后关闭窗口")
                return
        if not self.persistence.close():
            event.ignore()
            self.show_status("数据库正在收尾，请稍后再次关闭窗口")
            return
        super().closeEvent(event)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

"""有界后台写入队列：数据库不可用时检测仍可运行，明确通知保存失败。"""
from queue import Queue, Full
from threading import Thread, BoundedSemaphore
from pathlib import Path

from config import SAVE_INTERVAL, EVENT_CLEAR_SECONDS, ROOT
from database import Database, connection_error_message


class Persistence:
    def __init__(self, notify, database_factory=Database):
        self.notify = notify
        self.database_factory = database_factory
        # 帧数据最多 256 条；任务开始/结束不受帧拥塞影响，以免任务无法收尾。
        self.queue = Queue()
        self.frame_slots = BoundedSemaphore(256)
        self.closed = False
        self.stop_queued = False
        self.thread = Thread(target=self._run, name='mysql-writer', daemon=True)
        self.thread.start()

    def submit(self, operation, *args):
        if self.closed:
            return False
        if operation == 'frame' and not self.frame_slots.acquire(blocking=False):
            self.notify(False, '数据库保存队列已满，本次记录未保存')
            return False
        self.queue.put_nowait((operation, args))
        return True

    def _run(self):
        database = None
        tasks = {}
        try:
            try:
                database = self.database_factory()
                self.notify(True, 'MySQL 已连接')
            except Exception as exc:
                self.notify(False, connection_error_message(exc))
            while True:
                item = self.queue.get()
                try:
                    if item is None:
                        break
                    operation, args = item
                    if database is None:
                        continue
                    if operation == 'start':
                        database.start_task(*args)
                        tasks[args[0]] = dict(last_saved=None, overflow=None, clear_since=None,
                                              event_id=None)
                    elif operation == 'frame':
                        task_id, index, media_ms, timestamp, clock, device, objects, stats = args[:8]
                        snapshot_bytes = args[8] if len(args) > 8 else None
                        state = tasks[task_id]
                        if stats.has_overflow:
                            state['clear_since'] = None
                        elif state['clear_since'] is None:
                            state['clear_since'] = clock
                        recover = (state['event_id'] is not None and not stats.has_overflow
                                   and clock - state['clear_since'] >= EVENT_CLEAR_SECONDS)
                        due = (state['last_saved'] is None or clock - state['last_saved'] >= SAVE_INTERVAL
                               or state['overflow'] != stats.has_overflow or recover)
                        if due:
                            snapshot_path = None
                            snapshot_failed = False
                            if stats.has_overflow and state['event_id'] is None and snapshot_bytes:
                                try:
                                    folder = ROOT / 'storage' / 'snapshots'
                                    folder.mkdir(parents=True, exist_ok=True)
                                    path = folder / f'{task_id}_{index}.jpg'
                                    path.write_bytes(snapshot_bytes)
                                    snapshot_path = str(path)
                                except OSError:
                                    snapshot_failed = True
                            event_id = database.save_result(task_id, index, media_ms, timestamp,
                                                            device, objects, stats, state['event_id'], recover,
                                                            **({'snapshot_path': snapshot_path} if snapshot_path else {}))
                            state.update(event_id=event_id, last_saved=clock)
                            self.notify(True, '检测记录已保存到 MySQL')
                            if snapshot_failed:
                                self.notify(False, '检测记录已保存，但预警截图写入失败，请检查目录权限')
                        state['overflow'] = stats.has_overflow
                    elif operation == 'finish':
                        database.finish_task(*args)
                        tasks.pop(args[0], None)
                except Exception:
                    # 不自动重放：网络中断时提交状态可能不确定，避免静默重复。
                    self.notify(False, 'MySQL 保存失败，本次记录未确认保存；请检查连接后重启程序')
                finally:
                    if item is not None and item[0] == 'frame':
                        self.frame_slots.release()
                    self.queue.task_done()
        finally:
            if database is not None:
                database.close()

    def close(self, timeout=5):
        self.closed = True
        if not self.stop_queued:
            try:
                self.queue.put(None, timeout=timeout)
                self.stop_queued = True
            except Full:
                return False
        self.thread.join(timeout)
        return not self.thread.is_alive()

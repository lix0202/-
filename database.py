"""MySQL 操作；每个 Database 实例仅由创建它的线程使用。"""
from contextlib import contextmanager
from datetime import datetime, timezone

from config import RESOURCE_ROOT, STATION_ID, mysql_options


def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def connection_error_message(error):
    """只展示可操作的原因，不输出连接凭据或原始连接字符串。"""
    code = getattr(error, 'errno', None)
    reasons = {
        1045: '用户名或密码错误，或该账号不允许从当前地址登录',
        1044: '当前账号没有访问项目数据库的权限',
        1049: '数据库不存在，请先创建 garbage_overflow（或配置的数据库）',
        1146: '数据表尚未创建，请在“数据库设置”中创建数据库和表',
        2003: '无法连接 MySQL，请检查服务、地址和端口',
        2005: '数据库地址无法解析，请检查 GOW_DB_HOST',
        2013: 'MySQL 连接中断，请检查服务和网络',
    }
    if isinstance(error, ModuleNotFoundError):
        reason = '当前 Python 环境缺少 MySQL 驱动，请安装 requirements-database.txt'
    elif isinstance(error, ValueError):
        reason = '环境变量未配置或格式不正确，请检查 GOW_DB_USER、GOW_DB_PASSWORD 和端口'
    elif isinstance(error, RuntimeError):
        reason = '数据库结构尚未初始化，请在“数据库设置”中创建数据库和表'
    else:
        reason = reasons.get(code, f'连接失败（错误代码 {code if code is not None else type(error).__name__}），请运行连接检查脚本')
    return 'MySQL 未连接：' + reason


class Database:
    def __init__(self, check_schema=True):
        import mysql.connector
        self.connection = mysql.connector.connect(**mysql_options())
        try:
            if check_schema:
                with self.transaction() as cursor:
                    cursor.execute('SELECT version FROM schema_migrations WHERE version=1')
                    if cursor.fetchone() is None:
                        raise RuntimeError('请先运行 python scripts/init_database.py')
        except Exception:
            self.close()
            raise

    @contextmanager
    def transaction(self):
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def initialize(self):
        # MySQL DDL 隐式提交；失败时可以修复原因后重新运行，无删除操作。
        sql = (RESOURCE_ROOT / 'migrations/001_initial.sql').read_text(encoding='utf-8')
        with self.transaction() as cursor:
            for statement in sql.split(';'):
                if statement.strip():
                    cursor.execute(statement)
        sql = (RESOURCE_ROOT / 'migrations/002_snapshots.sql').read_text(encoding='utf-8')
        with self.transaction() as cursor:
            for statement in sql.split(';'):
                if statement.strip():
                    cursor.execute(statement)

    def history(self, only_overflow=False):
        with self.transaction() as cursor:
            cursor.execute('SELECT r.id,r.detected_at,s.code,t.source_type,r.total_count,'
                           'r.overflow_count,r.max_confidence,t.status,p.path '
                           'FROM detection_results r JOIN detection_tasks t ON t.id=r.task_id '
                           'JOIN stations s ON s.id=t.station_id '
                           'LEFT JOIN warning_snapshots p ON p.result_id=r.id '
                           + ('WHERE r.overflow_count>0 ' if only_overflow else '')
                           + 'ORDER BY r.id DESC LIMIT 100')
            return cursor.fetchall()

    def start_task(self, task_id, source_type, source_uri, model_path, conf, imgsz, started_at):
        with self.transaction() as cursor:
            cursor.execute('INSERT INTO stations (code,name,created_at) VALUES (%s,%s,%s) '
                           'ON DUPLICATE KEY UPDATE code=code', (STATION_ID, STATION_ID, started_at))
            cursor.execute('SELECT id FROM stations WHERE code=%s', (STATION_ID,))
            station_id = cursor.fetchone()[0]
            cursor.execute('INSERT INTO detection_tasks '
                           '(id,station_id,source_type,source_uri,model_path,confidence_threshold,'
                           'image_size,started_at,status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                           (task_id, station_id, source_type, source_uri, model_path, conf, imgsz,
                            started_at, 'running'))

    def save_result(self, task_id, frame_index, media_ms, timestamp, device, objects, stats,
                    event_id, recover, snapshot_path=None):
        with self.transaction() as cursor:
            cursor.execute('INSERT INTO detection_results '
                           '(task_id,frame_index,media_time_ms,detected_at,total_count,overflow_count,'
                           'max_confidence,device) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
                           (task_id, frame_index, media_ms, timestamp, stats.total,
                            stats.overflow_count, stats.max_confidence if stats.total else None, device))
            result_id = cursor.lastrowid
            if snapshot_path:
                cursor.execute('INSERT INTO warning_snapshots (result_id,path) VALUES (%s,%s)',
                               (result_id, snapshot_path))
            if objects:
                cursor.executemany('INSERT INTO detection_objects '
                                   '(result_id,class_id,class_name,confidence) VALUES (%s,%s,%s,%s)',
                                   [(result_id, *obj) for obj in objects])
            if stats.has_overflow:
                if event_id is None:
                    cursor.execute('INSERT INTO overflow_events '
                                   '(task_id,first_result_id,last_result_id,started_at,last_seen_at) '
                                   'VALUES (%s,%s,%s,%s,%s)',
                                   (task_id, result_id, result_id, timestamp, timestamp))
                    event_id = cursor.lastrowid
                else:
                    cursor.execute('UPDATE overflow_events SET last_result_id=%s,last_seen_at=%s '
                                   'WHERE id=%s', (result_id, timestamp, event_id))
            elif recover and event_id is not None:
                cursor.execute("UPDATE overflow_events SET ended_at=%s,end_reason='recovered' WHERE id=%s",
                               (timestamp, event_id))
                event_id = None
        return event_id

    def finish_task(self, task_id, status, timestamp, error=None):
        with self.transaction() as cursor:
            cursor.execute('UPDATE detection_tasks SET ended_at=%s,status=%s,error_message=%s WHERE id=%s',
                           (timestamp, status, error, task_id))
            cursor.execute('UPDATE overflow_events SET ended_at=%s,end_reason=%s '
                           'WHERE task_id=%s AND ended_at IS NULL', (timestamp, status, task_id))

    def close(self):
        self.connection.close()

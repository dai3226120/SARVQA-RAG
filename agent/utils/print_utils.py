"""
打印与格式化工具函数
线程安全的打印、时间格式化、分隔线等
"""

from .thread_lock import lock


def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with lock:
        print(*args, **kwargs)


def format_elapsed_time(seconds):
    """格式化耗时"""
    if seconds < 0:
        return "计算中..."
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}时{minutes}分{secs:.1f}秒"
    if minutes > 0:
        return f"{minutes}分{secs:.1f}秒"
    return f"{secs:.2f}秒"


def print_separator(title=None, char="="):
    """打印分隔线"""
    if title:
        safe_print(f"{char} {title} {char}")
        safe_print(char * 80)
    else:
        safe_print(char * 80)


def print_status(message, status="info"):
    """打印状态信息"""
    status_map = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "debug": "🔧"
    }
    safe_print(f"{status_map.get(status, 'ℹ️')} {message}")


def print_section(title):
    """打印章节标题"""
    safe_print("")
    safe_print("-" * 80)
    safe_print(f"{title}")
    safe_print("-" * 80)

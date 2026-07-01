"""
线程锁模块
提供全局线程锁，供 benchmark 与 agent 共用
"""

import threading

lock = threading.Lock()

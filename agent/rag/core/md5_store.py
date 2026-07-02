"""
统一的 MD5 校验与记录模块
合并自原 vector_store.py / rag_rscsv_builder.py / rag_rsfit_builder.py 三处重复实现
"""
import os
from utils.logger_handler import logger


class Md5Store:
    """
    文件 MD5 哈希校验与持久化记录
    用于增量处理场景：判断源文件是否已变更，避免重复入库
    """

    def __init__(self, store_path: str):
        """
        Args:
            store_path: MD5 记录文件的绝对路径
        """
        self._store_path = store_path
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """确保记录文件存在"""
        if not os.path.exists(self._store_path):
            os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
            open(self._store_path, "w", encoding="utf-8").close()
            logger.debug(f"[Md5Store] 创建空记录文件: {self._store_path}")

    def is_registered(self, md5_hex: str) -> bool:
        """
        检查给定 MD5 值是否已存在于记录文件中

        Args:
            md5_hex: 待校验的 MD5 十六进制字符串

        Returns:
            True 表示已存在
        """
        if not md5_hex:
            return False
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                return any(line.strip() == md5_hex for line in f.readlines())
        except Exception as e:
            logger.error(f"[Md5Store] 读取记录文件失败: {e}")
            return False

    def register(self, md5_hex: str) -> None:
        """
        将新的 MD5 值追加写入记录文件

        Args:
            md5_hex: 待记录的 MD5 十六进制字符串
        """
        if not md5_hex:
            return
        with open(self._store_path, "a", encoding="utf-8") as f:
            f.write(md5_hex + "\n")
        logger.debug(f"[Md5Store] 已记录 MD5: {md5_hex[:8]}...")

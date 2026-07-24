import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_dir: Path, log_level: str = "INFO"):
    """
    系统日志初始化函数。
    必须由入口程序(如 app.py)在启动时显式调用。

    :param log_dir: 日志存放的绝对路径，由调用方传入，实现彻底解耦
    :param log_level: 控制台输出的日志级别
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # 移除默认配置
    logger.remove()

    # 1. 控制台输出
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # 2. 文件输出 (保留 30 天，午夜轮转)
    logger.add(
        log_dir / "system_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG",  # 文件里永远存最全的 DEBUG 信息，方便溯源
        encoding="utf-8",
        enqueue=True,  # 开启异步写入，防阻塞
    )

    logger.success("🚀 Loguru 日志系统引导完成！")


# 导出原始的 logger 和初始化函数
__all__ = ["logger", "setup_logger"]

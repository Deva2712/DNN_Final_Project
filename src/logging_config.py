"""
Logging Configuration

Sets up comprehensive logging for the entire project with appropriate handlers,
formatters, and log levels for different components.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "outputs/training_logs/experiment.log",
    console_output: bool = True,
    debug_file: Optional[str] = None
):
    """
    Configure comprehensive logging for the project.
    
    Sets up:
    - File handler for all logs (DEBUG and above)
    - Console handler for INFO and above (configurable)
    - Optional separate DEBUG file for detailed diagnostics
    - Appropriate formatters with timestamps
    - Reduced verbosity for external libraries
    
    Args:
        log_level: Console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to main log file
        console_output: Whether to output logs to console
        debug_file: Optional path to separate DEBUG log file
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create detailed formatter for file logs
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create simpler formatter for console
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - captures DEBUG and above
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Optional separate DEBUG file for detailed diagnostics
    if debug_file:
        debug_path = Path(debug_file)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_handler = logging.FileHandler(debug_file, mode='a')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        logger.addHandler(debug_handler)
    
    # Console handler - configurable level
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Reduce verbosity of external libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('torchvision').setLevel(logging.WARNING)
    
    logger.info("=" * 80)
    logger.info("Logging configured successfully")
    logger.info(f"Console log level: {log_level}")
    logger.info(f"Main log file: {log_file}")
    if debug_file:
        logger.info(f"Debug log file: {debug_file}")
    logger.info("=" * 80)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Name of the module (typically __name__)
    
    Returns:
        logger: Logger instance
    """
    return logging.getLogger(name)


def log_system_info():
    """
    Log system and environment information for reproducibility.
    """
    import torch
    import platform
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("System Information")
    logger.info("=" * 80)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 80)

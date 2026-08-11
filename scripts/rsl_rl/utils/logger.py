import logging
from logging.handlers import RotatingFileHandler

from pathlib import Path

import os

import logging
import sys

def redirect_python_streams(app_logger):
    sys.stdout = StreamToLogger(
        app_logger,
        logging.INFO,
        sys.__stdout__,
    )

    sys.stderr = StreamToLogger(
        app_logger,
        logging.WARNING,
        sys.__stderr__,
    )
    
    
class StreamToLogger:
    """print/stdout/stderr 출력을 Python logger로 전달."""

    def __init__(self, logger, level, original_stream):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream
        self.buffer = ""

    def write(self, message):
        if not message:
            return 0

        # 진행률 출력의 \r도 한 줄로 처리
        self.buffer += message.replace("\r", "\n")

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)

            if line.strip():
                self.logger.log(self.level, line.rstrip())

        return len(message)

    def flush(self):
        if self.buffer.strip():
            self.logger.log(self.level, self.buffer.rstrip())
            self.buffer = ""

        for handler in self.logger.handlers:
            handler.flush()

    def isatty(self):
        return self.original_stream.isatty()

    def fileno(self):
        return self.original_stream.fileno()

# create_logger("phase1", "~/logs")
def create_logger(
    name: str,
    log_directory,
    log_cfgs: str
):
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    full_dir = os.path.join(log_directory, f"{name}.log")
    handler_configs = log_cfgs['handlers']
    logger.propagate = False
    
    for handler_config in handler_configs.values():
        if handler_config['class'] == 'logging.StreamHandler':
            handler = logging.StreamHandler()
        elif handler_config['class'] == 'logging.handlers.RotatingFileHandler':
            handler = logging.handlers.RotatingFileHandler(
                filename = full_dir,
                maxBytes = handler_config.get('maxBytes', 0),
                backupCount=handler_config.get('backupCount', 0)
            )
    
        formatter_config = log_cfgs['formatters'][handler_config['formatter']]
        formatter = logging.Formatter(
            fmt=formatter_config['format'],
            datefmt=formatter_config['datefmt']
        )
        handler.setFormatter(formatter)
        handler.setLevel(handler_config['level'])
        
        # Add the handler to the logger
        logger.addHandler(handler) 
    
    logger.setLevel(log_cfgs['root']['level'])
    
    
    return logger
    
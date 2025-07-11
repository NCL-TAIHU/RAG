import logging

def setup_logger(name, log_file = None, console = True, file = True, level=logging.INFO):
    if file is True: assert log_file is not None, "log_file must be specified if file is True"
    logger = logging.getLogger(name)
    logger.setLevel(level) 
    # ✨ Formatter with filename and line number
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )      # 🔁 Remove all existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if file: 
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
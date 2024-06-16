import logging
import sys, os

def set_logger_level(logger, level):
    levels = {
            'DEBUG': logging.DEBUG,
            'INFO' : logging.INFO,
            }
    if level not in levels:
        raise ValueError(f'Not implemented logger level {level}')
    logger.setLevel(levels[level])
    return 

def create_logger(save_folder, debug=True): 
    log_level = logging.DEBUG if debug else logging.INFO
    os.makedirs(save_folder, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    log_formatter = logging.Formatter("%(asctime)s [ %(levelname)-5.5s]:  %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    filepath = os.path.join(save_folder,'out.log')
    if os.path.exists(filepath):
        os.remove(filepath)
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(log_formatter)
    if (root_logger.hasHandlers()):
        root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return root_logger



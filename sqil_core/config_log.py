import logging

from colorama import Fore, Style, init

# Initialize Colorama
init(autoreset=True, strip=False, convert=False)


class SqilFormatter(logging.Formatter):
    FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    COLOR_MAP = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.COLOR_MAP.get(record.levelname, "")
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


class SqilLogger(logging.Logger):
    # By default show the stack trace when errors are logged
    def error(self, msg, *args, exc_info=True, **kwargs):
        super().error(msg, *args, exc_info=exc_info, **kwargs)


logging.setLoggerClass(SqilLogger)
logger = logging.getLogger("sqil_logger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(SqilFormatter(SqilFormatter.FORMAT))

# Avoid adding multiple handlers if the logger is reused
if not logger.hasHandlers():
    logger.addHandler(console_handler)

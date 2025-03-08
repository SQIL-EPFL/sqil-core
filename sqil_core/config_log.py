import logging

from colorama import Fore, Style, init

# Initialize Colorama
init(autoreset=True, strip=False, convert=False)


class CustomFormatter(logging.Formatter):
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


# Create and configure the logger
logger = logging.getLogger("sqil_logger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(CustomFormatter(CustomFormatter.FORMAT))

# Avoid adding multiple handlers if the logger is reused
if not logger.hasHandlers():
    logger.addHandler(console_handler)

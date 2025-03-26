import logging
import colorlog


class Logger:
    """Class to handle all logging functionality"""

    def __init__(self):
        """Initialize logger with color formatting"""
        handler = colorlog.StreamHandler()

        # Custom formatter class to handle complex color formatting
        class CustomColorFormatter(colorlog.ColoredFormatter):
            def format(self, record):
                # Format the message with standard ColoredFormatter
                formatted = super().format(record)

                # Add green color to keys
                formatted = formatted.replace("line_no=", "\033[32mline_no=\033[0m")
                formatted = formatted.replace("time=", "\033[32mtime=\033[0m")

                return formatted

        formatter = CustomColorFormatter(
            '%(log_color)s[%(levelname)s]\033[0m\t \033[95m%(message)s\033[0m\t %(white)sline_no=%(lineno)d\t time=%(asctime)s\t', log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={
                'white': {
                    'DEBUG': 'white',
                    'INFO': 'white',
                    'WARNING': 'white',
                    'ERROR': 'white',
                    'CRITICAL': 'white',
                }
            },
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        # Remove any non-colorlog handlers
        for hdlr in self.logger.handlers[:]:
            if not isinstance(hdlr, colorlog.StreamHandler):
                self.logger.removeHandler(hdlr)

    def info(self, message):
        """Log info level message"""
        self.logger.info(message)

    def error(self, message):
        """Log error level message"""
        self.logger.error(message)

    def warning(self, message):
        """Log warning level message"""
        self.logger.warning(message)

    def debug(self, message):
        """Log debug level message"""
        self.logger.debug(message)

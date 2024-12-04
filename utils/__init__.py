from termcolor import cprint, colored
from .timer import Timer, timer

class Logger:
    @staticmethod
    def INFO(msg, **kwargs):
        cprint(f'INFO: {msg}', color='green', **kwargs)
    
    @staticmethod
    def WARN(msg, **kwargs):
        cprint(f'WARN: {msg}', color='yellow', **kwargs)

    @staticmethod
    def ERROR(msg, raise_exception=True, **kwargs):
        if raise_exception:
            raise Exception(colored(msg, color='red'))
        else:
            cprint(f'ERROR: {msg}', color='red', **kwargs)

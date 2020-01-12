from contextlib import contextmanager
from logging import getLogger, StreamHandler, FileHandler, Formatter
from time import time


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


def get_logger(name, log_level="DEBUG", output_file=None, handler_level="INFO", format_str="[%(asctime)s] %(message)s"):
    """
    :param str name:
    :param str log_level:
    :param str | None output_file:
    :return: logger
    """
    logger = getLogger(name)

    formatter = Formatter(format_str)

    handler = StreamHandler()
    logger.setLevel(log_level)
    handler.setLevel(handler_level)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if output_file:
        file_handler = FileHandler(output_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(handler_level)
        logger.addHandler(file_handler)

    return logger


def class_to_dict(cls):
    def props(cls):
        return [[i, getattr(cls, i)] for i in cls.__dict__.keys() if i[:1] != '_']

    return dict(props(cls))


def calculate_original_img_size(origin_size: int, upscale_factor: int) -> int:
    """
    元の画像サイズを縮小拡大したいときに元の画像をどの大きさに
    resize する必要があるかを返す関数

    例えば 202 px の画像を 1/3 に縮小することは出来ない(i.e. 3の倍数ではない)ので
    事前に 201 px に縮小しておく必要がありこの関数はその計算を行う
    すなわち

    calculate_original_img_size(202, 3) -> 201

    となる

    Args:
        origin_size:
        upscale_factor:

    Returns:

    """
    return origin_size - (origin_size % upscale_factor)


def get_model_device(model):
    return next(model.parameters()).device

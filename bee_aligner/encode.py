r""" encode(text: List[str], pt="laser"). """

from typing import List, Optional, Union

# import sys
# from pathlib import Path
import warnings

import numpy as np

import logzero
from logzero import logger

from diskcache import FanoutCache

# from bee_aligner import spinner_start, spinner_stop
# from bee_aligner.bumblebee import bumblebee  # noqa: F401
# from bee_aligner.bumblebee1024 import bumblebee1024  # noqa: F401
from bee_aligner.detect_lang_pg import detect_lang

warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=SourceChangeWarning)

cache = FanoutCache("diskcache")


# fmt: off
# cache affects tests, beware.
# should comment out until the last moment
@cache.memoize(typed=True, expire=36000, tag='bee-aligner-encode')
def encode(
        text: Union[str, List[str]],
        lang: Optional[str] = None,  # default to auto
        pt: str = "laser",
        debug: Optional[bool] = False,
) -> np.ndarray:
    # fmt: on
    """ encode Union[str, list]: text -> np.ndarray.

    if pt == "laser", lang = the language code(s) (ISO 639-1)
    bert: lang not needed

    base: 120ms
    4500 chars: 66s
    2400        32
    1000      : 15s
    100       : 3.6s
    200       : 4s
    400         6
    """

    if debug:
        logzero.loglevel(10)
    else:
        logzero.loglevel(20)

    if isinstance(text, str):
        text = [text]

    if pt in ["laser"]:
        if lang is None:
            lang = detect_lang(" ".join(text))
            logger.info(" Detectec language: %s", lang)
        # bee = importlib.import_module("bee_aligner.bumblebee1024")
        # bee1024 = bee.bumblebee1024()

        # from bee_aligner.bumblebee1024 import bumblebee1024
        # bee1024 = bumblebee1024(debug=debug)
        from bee_aligner.bumblebee1024 import bee1024

        res = bee1024.embed_sentences(text, [lang] * len(text))
    else:
        # bee = importlib.import_module("bee_aligner.bumblebee")
        # bee512 = bee.bumblebee()
        from bee_aligner.bumblebee import bumblebee

        bee512 = bumblebee(debug=debug)
        res = bee512.encode(text)
    return res

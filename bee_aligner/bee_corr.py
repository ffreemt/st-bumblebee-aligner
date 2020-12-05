""" compute cos_mat (correlation matrix)
"""
from typing import List, Optional

import numpy as np

# import torch
from diskcache import FanoutCache

import logzero
from logzero import logger

# from bee_aligner import SPINNER, spinner_start, spinner_stop
from bee_aligner.cos_matrix2 import cos_matrix2
from bee_aligner.encode import encode
from bee_aligner.detect_lang_pg import detect_lang

cache = FanoutCache("diskcache")  # pylint: disable=invalid-name


# fmt: off
@cache.memoize(typed=True, expire=36000, tag='bee_corr')
def bee_corr(  # pylint: disable=invalid-name, too-many-arguments, too-many-branches, bad-continuation
        src_text: List[str],
        tgt_text: List[str],
        src_lang: Optional[str] = None,  # default to auto
        tgt_lang: Optional[str] = None,  # default to auto
        debug: Optional[bool] = False,
        pt: str = "laser",  # 'bert', 'laser' (default)
) -> np.ndarray:
    # fmt: on
    """ compute cos_mat.

    bumblebee correlation coefficients.

        embed correlation matrix
In [17]: bee_corr(['i love you', 'i hate you', 'i like you'], ['测试', '我爱你' , '你爱我', '我恨你'])
Out[17]:
array([[0.14067021, 0.9664803 , 0.95094556, 0.6901501 ],
       [0.10837743, 0.67109114, 0.661757  , 0.94350916],
       [0.13074915, 0.85840636, 0.8426784 , 0.6897449 ]], dtype=float32)
In [18]: bee_corr(['i love you', 'i hate you', 'i like you', 'i dislike you'], ['测试', '我爱你' , '你爱我', '我恨你'])
array([[0.14067024, 0.9664803 , 0.95094544, 0.69015014],
       [0.10837745, 0.67109126, 0.66175705, 0.9435092 ],
       [0.13074915, 0.85840625, 0.8426785 , 0.68974483],
       [0.09699164, 0.65500754, 0.6448661 , 0.88207084]], dtype=float32)

bee_corr(['i love you', 'i hate you', 'i like you', 'i dislike you'], ['测试', 'ich liebe dich', 'ich hasse dich'])
Out[19]:
array([[0.14067021, 0.97593087, 0.76346785],
       [0.1083775 , 0.66938734, 0.95404804],
       [0.13074915, 0.8375261 , 0.75783646],
       [0.09699164, 0.64676434, 0.8893524 ]], dtype=float32)

    # embed1 = encode(text1)
    # embed2 = encode(text2)

    # return cos_matrix2(encode(text1), encode(text2))

    flist[-3][2] in ['0.85']
    flist[-4][2] in ]'0.81']
    """

    if debug:
        logzero.loglevel(10)
    else:
        logzero.loglevel(20)

    if pt in ["laser"]:
        if src_lang is None or src_lang in [""]:
            src_lang = detect_lang(" ".join(src_text))
            logger.info(" Detectec src language: %s", src_lang)
        if tgt_lang is None or tgt_lang in [""]:
            tgt_lang = detect_lang(" ".join(tgt_text))
            logger.info(" Detectec tgt language: %s", tgt_lang)

    # _ = """  # moved from bee_aligner

    logger.info(" Processing the first file")
    # love10 len(src_text) == 1334, len(tgt_text) == 10080
    # In [27]: sum(len(elm.split()) for elm in src_text)
    # Out[27]: 15134 words, 66247 chars 6s/1000chars, 4.4chars/word
    # %time 7min 27s  30s/1000 words
    # sum(len(elm) for elm in src_text)
    # tgt_text chinese 28255 chars
    # %time  5min 23s 10s/1000chars

    # SPINNER.text = " beesy processing the first file"
    # SPINNER.start()

    try:
        src_vec = encode(src_text, pt=pt, lang=src_lang, debug=debug)
        # SPINNER.ok("OK")
        # SPINNER.ok("✅ ")
    except Exception as exc:
        logger.error("encode(src_text) exc: %s", exc)
        # SPINNER.fail("X ")
        raise
    finally:
        # SPINNER.text = "done processing the first file"
        ...

    logger.info(" Processing the second file")
    # SPINNER.text = " beesy processing the second file"
    # SPINNER.start()

    try:
        tgt_vec = encode(tgt_text, pt=pt, lang=tgt_lang, debug=debug)
        # SPINNER.ok("✅ ")
        # SPINNER.ok("OK")
    except Exception as exc:
        logger.error("encode(tgt_text) exc: %s", exc)
        # SPINNER.fail("X")
        raise
    finally:
        # SPINNER.text = ""
        # SPINNER.text = "done processing the second file"
        ...

    #
    logger.info(" Crunching some numbers...")
    # spinner_start(" beesy crunching some numbers...")
    try:
        cos_mat = cos_matrix2(src_vec, tgt_vec)
    except Exception as exc:
        logger.error("exc: %s", exc)
        raise
    finally:
        # spinner_stop(" done crunching numbers...")
        ...

    # """

    return cos_mat

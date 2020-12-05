r'''
bee_aligner

oracle-sentence-transformer.py

https://pytorch.org/ pip torch
    pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    # 292M
pip install sentence-transformer  # scipy 105M numpy 53M
pip install logzero
# pip install google-tr-free
# pip install requests-cache
# pip install js2py
pip install path
pip install pytest
pip install absl-py  # 848K

pip install pyinstaller

pip install pandas ipython  # pandas 43 MB, ipython 5.7M
# du -sh venv  # 702M

'''

from typing import List, Optional, Tuple, Union

import numpy as np
from threading import currentThread

import logzero
from logzero import logger

# from bee_aligner.st_cache_bee_corr import bee_corr
from bee_aligner.bee_corr import bee_corr

from bee_aligner.text_to_paras import text_to_paras

from bee_aligner.find_aligned_pairs import find_aligned_pairs
from bee_aligner.gen_row_alignment import gen_row_alignment

_ = '''
embeddings1b = encode(sents_en)  # %time 13.9 s
embeddings2b = encode(sents_zh)  # %time 21.7 s
c2b = cos_matrix2(np.asarray(embeddings1b), np.asarray(embeddings2b))
# '''


def bee_aligner(  # pylint: disable=too-many-locals
        src_text: Union[list, str],
        tgt_text: Union[list, str],
        cos_mat: Optional[np.ndarray] = None,  # correlation matrix from bee_corr
        thr: Optional[float] = None,  # 1/1.618, golden radio = 1.618
        tol: int = 4,
        debug: Optional[bool] = False,
) -> List[Tuple[str, str, float]]:
    ''' bee aligner
    if thr is None: set to
        3 * np.mean(cos_mat)  # (3 * av_corr)

    refer also to bert_aligner
    in pypi-projects\bert_aligner
    '''

    # for GUI in check_thread_update
    c_th = currentThread()

    # convert str to list
    if isinstance(src_text, str):  # pragma: no cover
        src_text = text_to_paras(src_text)
    if isinstance(tgt_text, str):  # pragma: no cover
        tgt_text = text_to_paras(tgt_text)

    if not isinstance(src_text, list):  # pragma: no cover
        raise SystemExit(f'[{__file__}]: src_text [{type(src_text)}] is not a list, exiting...')
    if not isinstance(tgt_text, list):  # pragma: no cover
        raise SystemExit(f'[{__file__}]: tgt_text [{type(tgt_text)}] is not a list, exiting...')

    if debug:
        logzero.loglevel(10)
    else:
        logzero.loglevel(20)

    # if cos_mat not supplied, calculate it on spot
    # and cached to bee_aligner.cos_mat
    if cos_mat is None:
        _ = """
        try:
            cos_mat = bee_aligner.cos_mat
        except Exception as exc:
            logger.error("exc: %s (first run expected) ", exc)
            logger.info(" first run cos_mat not supplied, caching cos_mat")
            cos_mat = bee_corr(src_text, tgt_text)
            bee_aligner.cos_mat = cos_mat
        # """
        cos_mat = bee_corr(src_text, tgt_text)

    _ = """  # moved to bee_corr
    logger.info(' Processing the first file')
    # love10 len(src_text) == 1334, len(tgt_text) == 10080
    # In [27]: sum(len(elm.split()) for elm in src_text)
    # Out[27]: 15134 words, 66247 chars 6s/1000chars, 4.4chars/word
    # %time 7min 27s  30s/1000 words
    # sum(len(elm) for elm in src_text)
    # tgt_text chinese 28255 chars
    # %time  5min 23s 10s/1000chars

    SPINNER.text = ' beesy processing the first file'
    SPINNER.start()
    src_vec = encode(src_text)
    SPINNER.text = 'done processing the first file'
    # SPINNER.ok("✅ ")
    SPINNER.ok("OK")

    logger.info(' Processing the second file')
    SPINNER.text = ' beesy processing the second file'
    SPINNER.start()
    tgt_vec = encode(tgt_text)
    SPINNER.text = 'done processing the second file'
    # SPINNER.ok("✅ ")
    SPINNER.ok("OK")
    SPINNER.text = ''

    #
    logger.info('  Crunching some numbers...')
    spinner_start(' beesy crunching some numbers...')
    cos_mat = cos_matrix2(src_vec, tgt_vec)
    spinner_stop(' done crunching numbers...')
    # """

    src_len = len(src_text)
    tgt_len = len(tgt_text)

    # iterate on thr
    # av_corr = sum(sum(cos_mat)) / cos_max.size
    # if resu.__len__() < 0.33 * src_len, set thr = min(3 * av_corr, thr)

    # set default to 3 * av_correlation
    if thr is None:
        # thr = 3 * sum(sum(cos_mat)) / cos_max.size
        thr = 1.1 * np.mean(cos_mat)

        # make sure it's not too big
        if thr > 1 / 1.618:
            thr = 1 / 1.618

    logger.debug(' %s', '  Doing some more processing... find_aligned_pairs')
    t_set, _ = find_aligned_pairs(
        cos_mat,
        thr=thr,
        tol=tol,
        matrix=True)

    logger.debug(' %s', '  Hold on...gen_row_alignment')
    resu = gen_row_alignment(t_set, src_len, tgt_len)
    para_list = []

    # for idx in range(len(resu)):
    for idx, _ in enumerate(resu):
        idx0, idx1, idx2 = resu[idx]  # pylint: disable=invalid-name
        # out = ['' if idx0 == '' else src_text[int(idx0)], '' if idx1 == '' else tgt_text[int(
        out = ('' if idx0 == '' else src_text[int(idx0)], '' if idx1 == '' else tgt_text[int(
            idx1)], '' if idx2 == '' else f'{float(idx2):.2f}')
        para_list.append(out)

    c_th.para_list = para_list  # type: ignore

    return para_list

# spinner_stop("end of initialization...")

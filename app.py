"""Run st_app.py in ipython."""
from typing import List

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

import base64
from io import BytesIO
from polyglot.text import Detector, Text
from sentence_splitter import split_text_into_sentences
from timeit import default_timer
import more_itertools as mit

import logging
import logzero
from logzero import logger
# from absl import app, flags

from bee_aligner import __version__
from bee_aligner.color_table_applymap import color_table_applymap
from bee_aligner.fetch_sent_corr import fetch_sent_corr
# from bee_aligner.plist_to_slist import plist_to_slist
from bee_aligner.single_or_dual import single_or_dual
from bee_aligner.plist_to_slist import plist_to_slist
from bee_aligner.text_to_plist import text_to_plist
# from bee_aligner.bee_aligner import bee_aligner

# import tinybee.embed_text
from fetch_embed import fetch_embed

from tinybee.cos_matrix2 import cos_matrix2
from tinybee.find_pairs import find_pairs
from tinybee.cmat2tset import cmat2tset
from tinybee.gen_iset import gen_iset
from tinybee.gen_aset import gen_aset
from tinybee.align_texts import align_texts
from tinybee.lowess_pairs import lowess_pairs
from tinybee.plot_tset import plot_tset

# use sentence_splitter if supported
LANG_S = ["ca", "cs", "da", "nl", "en", "fi", "fr", "de",
          "el", "hu", "is", "it", "lv", "lt", "no", "pl",
          "pt", "ro", "ru", "sk", "sl", "es", "sv", "tr"]

# FLAGS = flags.FLAGS
# flags.DEFINE_boolean("debug", False, "Show debug messages", short_name="d")

LOGLEVEL = 10
logzero.loglevel(LOGLEVEL)

from diskcache import FanoutCache
cache = FanoutCache()


@cache.memoize(typed=True, expire=2 * 60 * 60, tag='embed_text')  # 60s
# @st.cache does not seem to work in ipython
# @st.cache
def embed_text(text: List[str]):
    """Cache tinybee.embed_text.embed_text(text).

    fetch_embed(item, livepbar=False)
    """
    # return tinybee.embed_text.embed_text(*args, **kwargs)
    return fetch_embed(text, livepbar=False)


def seg_text(text: str, lang: str) -> List[str]:
    """ split text to sentences.

    use sentence_splitter if supported,
    else use polyglot.text.Text
    """
    if lang in LANG_S:
        return split_text_into_sentences(text, lang)

    return [elm.string for elm in Text(text, lang).sentences]


def split_text(text, sep='\n'):
    if isinstance(text, bytes):
        text = text.decode("utf8")
    return [elm.strip() for elm in text.split(sep) if elm.strip()]


pd.set_option('precision', 2)
pd.options.display.float_format = '{:,.2f}'.format

src_file = "test_en.txt"
tgt_file = "test_zh.txt"

src_text = split_text(Path(f"data/{src_file}").read_text("utf8"))
tgt_text = split_text(Path(f"data/{tgt_file}").read_text("utf8"))

print(f" src tgt lengths: {len(src_text)}, {len(tgt_text)}")

op_mode = "Para Align"
op_mode = "Sent Align"

# src_file = src_fileio.name
# tgt_file = tgt_fileio.name
src_plen = len(src_text)
tgt_plen = len(tgt_text)

print(f"{src_file} ({src_plen} paras) {tgt_file} ({tgt_plen} paras) ready")
logger.debug(" src_text: %s, tgt_text: %s ", len(src_text), len(tgt_text))

if op_mode in ["Sent Align"]:
    src_lang = Detector(" ".join(src_text)).language.code
    tgt_lang = Detector(" ".join(tgt_text)).language.code

    src_sents = []
    for elm in src_text:
        src_sents.extend(seg_text(elm, src_lang))

    tgt_sents = []
    for elm in tgt_text:
        tgt_sents.extend(seg_text(elm, tgt_lang))

    st.info(f"texts ({src_lang}, {tgt_lang}) segmented to {len(src_sents)}, {len(tgt_sents)} sents, respectively")

    # st.write(src_sents)
    # st.write(tgt_sents)

    src_blocks = src_sents
    tgt_blocks = tgt_sents
else:
    src_blocks = src_text
    tgt_blocks = tgt_text

src_blen = len(src_blocks)
tgt_blen = len(tgt_blocks)
print(f" {src_blen} src blocks, {tgt_blen} tgt blocks")

logger.debug("embedding src_text")
src_embed = []
len_ = len(src_blocks)
tot = len_ // 32 + bool(len_ % 32)
idx = 0
for elm in mit.chunked(src_blocks, 32):
    idx += 1
    logger.debug("  %s, %s", idx, idx / tot)
    try:
        _ = embed_text(elm)
    except Exception as e:
        st.write(e)
        _ = [[str(e)] + [""] * 31]
    src_embed.extend(_)
    # pbar.progress(idx/tot)

logger.debug("embedding tgt_text")
tgt_embed = []
len_ = len(tgt_blocks)
tot = len_ // 32 + bool(len_ % 32)
idx = 0
for elm in mit.chunked(tgt_blocks, 32):
    idx += 1
    logger.debug("  %s, %s", idx, idx / tot)
    try:
        _ = embed_text(elm)
    except Exception as e:
        st.write(e)
        _ = [[str(e)] + [""] * 31]
    tgt_embed.extend(_)
    # pbar.progress(idx / tot)

cmat = cos_matrix2(src_embed, tgt_embed)

cmat = np.array(cmat)

tset = cmat2tset(cmat)

plt.figure(2)

argmax = np.argmax(cmat, axis=0)
max = np.max(cmat, axis=0)

# the same as tset = cmat2tset with 3rd col: max
vset = [*zip(range(tgt_blen), argmax, max)]  # max value set
sns.scatterplot(
    data=vset,
    columns=['y00', 'argmax', 'max']),
    x='y00',
    y='argmax',
    size='max',
    hue='max',
)

iset = gen_iset(cmat)
plt.figure(2)
sns.scatterplot(
    data=pd.DataFrame(iset,
    columns=['y00', 'argmax']),
    x='y00',
    y='argmax',
)
plt.xlim(0, tgt_len)
plt.ylim(0, src_len)

src_len = len(src_blocks)
tgt_len = len(tgt_blocks)

aset = gen_aset(vset, src_len, tgt_len)
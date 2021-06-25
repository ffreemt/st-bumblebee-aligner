r"""Showcase

tset
    eliminate outliers based on clustering
        gen_row_align: further eliminate non-compliant points
    alternative 1. use to generate aset
            generate iset (based on interpolation)
                use iset as guide (trace) on tset -> aset for aligning
            alternative 2: gen aset

---
based on bk\st_app.py--

bk\st_app.py--:
pset = find_pairs(cmat, 3)

tinybee.find_pairs gen_aset
embed_text: @st.cache fetch_embed.fetch_embed
    src_embed = embed_text(src_blocks
    tgt_embed = embed_text(tgt_blocks

cmat = cos_matrix2(src_embed, tgt_embed)
tset = cmat2tset(cmat)
pset = find_pairs(cmat, 3)
    used iset = gen_iset(cmat)
        [used dbscan clustering
        and pset = gen_row_align (tset)
        iset = interpolate_pset(pset, tgt_len)
        ]
    pset: tset with iset [(optics labels_ > -1 + gen_row_align( constrainst) + interpolate_pset)]
aset = gen_aset(pset, src_len, tgt_len)
aligned_blocks = align_texts(aset, src_blocks, tgt_blocks)

"""

__version__ = "0.1.2a"
__version__ = "0.1.2b"

__intructins__ = f"""
*   Set up options in the left sidebar

*   Click expanders\n +: to reveal more details; -: to hide them

*   Press '**Click to start aligning**' to get the ball rolling. (The button will appear when everything is ready.)

*   bumblebee-ng v.{__version__} from mu@qq41947782's keyboard in cyberspace. Join **qq group 316287378** for feedback and questions or to be kept updated. This web version of bumblebee is the twin brother of **desktop bumblebee**.
"""
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import joblib
import streamlit as st

import base64
from io import BytesIO
from polyglot.text import Detector, Text
from sentence_splitter import split_text_into_sentences

# from timeit import default_timer
import more_itertools as mit
from pendulum import now
import langid

from toolz.functoolz import pipe
from tqdm import tqdm

# python -m textblob.download_corpora
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from sklearn.preprocessing import normalize

import logging
import logzero
from logzero import logger
# from absl import app, flags

# from bee_aligner import __version__

from fastlid import fastlid
from bee_aligner.color_table_applymap import color_table_applymap
from bee_aligner.fetch_sent_corr import fetch_sent_corr
# from bee_aligner.plist_to_slist import plist_to_slist
from bee_aligner.single_or_dual import single_or_dual
from bee_aligner.plist_to_slist import plist_to_slist
from bee_aligner.text_to_plist import text_to_plist
# from bee_aligner.bee_aligner import bee_aligner

# import tinybee.embed_text
from fetch_embed import fetch_embed

import tinybee
from tinybee.cos_matrix2 import cos_matrix2
from tinybee.find_pairs import find_pairs
from tinybee.cmat2tset import cmat2tset
from tinybee.gen_iset import gen_iset
from tinybee.gen_aset import gen_aset
from tinybee.align_texts import align_texts

# linear regression UFast-Engine
from tinybee.lrtrace_tset import lrtrace_tset
from tinybee.gen_row_align import gen_row_align

from light_scores import light_scores
from light_aligner.bingmdx_tr import bingmdx_tr

from batch_tr import batch_tr

from fast_scores import fast_scores
from fast_scores.process_zh import process_zh
from fast_scores.process_en import process_en
from fast_scores.en2zh import en2zh

# import SessionState
# sess_state = SessionState.get(src_fileio='', tgt_fileio='')
from get_state import get_state

sns.set()
sns.set_style("whitegrid")

# use sentence_splitter if supported
LANG_S = ["ca", "cs", "da", "nl", "en", "fi", "fr", "de",
          "el", "hu", "is", "it", "lv", "lt", "no", "pl",
          "pt", "ro", "ru", "sk", "sl", "es", "sv", "tr"]

# FLAGS = flags.FLAGS
# flags.DEFINE_boolean("debug", False, "Show debug messages", short_name="d")

LOGLEVEL = 10
logzero.setup_default_logger(level=LOGLEVEL)

print(f"tinybee version: {tinybee.__version__}")


@st.cache
def embed_text(text: List[str]):
    """Cache fetch_embed.embed_text.embed_text(text).

    fetch_embed(item, livepbar=False)
    """
    # return tinybee.embed_text.embed_text(*args, **kwargs)
    return fetch_embed(text, livepbar=False)


def get_eta(disp_len, len_, scale=0.56, unit="seconds"):
    """Estimate toa and expiry time.

    Args:
        disp_len_: length to display in f"{disp_len} to process,"
    """
    if unit in ["seconds"]:
        tot_time = round(len_ * scale)
        arr_time = now().add(seconds=tot_time).in_timezone("Asia/Shanghai").format("YYYY-MM-DD HH:mm:ss z")
        _ = f"blocks to process: {disp_len}, eta: {tot_time} {unit} (~{arr_time})"
    else:
        unit = "minutes"
        tot_time = round(len_ * scale, 1)
        arr_time = now().add(minutes=tot_time).in_timezone("Asia/Shanghai").format("YYYY-MM-DD HH:mm:ss z")
        _ = f"blocks to process: {disp_len}, eta: {tot_time} {unit} (~{arr_time})"
    return _


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


@st.cache
def bee_corr(text1, text2, url=None):
    logger.debug('type: %s, text[:4]: %s', type(text1), text1[:10])
    if isinstance(text1, str):
        text1 = split_text(text1)
    if isinstance(text2, str):
        text2 = split_text(text2)
    if url is None:
        return fetch_sent_corr(text1, text2)
    else:
        return fetch_sent_corr(text1, text2, url=url)


# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/12
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="aligned_paras.xlsx">Download aligned paras xlsx file</a>'  # decode b'abc' => abc


def get_table_download_link_sents(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="aligned_sents.xlsx">Download aligned sents xlsx file</a>'  # decode b'abc' => abc


def fetch_file_contents(src_fileio, tgt_fileio):
    """Convert fle or fileio to list of lines."""
    if isinstance(src_fileio, str):
        src_file = src_fileio.copy()
    else:
        if src_fileio:
            src_file = src_fileio.getvalue()
        else:
            src_file = b""
        if isinstance(src_file, bytes):
            src_file = src_file.decode("utf8")

    src_text = split_text(src_file)

    if isinstance(tgt_fileio, str):
        tgt_file = tgt_fileio.copy()
    else:
        if tgt_fileio:
            tgt_file = tgt_fileio.getvalue()
        else:
            tgt_file = b""
        if isinstance(tgt_file, bytes):
            tgt_file = tgt_file.decode("utf8")

    tgt_text = split_text(tgt_file)

    len_ = len(src_text) + len(tgt_text)
    if len_ > 300:
        st.warning(" This is likely to take a while...")
        len_ = len(src_text) + len(tgt_text)

        logger.info("total paras: %s", len_)

        contents_expander = st.beta_expander("Contents preview", expanded=False)
        with contents_expander:
            _ = 3
            st.subheader(f"first {_} paras in file1 and file2, respectively")
            st.write(f" {src_text[:_]} ")
            st.write(f" {tgt_text[:_]} ")
            st.subheader(f"last {_} paras in file1 and file2, respectively")
            st.write(f" {src_text[-_:]} ")
            st.write(f" {tgt_text[-_:]} ")

            len1 = len(src_text)
            len2 = len(tgt_text)
            est_time = round(len1 / 32) + bool(len1 % 32)
            est_time += round(len2 / 32) + bool(len2 % 32)
            est_time *= 13 / 60

            st.info([
                f" file1: {len(src_text)} paras",
                f" file2: {len(tgt_text)} paras",
            ])
    return src_text, tgt_text


def front_cover():
    # global src_fileio, op_mode, model_url
    # return src_fileio, tgt_fileio, op_mode, model_url

    # src_fileio = ""
    # tgt_fileio = ""
    # op_mode = ""
    # model_url = ""

    st.sidebar.markdown("### Streamlit powered bumblebee-ng aligner")
    sb_tit_expander = st.sidebar.beta_expander("More info (click to toggle)", expanded=True)
    with sb_tit_expander:
        st.write(f"Showcasing v.{__version__}, refined, quasi-prodction-ready🚧...")
        # branch
        # st.markdown(
        st.write(
            """What would you like to do?
The following alignment engines are available.

**UFast-Engine**: ultra-fast, based on a home-brewed algorithm, faster than blazing fast but can only process en-zh para/sent pairs, not as sophisticated as DL-Engine;

**SFast-Engine**: super-fast, based on machine translation;

**Fast-Engine**: based on yet another home-brewed algorithm, blazing fast but can only process en-zh para/sent pairs;

**DL-Engin**: based on machine learning, multilingual, one para/sent takes about 1s."""
        )

    _ = '''  # moved to 'form submit button' below
    sb_tit_expander = st.sidebar.beta_expander("Select operation mode (click to toggle)", expanded=False)
    with sb_tit_expander:
        op_mode = st.selectbox(
            "Select operation",
            ("Para Align", "Sent Align")
        )
        _ = """
        model_sel = st.selectbox(
            "Select model",
            ("Model 1", "Google USE")
        )
        # """

    # selection_expander = st.beta_expander("Operation and model selected", expanded=False)
    selection_expander = st.beta_expander(f"Operation mode selected: {op_mode}", expanded=False)
    with selection_expander:
        st.success(op_mode)
        # st.success(model_sel)
    # '''

    # model_url
    _ = """
    if model_sel in ["Model 1"]:
        model_url = None
        # default to preset url "http://216.24.255.63:8000/text/"
    else:
        model_url = "http://216.24.255.63:8008/text/"
    # """

    _ = """  # dealt with in main
    # st.sidebar.subheader("Pick two files")
    sb_pick_filels = st.sidebar.beta_expander("Pick two files", expanded=True)
    with sb_pick_filels:
        src_fileio = st.file_uploader("Choose a file (utf8 txt)", type=['txt',], key="src_text")
        tgt_fileio = st.file_uploader("Choose a file (utf8 txt)", type=['txt',], key="tgt_text")

    logger.debug("type src_fileio: %s", type(src_fileio))
    logger.debug("type tgt_fileio: %s", type(tgt_fileio))
    # """

    # return src_fileio, tgt_fileio, op_mode, model_url
    # return src_fileio, tgt_fileio, model_url


def align_sents():
    st.write("align sents to be implemented...")
    return

    _ = """
    # align sents
    if st.sidebar.checkbox(
        "tick to proceed with sent alignment (w/o para align)",
        value=False,
        key="sent-align",
    ):
    ("Para/Sent Align", "Simple Sent Align")
    # """

    # if op_mode in ["Simple Sent Align"]:

    if not (file1_flag and file2_flag):
        # st.info("Pick two files first")
        instruction1()
    else:
        # st.info(" sent alignment to be implemented ")
        sents1 = []
        for elm in src_text:
            if elm.strip():
                sents1.extend(seg_text(elm, lang1))
        st.info(sents1[:3])
        sents2 = []
        for elm in tgt_text:
            if elm.strip():
                sents2.extend(seg_text(elm, lang2))
        st.info(sents2[:3])
        len1s = len(sents1)
        len2s = len(sents2)
        st.info([
            f"file1: {len1s} sents", f"{lang1}",
            f"file2: {len2s} sents", f"{lang2}",
        ])

        est_time1 = len1s // 32 + bool(len1s % 32)
        est_time1 += len2s // 32 + bool(len2s % 32)
        est_time1 *= 7 / 60

        st.info(f"The first run may take a while, about {est_time1:.1f} min")
        try:
            # cos_mat = np.asarray(bee_corr(src_text, tgt_text))
            cos_mat1 = np.asarray(bee_corr(sents1, sents2, url=model_url))
        except Exception as exc:
            # st.write("exc: ", exc)
            logger.error("exc: %s" % exc)
            # st.stop()
            raise SystemExit(1) from Exception

        st.markdown("### cosine similarity matrix (sents)")
        st.dataframe(pd.DataFrame(cos_mat1.round(2)).style.highlight_max(axis=0))

        mean1 = cos_mat1.mean()
        var1 = cos_mat1.var()

        fig, ax = plt.subplots()
        # fig = plt.figure()  # (figsize= (10,7))

        # cmap = sns.diverging_palette(20, 220, as_cmap=True)
        sns.heatmap(cos_mat1, vmin=0, vmax=1)

        plt.xlabel(f"{tgt_file}")
        plt.ylabel(f"{src_file}")
        plt.title(f"mean={mean1.round(2)} var={var1.round(2)} cosine similarity (sents) heatmap")
        st.pyplot(fig)

        # st.markdown("## fine-tune alignment")
        st.header("fine-tune sent alignment")

        thr_i = float(round(min(mean1 + 30 * var1, 0.5), 2))
        st.info(f"inital threshold={thr_i:.2f}")

        thr1 = st.slider(
            "threshold (0...1) -- drag or click to adjust threshold value",
            min_value=0.,
            max_value=1.,
            value=thr_i,
            step=0.05,
            # key="sliderkey",
        )

        p_list1 = bee_aligner(sents1, sents2, cos_mat=cos_mat1, thr=thr1)

        if p_list1 is not None:
            df1 = pd.DataFrame(p_list1)
            st.markdown("#### sent alignment at a glance")
            st.info("(hove over a cell to disply full text)")

            st.dataframe(df1)
            st.subheader("sent alignment in detail")
            if st.checkbox(f"tick to show detailed sent alignment (threhold={thr1})", value=0, key="para_df"):
                st.table(df1)

            s_df1 = color_table_applymap(df1)

            st.sidebar.subheader("aligned sent xlsx file for downloading")
            st.sidebar.markdown(get_table_download_link_sents(s_df1), unsafe_allow_html=True)


def align_paras():
    st.write("align paras...")
    ...


def instruction1():
    st.info("Pick two files first\n\nFrom the left sidebar, pick two files, for example, one contains English text, one contains Chinese text. All files should be in **utf-8 txt** format (**no pdf, docx format etc supported**). bumblebee supports many language pairs, not just English-Chinese.")


def back_cover():
    logger.debug("back_cover entry")
    back_cover_expander = st.beta_expander("Instructions")
    with back_cover_expander:
        st.markdown(__intructins__)

    logger.debug("back_cover exit")


st.set_page_config(
    page_title=f"Bumblebee-ng v{__version__}",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Run main."""
    # print(args)

    state = get_state()

    pd.set_option("precision", 2)
    pd.options.display.float_format = "{:,.2f}".format

    logzero.loglevel(LOGLEVEL)

    logger.info(f"tinybee version: {tinybee.__version__}")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
    )

    logging.debug("debug main start")
    logging.info("info main start")

    logger.debug("\n\t\t --- main started... ---")
    # logger.debug(" sess_states: %s", sess_state)

    _ = """
    if FLAGS.d:
        logzero.loglevel(10)
    else:
        logzero.loglevel(20)
    # """
    back_cover()
    logger.debug(" back_cover done...")

    p_list = None
    p_list1 = None
    file1_flag, file2_flag = False, False

    # src_text = []
    # tgt_text = []
    # src_fileio, tgt_fileio = None, None
    # src_file, tgt_file = "", ""

    # src_fileio, tgt_fileio, op_mode, model_url = front_cover()
    # src_fileio, tgt_fileio, model_url = front_cover()
    # logger.debug(" front_cover done...")

    # front_cover()
    st.sidebar.markdown("### Streamlit powered bumblebee-ng aligner")
    sb_tit_expander = st.sidebar.beta_expander("More info (click to toggle)", expanded=True)
    with sb_tit_expander:
        st.write(f"Showcasing v.{__version__}, refined, quasi-prodction-ready👷")
        # branch
        # st.markdown(
        st.write(
            """What would you like to do?
The following alignment engines are available.

**UFast-Engine**: based on yet another home-brewed algorithm, faster than blazing fast but can only process en-zh para/sent pairs, not as sophisticated as DL-Engine;

**SFast-Engine**: super-fast, based on machine translation;

**Fast-Engine**: based on a home-brewed algorithm, blazing fast but can only process en-zh para/sent pairs, not as sophisticated as DL-Engine;

**DL-Engin**: based on machine learning, multilingual, processing one para/sent takes about 1s."""
        )

    # src_fileio tgt_fileio
    sb_pick_filels = st.sidebar.beta_expander("Pick two files", expanded=True)
    with sb_pick_filels:
        src_fileio = st.file_uploader("Choose source file (utf8 txt)", type=['txt', ], key="src_text", accept_multiple_files=True)

        tgt_fileio = st.file_uploader("Choose target file (utf8 txt)", type=['txt', ], key="tgt_text", accept_multiple_files=True)

    if src_fileio:
        logger.debug(" type(src_fileio): %s", type(src_fileio))
        logger.debug("src_fileio[-1].name: [%s]", src_fileio[-1].name)

        # state.src_fileio = src_fileio
        state.src_file = src_fileio[-1].getvalue().decode()
        state.src_filename = src_fileio[-1].name
    if tgt_fileio:
        logger.debug("tgt_fileio[-1].name: [%s]", tgt_fileio[-1].name)
        # state.tgt_fileio = tgt_fileio
        state.tgt_file = tgt_fileio[-1].getvalue().decode()
        state.tgt_filename = tgt_fileio[-1].name

    fileio_slot = st.empty()

    # logger.debug(" src_fileio: %s", state.src_fileio)
    # logger.debug(" tgt_fileio: %s", state.tgt_fileio)

    if not (state.src_filename and state.tgt_filename):
        logger.debug(" not (state.tgt_fileio and state.src_fileio) is True")
        fileio_slot.text(f""" available files:
srcfile [{state.src_filename}] tgtfile [{state.tgt_filename}], upload or re-upload files or refresh page""")
        # st.stop()

    # --- checkpoint ---
    logger.debug("   - checkpoint -")
    if not (state.src_filename and state.tgt_filename):
        return

    files = st.empty()
    # either of both empty
    if not (state.src_file.strip() and state.tgt_file.strip()):
        # files.write("At least one of the files contains nothing.")
        if not state.src_file.strip() and not state.tgt_file.strip():
            files.write("Both files empty!")
        elif not state.src_file.strip():
            files.write("Source file empty!")
        else:
            files.write("Target file empty!")

        return

    # overwrite possible previous message
    files.write("")

    # --- process ---
    logger.debug("    --- process ---")

    try:
        # fetch file contents
        src_text, tgt_text = fetch_file_contents(src_fileio[-1], tgt_fileio[-1])

        # src_text = src_fileio[-1].getvalue().decode().splitlines()
        # tgt_text = tgt_fileio[-1].getvalue().decode().splitlines()
        # src_text = [elm.strip() for elm in src_text if elm.strip()]
        # tgt_text = [elm.strip() for elm in tgt_text if elm.strip()]

        src_plen = len(src_text)
        tgt_plen = len(tgt_text)
    except Exception as e:
        logger.error(e)
        st.write("Got a problem: %s, quitting" % e)
        return

    fileio_slot.text(f"{state.src_filename} ({src_plen} paras)\n{state.tgt_filename} ({tgt_plen} paras) ready")

    logger.debug(" len src_text: %s, len tgt_text: %s ", len(src_text), len(tgt_text))

    if not src_text or not tgt_text:
        if not src_text:
            st.warning("Source file is apparently empty")
        if not tgt_text:
            st.warning("Target file is apparently empty")

    # logger.info(">>>> return")
    # return

    # ---
    text_info_exp = st.beta_expander("text info and samples", expanded=False)
    with text_info_exp:
        st.write(f"[{state.src_filename}] [{state.tgt_filename}] read in")
        st.write("number of src-text tgt-text paras:", len(src_text), len(tgt_text))
        st.write(src_text[-3:], tgt_text[-3:])

    # form submit button https://blog.streamlit.io/introducing-submit-button-and-forms/
    _ = """
    sb_engine_expander = st.sidebar.beta_expander("Select engine type", expanded=False)
    with sb_engine_expander:
        ali_engine = st.selectbox(
            "Select engine type",
            ("UFast-Engine", "SFast-Engine", "Fast-Engine", "DL-Engine")
        )
    # """

    ali_engine = st.sidebar.selectbox(
        "Select engine type",
        ("UFast-Engine", "SFast-Engine", "Fast-Engine", "DL-Engine")
    )
    explain_text = {
        "UFast-Engine": "home-brewed, currently en-zh only, ultra-fast",
        "Fast-Engine": "home-brewed, currently en-zh only, fast",
        "SFast-Engine": "super-fast, based on machine translation, multilingual (currently just x-zh or zh-x)",
        "DL-Engine": "Based on deep-learning, more powerful, multilingual, slower"
    }
    selection_expander = st.sidebar.beta_expander(f"Align engine selected: {ali_engine}", expanded=True)
    with selection_expander:
        st.success(f"{ali_engine}: {explain_text[ali_engine]}")

    # logger.info("2>>>> return")
    # return

    # ### operation mode
    src_blocks = ""
    tgt_blocks = ""
    try:
        op_mode = st.sidebar.selectbox(
            "Select operation mode",
            ("Para Align", "Sent Align"),
            key=1,
        )
        if op_mode in ["Sent Align"]:
            try:
                src_lang = Detector(" ".join(src_text)).language.code
            except Exception as e:
                src_lang = "en"
            try:
                tgt_lang = Detector(" ".join(tgt_text)).language.code
            except Exception as e:
                tgt_lang = "zh"

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

        logger.debug("state.src_filename: [%s], state.tgt_filename: [%s]", state.src_filename, state.tgt_filename)
        # if not tgt_file or not src_file: return

        twofiles_avail = f"tgt {state.tgt_filename} and src {state.src_filename}"
        logger.debug("twofiles_avail: %s", twofiles_avail)

        if not (state.tgt_filename and state.src_filename):
            st.warning(' ** Files not ready yet. **')
            logger.debug("src_file: [%s], tgt_file: [%s]", state.src_filename, state.tgt_filenae)
            logger.debug(" not (state.tgt_filenae and state.src_filename) is True")
            # st.stop()
    except Exception as e:
        logger.error(e)
        st.write("Got a proble: %s, quitting..." % e)
    logger.info("operation mode >>>> return")
    # return
    # -----

    # DL-Engine does not need src_lang tgt_lang
    src_lang = "zh"  # to make pyright happy
    src_lang = "en"

    # Estimate process time
    try:
        tot_len = len(src_blocks) + len(tgt_blocks)
        if ali_engine in ["DL-Engine"]:
            len_ = len(src_blocks)
            tot = len_ // 32 + bool(len_ % 32)
            len_ = len(tgt_blocks)
            tot += len_ // 32 + bool(len_ % 32)
            tot_time = round(tot * 0.5, 1)
            arr_time = now().add(minutes=tot_time).in_timezone("Asia/Shanghai").format("YYYY-MM-DD HH:mm:ss z")
            eta_msg = f"blocks to process: {tot_len}, eta: {tot_time} minutes (~{arr_time})"
            # eta_msg = get_eta(tot, tot, 0.5, "minutes")
        elif ali_engine in ["Fast-Engine"]:
            # eta_msg = f"blocks to process: {tot_len}"
            # langid.set_languages(langs=['en', 'zh'])
            # src_lang = langid.classify(" ".join(src_blocks))
            # tgt_lang = langid.classify(" ".join(tgt_blocks))

            fastlid.set_languages = None
            src_lang = fastlid(src_blocks)[0]
            tgt_lang = fastlid(tgt_blocks)[0]

            if src_lang not in ['en', 'zh'] or tgt_lang not in ['en', 'zh']:
                logger.warning("We detected: src_lang=%s, tgt_lang=%s", src_lang, tgt_lang)
                logger.info("We ll preset them as ['en', 'zh'] and try again")
                fastlid.set_languages = ['en', 'zh']
                src_lang = fastlid(src_blocks)[0]
                tgt_lang = fastlid(tgt_blocks)[0]
                logger.info(" Retry result: src_lang=%s, tgt_lang=%s", src_lang, tgt_lang)

            if src_lang == tgt_lang:
                logger.error(" src_lang (%s) == tgt_lang (%s)", src_lang, tgt_lang)
                logger.info("The current version ignores this. Future versions may implement something to handle this.")
                return None

            # calculate estimated time of arrival for w4w conversion
            if src_lang in ['en']:
                eta_msg = get_eta(len(src_blocks), len(src_blocks))
            else:
                eta_msg = get_eta(len(tgt_blocks), len(tgt_blocks))
        elif ali_engine in ["SFast-Engine"]:  # SFast-Engine
            fastlid.set_languages = None
            src_lang = fastlid(src_blocks)[0]
            tgt_lang = fastlid(tgt_blocks)[0]
            if not (src_lang in ['zh'] or tgt_lang in ['zh']):
                st.warning(" src_lang (%s), tgt_lang (%s): one of these must be zh... result unpredictable" % (src_lang, src_lang))
                logger.warning(" src_lang (%s), tgt_lang (%s): one of these must be zh...", src_lang, src_lang)
                # return None

            if src_lang in ["zh"]:  # process tgt_blocks
                char_len = len(" ".join(tgt_blocks))
                if char_len > 50000:
                    eta_msg = get_eta(len(tgt_blocks), char_len / 10000, 1.5)  # secs
                else:
                    eta_msg = f" {len(tgt_blocks)} to process"
            else:  # process src_blocks
                char_len = len(" ".join(src_blocks))
                if char_len > 50000:
                    eta_msg = get_eta(len(tgt_blocks), char_len / 10000, 1.5)  # secs
                else:
                    eta_msg = f" {len(src_blocks)} blocks to process"
        else:  # if ali_engine in ["UFast-Engine"]:
            # UFast-Engine w4w sklearn tfidf
            fastlid.set_languages = None
            src_lang = fastlid(src_blocks)[0]
            tgt_lang = fastlid(tgt_blocks)[0]
            if not (src_lang in ['zh'] or tgt_lang in ['zh']):
                st.warning(" src_lang (%s), tgt_lang (%s): one of these must be zh... result unpredictable" % (src_lang, src_lang))
                logger.warning(" src_lang (%s), tgt_lang (%s): one of these must be zh...", src_lang, src_lang)

                st.warning(" src_lang (%s), tgt_lang (%s): one of these must be zh..." % (src_lang, src_lang))

                # return None

            eta_msg = f" {len(src_blocks) + len(tgt_blocks)} blocks to process"
    except Exception as e:
        logger.error(e)
        raise SystemError(e) from e

    logger.info("Estimate time>>>> return")
    # return

    try:
        blocks_exp = st.beta_expander(eta_msg, expanded=False)
        with blocks_exp:
            st.write(src_blocks)
            st.write(tgt_blocks)
    except Exception as e:
        logger.error(e)

    logger.debug(" blocks displayed ")
    print(" blocks displayed ")
    # return

    pset = [[]]

    # with st.sidebar.form('Form1'):
    with st.form('Form1'):
        # op_mode = st.selectbox('Select', ['Para Align', 'Sent Align'], key=1)
        st.write("**Ready when you are**")
        submitted1 = st.form_submit_button('Click to start aligning 🖱️')
        # click submit to get going...
        if submitted1:

            if ali_engine in ["DL-Engine"]:

                # cmat = bee_corr(src_blocks, tgt_blocks, url=model_url)

                # src_embed = embed_text(src_blocks)  # 329, 6min
                # tgt_embed = embed_text(tgt_blocks)  # 291,

                pbar = st.progress(0)
                logger.debug("embedding src_text")
                src_embed = []
                len_ = len(src_blocks)
                tot = len_ // 32 + bool(len_ % 32)
                idx = 0
                for elm in mit.chunked(src_blocks, 32):
                    idx += 1
                    logger.debug(" %s, %s", idx, idx / tot)
                    try:
                        _ = embed_text(elm)
                    except Exception as e:
                        st.write(e)
                        _ = [[str(e)] + [""] * 31]
                    src_embed.extend(_)
                    pbar.progress(idx / tot)

                pbar = st.progress(0)
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
                    pbar.progress(idx / tot)

                cmat = cos_matrix2(src_embed, tgt_embed)

                cmat = np.array(cmat)
            elif ali_engine in ["Fast-Engine"]:  # Fast-Engine w4w en2zh bm25
                diggin_time = st.empty()
                diggin_time.write(" diggin...")
                then = now()
                if src_lang in ['en']:
                    paras_w4w = []
                    for elm in tqdm(src_blocks):
                        paras_w4w.append(bingmdx_tr(elm))
                    lmat_w4w = light_scores([" ".join([*elm]) for elm in paras_w4w], [" ".join([*elm]) for elm in tgt_blocks])
                else:
                    paras_w4w = []
                    for elm in tqdm(tgt_blocks):
                        paras_w4w.append(bingmdx_tr(elm))
                    lmat_w4w = light_scores([" ".join([*elm]) for elm in src_blocks], [" ".join([*elm]) for elm in paras_w4w])

                # cmat = lmat_w4w.T
                cmat = lmat_w4w

                cmat = normalize(cmat, axis=0)

                time_elapsed = (now() - then).in_words()
                # st.write(f" Time used to figure out correlation matrix: {time_elapsed}")
                diggin_time.write(f" Time used to figure out correlation matrix: {time_elapsed}")

                logger.info(" lmat_w4w.shape: %s, len(src_blocks): %s, len(tgt_blocks): %s", lmat_w4w.shape, len(src_blocks), len(tgt_blocks))
            elif ali_engine in ["UFast-Engine"]:  # UFast-Engine w4w sklearn Tfidf
                diggin_time = st.empty()
                diggin_time.write(" diggin...")
                then = now()
                if src_lang in ['zh']:  # process tgt_blocks
                    # tr_blocks = batch_tr(tgt_blocks)
                    # cmat = light_scores([" ".join([*elm]) for elm in src_blocks], [" ".join([*elm]) for elm in tr_blocks])

                    # tr_blocks = process_en(tgt_blocks)
                    # fftr = pipe(ffen, *(process_en, en2zh, list))
                    _ = process_zh(src_blocks)
                    tr_blocks = pipe(tgt_blocks, *(process_en, en2zh, list))
                    # cmat = fast_scores(_, tr_blocks)
                    cmat = fast_scores(tr_blocks, _)
                else:  # tgt_blocks [zh], process src_blocks
                    # tr_blocks = batch_tr(src_blocks)
                    # cmat = light_scores([" ".join([*elm]) for elm in tr_blocks], [" ".join([*elm]) for elm in tgt_blocks])

                    _ = process_zh(tgt_blocks)
                    tr_blocks = pipe(src_blocks, *(process_en, en2zh, list))

                    # cmat = fast_scores(tr_blocks, _)
                    cmat = fast_scores(_, tr_blocks)

                # joblib.dump(cmat, "data/cmat.lzma")
                # logger.info(f" cmat written to data/cmat.lzma: {cmat.shape}")

                time_elapsed = (now() - then).in_words()
                diggin_time.write(f" Time used to figure out correlation matrix: {time_elapsed}")

                # cmat = cmat.T already done in fast_scores
                # cmat = normalize(cmat, axis=0)

                logger.info(" cmat.shape: %s, len(src_blocks): %s, len(tgt_blocks): %s", cmat.shape, len(src_blocks), len(tgt_blocks))

            else:  # SFast-Engine MT
                diggin_time = st.empty()
                diggin_time.write(" diggin...")
                then = now()
                if src_lang in ['zh']:  # process tgt_blocks
                    tr_blocks = batch_tr(tgt_blocks)
                    cmat = light_scores([" ".join([*elm]) for elm in src_blocks], [" ".join([*elm]) for elm in tr_blocks])
                else:  # process src_blocks
                    tr_blocks = batch_tr(src_blocks)
                    cmat = light_scores([" ".join([*elm]) for elm in tr_blocks], [" ".join([*elm]) for elm in tgt_blocks])

                # cmat = cmat.T  # comment out after updating light_scores to 0.1.2

                cmat = normalize(cmat, axis=0)

                time_elapsed = (now() - then).in_words()
                diggin_time.write(f" Time used to figure out correlation matrix: {time_elapsed}")

                logger.info(" cmat.shape: %s, len(src_blocks): %s, len(tgt_blocks): %s", cmat.shape, len(src_blocks), len(tgt_blocks))

            logger.debug("ali-engine: %s", ali_engine)
            print("ali-engine: %s" % ali_engine)

            heatmap_exp = st.beta_expander("heatmap", expanded=False)
            with heatmap_exp:
                fig, ax = plt.subplots()

                # sns.heatmap(cmat, ax=ax, cmap="hot")
                # sns.heatmap(cmat, ax=ax, cmap="gist_heat_r")
                # sns.heatmap(cmat)
                sns.heatmap(cmat, cmap="Blues")

                ax.invert_yaxis()

                ax.set_xlabel(f"{state.tgt_filename}")
                ax.set_ylabel(f"{state.src_filename}")
                ax.set_title("cosine similarity heatmap")
                st.pyplot(fig)

            # st.write(cmat)
            # st.write(" plotted")

            # st.markdown("### cosine similarity matrix")
            # st.dataframe(pd.DataFrame(cmat).style.highlight_max(axis=0))

            _, len_ = cmat.shape
            tset = cmat2tset(cmat)
            _ = '''
            if ali_engine in ("UFast-Engine",):
                row_len, col_len = cmat.shape
                _ = """
                try:
                    lr_tset = lrtrace_tset(tset)
                    pset = gen_row_align(lr_tset, row_len, col_len)
                except Exception as e:
                    logger.error(e)
                    st.error(str(e) + "--Something gone awry, quitting")
                    st.stop()
                    # return
                # """
                # pset = gen_row_align(tset, row_len, col_len)
                pset = find_pairs(cmat, 3)
            else:
                pset = find_pairs(cmat, 3)  # pair set with metrics
            # '''

            pset = gen_row_align(tset, *cmat.shape)

            st.write(f"{len(pset)} 'good' pairs found, ", f"{round(len(pset) / len_, 2) * 100}%")

            # _ = """
            pset_expand = st.beta_expander("pset: aligned pairs with metrics", expanded=True)
            with pset_expand:
                fig, ax = plt.subplots()
                df = pd.DataFrame(pset, columns=['y00', 'yargmax', 'cos'])
                sns.set_style("whitegrid")
                sns.scatterplot(
                    data=df,
                    x='y00',
                    y='yargmax',
                    hue='cos',
                    size='cos',
                    sizes=(1, 20),
                )

                ax.set_xlabel(f"{state.tgt_filename}")
                ax.set_ylabel(f"{state.src_filename}")
                ax.set_title("aligned pairs with illustrated cosine similarity")

                ax.set_xlim(xmin=0, xmax=len(tgt_blocks) - 1)
                ax.set_ylim(ymin=0, ymax=len(src_blocks) - 1)
                ax.grid(True)
                st.pyplot(fig)

            # """

            # iset
            # _ = """
            ymin, ymax = 0, len(src_blocks) - 1
            xmin, xmax = 0, len(tgt_blocks) - 1
            iset = gen_iset(cmat)

            # logger.debug(" **tset**: \n%s", tset)
            # logger.debug(" **iset**: \n%s", iset)

            iset_expand = st.beta_expander("interpolated pairs", expanded=True)
            with iset_expand:
                fig, ax = plt.subplots()
                df = pd.DataFrame(iset, columns=['y00', 'yargmax'])
                sns.set_style("whitegrid")
                sns.scatterplot(
                    data=df,
                    x='y00',
                    y='yargmax'
                )

                ax.set_xlabel(f"{state.tgt_filename}")
                ax.set_ylabel(f"{state.src_filename}")
                ax.set_title("interpolated pairs")

                ax.set_xlim(xmin=xmin, xmax=xmax)
                ax.set_ylim(ymin=ymin, ymax=ymax)
                ax.grid(True)
                st.pyplot(fig)
            # """

            src_len = len(src_blocks)
            tgt_len = len(tgt_blocks)

            logger.info("Saved to data/pset.lzma")
            logger.info(" pset: %s", pset)
            # print(f"***logger.level: {logger.level}")
            # if logger.level < 30: joblib.dump(pset, "data/pset.lzma")

            # aset = gen_aset(pset, src_len, tgt_len)
            _ = tuple((int(elm0), int(elm1), elm2) for elm0, elm1, elm2 in pset)
            aset = gen_aset(_, src_len, tgt_len)

            # st.write("\n aset ", len(aset))
            # st.write(pd.DataFrame(aset))

            logger.info("aset: %s", aset)
            print("aset: ", aset)

            # return

            # st.write(pd.DataFrame(aset))
            # s_df = color_table_applymap(df)

            aset_exp = st.beta_expander(" aset ", expanded=False)
            with aset_exp:
                df = pd.DataFrame(aset, columns=['src', 'tgt', 'cos'])
                st.table(color_table_applymap(df))

            _ = """
            aset_expand = st.beta_expander("aligned pairs with metrics", expanded=True)
            with aset_expand:
                fig, ax = plt.subplots()
                df = pd.DataFrame(aset, columns=['y00', 'yargmax', 'cos'])
                sns.set_style("whitegrid")
                sns.scatterplot(
                    data=df,
                    x='y00',
                    y='yargmax',
                    hue='cos',
                    size='cos',
                    sizes=(1, 120),
                )

                ax.set_xlabel(f"{tgt_file}")
                ax.set_ylabel(f"{src_file}")
                ax.set_title("aligned pairs with cosine similarity")

                ax.grid()
                st.pyplot(fig)
            # """

            try:
                aligned_blocks = align_texts(aset, src_blocks, tgt_blocks)  # -> texts
            except Exception as e:
                logger.error(e)
                st.error(e.__str__() + " -- Something has gone awry, quitting")
                # st.stop()
                raise SystemExit(e) from e
                # return

            aligned_expander = st.beta_expander("aligned blocks", expanded=False)
            with aligned_expander:
                _ = pd.DataFrame(aligned_blocks)
                st.table(color_table_applymap(_))

            logger.debug(" **aligned_blocks[:15]** \n%s", aligned_blocks[:15])

            logger.debug(" Resetting file1_flag, file2_flag")
            file1_flag, file2_flag = False, False
        # end of "if submitted1:"


if __name__ == "__main__":
    main()

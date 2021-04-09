"""Showcase, based on st-bee2-aligner.py."""
from typing import List

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

# use sentence_splitter if supported
LANG_S = ["ca", "cs", "da", "nl", "en", "fi", "fr", "de",
          "el", "hu", "is", "it", "lv", "lt", "no", "pl",
          "pt", "ro", "ru", "sk", "sl", "es", "sv", "tr"]

# FLAGS = flags.FLAGS
# flags.DEFINE_boolean("debug", False, "Show debug messages", short_name="d")

LOGLEVEL = 10
logzero.loglevel(LOGLEVEL)


@st.cache
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
    src_file = src_fileio.getvalue()
    if isinstance(src_file, bytes):
        src_file = src_file.decode("utf8")

    src_text = split_text(src_file)

    tgt_file = tgt_fileio.getvalue()
    if isinstance(tgt_file, bytes):
        tgt_file = tgt_file.decode("utf8")

    lang2 = Detector(tgt_file).language.code

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
            est_time = round(len1/32) + bool(len1 % 32)
            est_time += round(len2/32) + bool(len2 % 32)
            est_time *= 13/60

            st.info([
                f" file1: {len(src_text)} paras",
                f" file2: {len(tgt_text)} paras",
            ])
    return src_text, tgt_text


def front_cover():
    # global src_fileio, op_mode, model_url
    # return src_fileio, tgt_fileio, op_mode, model_url

    src_fileio = ""
    tgt_fileio = ""
    op_mode = ""
    model_url = ""

    st.sidebar.markdown("# Streamlit powered bumblebee-ng aligner")
    sb_tit_expander = st.sidebar.beta_expander("More info (click to toggle)", expanded=True)
    with sb_tit_expander:
        st.write("(refined, quasi-prodction-ready:sunglasses:)")
        # branch
        st.write(
            "What would you like to do?"
            " (one para/send takes about 1s)"
        )

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

    # model_url
    _ = """
    if model_sel in ["Model 1"]:
        model_url = None
        # default to preset url "http://216.24.255.63:8000/text/"
    else:
        model_url = "http://216.24.255.63:8008/text/"
    # """

    # st.sidebar.subheader("Pick two files")
    sb_pick_filels = st.sidebar.beta_expander("Pick two files", expanded=True)
    with sb_pick_filels:
        src_fileio = st.file_uploader("Choose a file (utf8 txt)", type=['txt',], key="src_text")
        tgt_fileio = st.file_uploader("Choose a file (utf8 txt)", type=['txt',], key="tgt_text")

    return src_fileio, tgt_fileio, op_mode, model_url


def align_paras_sents(src_text, tgt_text, src_fileio, tgt_fileio, model_url):
    """TODO."""
    # global tgt_fileio
    # ("Para/Sent Align", "Simple Sent Align")
    # if op_mode in ["Para/Sent Align"]:
    if not (src_text and tgt_text):
        # st.info("Pick two (non-empty) files first")
        instruction1()
    else:
        # st.write(f" Processing... first run can take a while, ~{2 * len_ // 100 + 1}-{3 * len_ // 100 + 1}  min. Please wait...")

        # st.write(f" Processing... first run can take a while, ~{est_time:.1f}  min. Please wait...")

        try:
            # cos_mat = np.asarray(bee_corr(src_text, tgt_text, url=model_url))
            cos_mat = np.asarray(align_texts(src_text, tgt_text))
        except Exception as exc:
            st.write("exc: %s" % exc)
            st.stop()
            raise SystemExit(1) from exc

        st.markdown("### cosine similarity matrix")
        st.dataframe(pd.DataFrame(cos_mat).style.highlight_max(axis=0))

        fig, ax = plt.subplots()
        # fig = plt.figure()  # (figsize= (10,7))

        # cmap = sns.diverging_palette(20, 220, as_cmap=True)
        sns.heatmap(cos_mat, vmin=0, vmax=1)

        # plt.xlabel("file2")
        # plt.ylabel("file1")
        plt.xlabel(f"{tgt_fileio.name}")
        plt.ylabel(f"{src_fileio.name}")
        plt.title("cosine similarity heatmap")
        st.pyplot(fig)

        # plt.close()

        # st.markdown("## fine-tune alignment")
        st.header("fine-tune alignment")

        thr = st.slider(
            "threshold (0...1) -- drag or click to adjust threshold value",
            min_value=0.,
            max_value=1.,
            value=0.5,
            step=0.05,
            key="thr-slider",
        )

        # p_list = bee_aligner(src_text, tgt_text, cos_mat=cos_mat, thr=thr)
        p_list = align_texts(src_text, tgt_text, cos_mat=cos_mat)

        if p_list is not None:
            df = pd.DataFrame(p_list)
            st.markdown("#### para alignment at a glance")
            st.info("(hove over a cell to disply full text)")

            st.dataframe(df)
            # st.dataframe(df.style.highlight_max(axis=0))

            st.subheader("para alignment in detail")
            if st.checkbox(f"tick to show detailed alignment (threhold={thr})", value=0, key="para_df"):
                st.table(df)

            s_df = color_table_applymap(df)

            st.sidebar.subheader("aligned paras xlsx file for downloading")
            st.sidebar.markdown(get_table_download_link(s_df), unsafe_allow_html=True)

        # para-sent
        if st.sidebar.checkbox(
            " align sent within paras already aligned",
            value=0,
            key="para-sent-align",
        ):
            if p_list is None:
                st.info(" align paras first")
            else:
                st.subheader(" Aligning sents ")
                # st.info(" TODO ")

                # ====
                then = default_timer()
                s_list = plist_to_slist(p_list)
                thr_s = [""] * len(s_list)
                c_mat = [""] * len(s_list)

                # final_aligned = [[], [], []]
                final_aligned = []

                # for elm in range(2):
                for elm in range(len(c_mat)):
                    _ = """
                    thr_s[elm] = st.slider(
                        "",
                        min_value=0.,
                        max_value=1.,
                        value=0.5,
                        step=0.05,
                        key=f"thr-slider-{elm}",
                    )
                    st.write(f"{elm+1}/{len(c_mat)}", thr_s[elm])
                    # """

                    st.write(f"{elm + 1}/{len(c_mat)}")
                    _ = int(elm)
                    thr_s[_] = 0.5  # type: ignore

                    # c_mat[elm] = bee_corr(s_list[elm][0], s_list[elm][1])
                    c_mat[elm] = bee_corr(  # type: ignore
                        s_list[elm][0],
                        s_list[elm][1],
                        # src_lang=lang1,
                        # tgt_lang=lang2,
                        url=model_url
                    )
                    # s_list_aligned = bee_aligner(
                    s_list_aligned = align_texts(
                        s_list[elm][0],
                        s_list[elm][1],
                        cos_mat=c_mat[elm],
                        # thr=thr_s[elm],
                    )

                    st.table(pd.DataFrame(s_list_aligned))

                    # final_aligned[0].extend([elm[0] for elm in s_list_aligned])
                    # final_aligned[1].extend([elm[1] for elm in s_list_aligned])
                    # final_aligned[2].extend([elm[2] for elm in s_list_aligned])
                    # [*zip(final_aligned[0], final_aligned[1], final_aligned[2])]
                    final_aligned.extend(s_list_aligned)

                logger.debug("total sents: %s", len(final_aligned))

                st.write(f"Time spent for sent alignment: {(default_timer() - then) / 60:.2f} min")
                st.write(f"Total sent pairs: {len(final_aligned)}")

                st.subheader("aligned sentences in one batch")
                df_sents = pd.DataFrame(final_aligned)
                # s_df_sents = color_table_applymap(df_sents)
                s_df_sents = df_sents

                if st.checkbox("Tick to show", value=0, key="finall align sentences"):
                    # st.table(final_aligned)
                    st.table(s_df_sents)

                logger.debug("aligned sents ready for downloading")

                st.sidebar.subheader("Aligned sents in xlsx for downloading")
                st.sidebar.markdown(get_table_download_link_sents(s_df_sents), unsafe_allow_html=True)

                # ====


def align_sents():
    st.write("align sents...")
    return

    # align sents
    # if st.sidebar.checkbox(
        # "tick to proceed with sent alignment (w/o para align)",
        # value=False,
        # key="sent-align",
    # ):
    # ("Para/Sent Align", "Simple Sent Align")

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
        est_time1 *= 7/60

        st.info(f"The first run may take a while, about {est_time1:.1f} min")
        try:
        #     cos_mat = np.asarray(bee_corr(src_text, tgt_text))
            cos_mat1 = np.asarray(bee_corr(sents1, sents2, url=model_url))
        except Exception as exc:
            # st.write("exc: ", exc)
            logger.error("exc: %s" % exc)
            st.stop()
            raise SystemExit(1) from Exception

        st.markdown("### cosine similarity matrix (sents)")
        st.dataframe(pd.DataFrame(cos_mat1.round(2)).style.highlight_max(axis=0))

        mean1 = cos_mat1.mean()
        var1 = cos_mat1.var()

        fig, ax = plt.subplots()
        # fig = plt.figure()  # (figsize= (10,7))

        # cmap = sns.diverging_palette(20, 220, as_cmap=True)
        sns.heatmap(cos_mat1, vmin=0, vmax=1)

        plt.xlabel(f"{tgt_fileio.name}")
        plt.ylabel(f"{src_fileio.name}")
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
        st.markdown("* Follow steps given in the left sidebar\n"
        "*   Click expanders\n +: to reveal more details; -: to hide them \n"
        f"* bumblebee-ng v.{__version__} from mu@qq41947782's keyboard in cyberspace. Join **qq group 316287378** for feedback and questions or to be kept updated. This web version of bumblebee is the twin brother of **desktop bumblebee**.")

    logger.debug("back_cover exit")


def main():
    """Run main."""
    # print(args)

    pd.set_option('precision', 2)
    pd.options.display.float_format = '{:,.2f}'.format

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logging.basicConfig(
        # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        # datefmt='%m-%d %H:%M',
    # )

    # logging.debug("debug main start")
    # logging.info("info main start")

    logger.debug(" main started...")

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
    src_text = []
    tgt_text = []

    src_fileio, tgt_fileio, op_mode, model_url = front_cover()
    logger.debug(" front_cover done...")

    fileio_slot = st.empty()

    if src_fileio is None or tgt_fileio is None:
        logger.debug(" fileio not ready...")
        files = f"{src_fileio.name if src_fileio else ''} {tgt_fileio.name if tgt_fileio else ''}"
        if not files.strip():
            files = "None"
        fileio_slot.text(f" available files: {files}, upload or re-upload files or refresh page")
        return

    # st.write("src_text (paras):", len(src_text), "tgt_text (paras): ", len(tgt_text))

    # fetch file contents
    src_text, tgt_text = fetch_file_contents(src_fileio, tgt_fileio)

    src_file = src_fileio.name
    tgt_file = tgt_fileio.name
    src_plen = len(src_text)
    tgt_plen = len(tgt_text)

    # srt_text tgt_text defined

    fileio_slot.text(f"{src_file} ({src_plen} paras) {tgt_file} ({tgt_plen} paras) ready")
    logger.debug(" src_text: %s, tgt_text: %s ", len(src_text), len(tgt_text))

    if not src_text or not src_text:
        if not src_text:
            st.warning("Source file is apparently empty")
        if not tgt_text:
            st.warning("Target file is apparently empty")
        return

    text_info_exp = st.beta_expander("text info and samples", expanded=False)
    with text_info_exp:
        # st.info(f"{src_fileio.name} {tgt_fileio.name} read in")
        st.write(f"{src_fileio.name} {tgt_fileio.name} read in")
        st.write("number of src-text tgt-text paras:", len(src_text), len(tgt_text))
        st.write(src_text[-3:], tgt_text[-3:])

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

    tot_len = len(src_blocks) + len(tgt_blocks)
    len_ = len(src_blocks)
    tot = len_ // 32 + bool(len_ % 32)
    len_ = len(tgt_blocks)
    tot += len_ // 32 + bool(len_ % 32)

    tot_time = round(tot * 0.5, 1)

    _ = f"blocks to process: {tot_len}, eta: {tot_time} min."
    blocks_exp = st.beta_expander(_, expanded=False)
    with blocks_exp:
        st.write(src_blocks)
        st.write(tgt_blocks)

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
        logger.debug("  %s, %s", idx, idx / tot)
        try:
            _ = embed_text(elm)
        except Exception as e:
            st.write(e)
            _ = [[str(e)] + [""] * 31]
        src_embed.extend(_)
        pbar.progress(idx/tot)

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

    heatmap_exp = st.beta_expander("heatmap", expanded=False)
    with heatmap_exp:
        fig, ax = plt.subplots()

        # sns.heatmap(cmat, ax=ax, cmap="hot")
        # sns.heatmap(cmat, ax=ax, cmap="gist_heat_r")
        # sns.heatmap(cmat)
        sns.heatmap(cmat, cmap="Blues")

        ax.invert_yaxis()

        ax.set_xlabel(f"{tgt_fileio.name}")
        ax.set_ylabel(f"{src_fileio.name}")
        ax.set_title("cosine similarity heatmap")
        st.pyplot(fig)

    # st.write(cmat)
    # st.write(" plotted")

    # st.markdown("### cosine similarity matrix")
    # st.dataframe(pd.DataFrame(cmat).style.highlight_max(axis=0))

    pset = find_pairs(cmat, 3)  # pair set with metrics
    st.write(f"{len(pset)} 'good' pairs found", f"{round(len(pset) / len_, 2) * 100}%")

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

        ax.set_xlabel(f"{tgt_fileio.name}")
        ax.set_ylabel(f"{src_fileio.name}")
        ax.set_title("aligned pairs with cosine similarity")

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

    logzero.loglevel(LOGLEVEL)
    tset = cmat2tset(cmat)

    logger.debug(" **tset**: \n%s", tset)
    logger.debug(" **iset**: \n%s", iset)

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

        ax.set_xlabel(f"{tgt_fileio.name}")
        ax.set_ylabel(f"{src_fileio.name}")
        ax.set_title("interpolated pairs with cosine similarity")

        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.grid(True)
        st.pyplot(fig)
    # """

    logzero.loglevel(LOGLEVEL)

    logger.debug("pset: \n%s", pset)

    # st.write(pset)

    src_len = len(src_blocks)
    tgt_len = len(tgt_blocks)
    aset = gen_aset(pset, src_len, tgt_len)

    # st.write("\n aset ", len(aset))
    # st.write(pd.DataFrame(aset))

    logger.debug("aset: %s", aset)
    # print("aset: ", aset)

    # return

    # st.write(pd.DataFrame(aset))
    # s_df = color_table_applymap(df)

    aset_exp = st.beta_expander(" aset ", expanded=False)
    with aset_exp:
        df = pd.DataFrame(aset, columns=['zh', 'en', 'cos'])
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

        ax.set_xlabel(f"{tgt_fileio.name}")
        ax.set_ylabel(f"{src_fileio.name}")
        ax.set_title("aligned pairs with cosine similarity")

        ax.grid()
        st.pyplot(fig)
    # """

    aligned_blocks = align_texts(aset, src_blocks, tgt_blocks)  # -> texts

    aligned_expander = st.beta_expander("aligned blocks", expanded=False)
    with aligned_expander:
        _ = pd.DataFrame(aligned_blocks)
        st.table(color_table_applymap(_))

    logger.debug(" **aligned_blocks[:50]** \n%s", aligned_blocks[:50])
    # print("**aligned_blocks**", aligned_blocks)

main()

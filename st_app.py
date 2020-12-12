"""

based on st-bee2-aligner.py
"""

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

import logzero
from logzero import logger

from bee_aligner.color_table_applymap import color_table_applymap
from bee_aligner.fetch_sent_corr import fetch_sent_corr
# from bee_aligner.plist_to_slist import plist_to_slist
from bee_aligner.single_or_dual import single_or_dual
from bee_aligner.plist_to_slist import plist_to_slist
from bee_aligner.text_to_plist import text_to_plist
from bee_aligner.bee_aligner import bee_aligner

logzero.loglevel(20)

# use sentence_splitter if supported
LANG_S = ["ca", "cs", "da", "nl", "en", "fi", "fr", "de",
          "el", "hu", "is", "it", "lv", "lt", "no", "pl",
          "pt", "ro", "ru", "sk", "sl", "es", "sv", "tr"]


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


def front_cover():
    global src_fileio, op_selectbox

    # st.sidebar.markdown("# web bumblebee aligner")
    st.sidebar.title("streamlit powered bumblebee-ng aligner ")
    # st.sidebar.markdown("total # of paras limited to 300")

    # branch
    st.sidebar.subheader("What would you like to do?")
    op_selectbox = st.sidebar.selectbox(
        "",
        ("Para/Sent Align", "Simple Sent Align")
    )
    st.success(op_selectbox)

    # st.sidebar.subheader("Select a model")
    mod_selectbox = st.sidebar.selectbox(
        "Select a model (optional)",
        ("Model 1", "Google USE")
    )
    st.success(mod_selectbox)

    # model_url
    if mod_selectbox in ["Model 1"]:
        model_url = None
        # default to preset url "http://216.24.255.63:8000/text/"
    else:
        model_url = ""http://216.24.255.63:8008/text/"

    # st.markdown("## pick two files")
    st.sidebar.subheader("pick two separate files")
    # or a single dual-language file

    src_fileio = st.sidebar.file_uploader("Choose a file (utf8 txt)", type=['txt',], key="src_text")

def instruction1():
    # , or, a single file containing English and Chinese text (free format)
    st.info("Pick two files first\n\nFrom the left sidebar, pick two files, for example, one contains English text, one contains Chinese text. All files should be in **utf-8 txt** format (**no pdf, docx format etc supported**). bumblebee supports many language pairs, not just English-Chinese.")


def back_cover():
    st.markdown("--- \n* use the left sidebar to navigate\n* bumblebee-ng v.0.1.1 from mu@qq41947782's keyboard in cyberspace. Join **qq group 316287378** for feedback and questions or to be kept updated. This web version of bumblebee is the twin brother of **desktop bumblebee**.")


pd.set_option('precision', 2)
pd.options.display.float_format = '{:,.2f}'.format

def main():
    """ main.
    """

    front_cover()

    p_list = None
    p_list1 = None
    file1_flag, file2_flag = False, False

    # fetch file contents
    if src_fileio is not None:
        file1_flag = True
        src_file = src_fileio.getvalue()
        if isinstance(src_file, bytes):
            src_file = src_file.decode("utf8")

        lang1 = Detector(src_file).language.code

        src_text = split_text(src_file)

        # s_or_d = single_or_dual(src_file)
        # st.info(["1st file: single or dual lang? ", str(s_or_d), len(s_or_d)])

        # if len(s_or_d) < 2:  # single dual-lang file

        tgt_fileio = st.sidebar.file_uploader("Choose another file", type=['txt',], key="tgt_text")

        if tgt_fileio is not None:
            file2_flag = True
            tgt_file = tgt_fileio.getvalue()
            if isinstance(tgt_file, bytes):
                tgt_file = tgt_file.decode("utf8")

            lang2 = Detector(tgt_file).language.code

            tgt_text = split_text(tgt_file)

            len_ = len(src_text) + len(tgt_text)
            if len_ > 300:
                st.markdown("Sorry, this will likely hog the petite server to it's virtual death (total number paragraphs limited to **300** ). We'll trim both texts to 50 and have a testrun. ")
                src_text = src_text[:150]
                tgt_text = tgt_text[:150]
                len_ = len(src_text) + len(tgt_text)

            logger.info("total paras: %s", len_)

            _ = 3
            st.subheader(f"first {_} paras in file1 and file 2, respectively")
            st.write(f" {src_text[:_]} ")
            st.write(f" {tgt_text[:_]} ")
            st.subheader(f"last {_} paras in file1 and file 2, respectively")
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

        _ = '''
        else:  # dual-lang file
            st.write("""
        It looks like a dual-lang file.
        We'll implement something to handle this, soon.
            """)
            # assert 0, ""
        # '''

    # align paras
    # if st.sidebar.checkbox(
        # "tick to proceed with para alignment",
        # value=False,
        # key="para-align",
    # ):
    if op_selectbox in ["Para/Sent Align"]:  # ("Para/Sent Align", "Simple Sent Align")
        if not (file1_flag and file2_flag):
            # st.info("Pick two files first")
            instruction1()
        else:
            # st.write(f" Processing... first run can take a while, ~{2 * len_ // 100 + 1}-{3 * len_ // 100 + 1}  min. Please wait...")
            st.write(f" Processing... first run can take a while, ~{est_time:.1f}  min. Please wait...")

            try:
                cos_mat = np.asarray(bee_corr(src_text, tgt_text, url=model_url))
            except Exception as exc:
                st.write("exc: %s" % exc)
                st.stop()
                raise SystemExit(1) from Exception

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

            p_list = bee_aligner(src_text, tgt_text, cos_mat=cos_mat, thr=thr)

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
                        thr_s[elm] = 0.5

                        # c_mat[elm] = bee_corr(s_list[elm][0], s_list[elm][1])
                        c_mat[elm] = bee_corr(
                            s_list[elm][0],
                            s_list[elm][1],
                            # src_lang=lang1,
                            # tgt_lang=lang2,
                            url=model_url
                        )
                        s_list_aligned = bee_aligner(
                            s_list[elm][0],
                            s_list[elm][1],
                            cos_mat=c_mat[elm],
                            thr=thr_s[elm],
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

    # align sents
    # if st.sidebar.checkbox(
        # "tick to proceed with sent alignment (w/o para align)",
        # value=False,
        # key="sent-align",
    # ):
    if op_selectbox in ["Simple Sent Align"]:  # ("Para/Sent Align", "Simple Sent Align")
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

    back_cover()

    # st.write(src_fileio.name)

main()
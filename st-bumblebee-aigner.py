"""

based on st-bee2-aligner.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import base64
from io import BytesIO
from polyglot.text import Detector, Text

import logzero
from logzero import logger

from bee_aligner.color_table_applymap import color_table_applymap
from bee_aligner.fetch_sent_corr import fetch_sent_corr
# from bee_aligner.plist_to_slist import plist_to_slist
from bee_aligner.single_or_dual import single_or_dual
from bee_aligner.text_to_plist import text_to_plist
from bee_aligner.bee_aligner import bee_aligner

logzero.loglevel(20)


def split_text(text, sep='\n'):
    if isinstance(text, bytes):
        text = text.decode("utf8")
    return [elm.strip() for elm in text.split(sep) if elm.strip()]


@st.cache
def bee_corr(text1, text2):
    logger.debug('type: %s, text[:4]: %s', type(text1), text1[:10])
    if isinstance(text1, str):
        text1 = split_text(text1)
    if isinstance(text2, str):
        text2 = split_text(text2)

    return fetch_sent_corr(text1, text2)


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
    global src_fileio

    # st.sidebar.markdown("# web bumblebee aligner")
    st.sidebar.title("web bumblebee-ng aligner ")
    st.sidebar.markdown("total # of paras limited to 300")

    # st.markdown("## pick two files")
    st.sidebar.subheader("pick two separate files or a single dual-language file")

    src_fileio = st.sidebar.file_uploader("Choose a file (utf8 txt)", type=['txt',], key="src_text")

    # , or, a single file containing English and Chinese text (free format)
    st.info("From the left sidebar, pick two files, for example, one contains English text, one contains Chinese text. All files should be in **utf-8 txt** format (**no pdf, docx format etc supported**). bumblebee supports many language pairs, not just English-Chinese.")


def back_cover():
    st.markdown("--- \n* bumblebee-ng v.0.1.0 from mu@qq41947782's keyboard in cyberspace. Join **qq group 316287378** for feedback and questions or to be kept updated. This web version of bumblebee is the twin brother of **desktop bumblebee**.")


pd.set_option('precision', 2)


def main():
    """ main.
    """

    front_cover()

    p_list = None
    file1_flag, file2_flag = False, False

    # fetch file contents
    if src_fileio is not None:
        file1_flag = True
        src_file = src_fileio.getvalue()
        if isinstance(src_file, bytes):
            src_file = src_file.decode("utf8")

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
    if st.sidebar.checkbox(
        "tick to proceed with para alignment",
        value=False,
        key="para-align",
    ):
        if not (file1_flag and file2_flag):
            st.info("Pick two files first")
        else:
            # st.write(f" Processing... first run can take a while, ~{2 * len_ // 100 + 1}-{3 * len_ // 100 + 1}  min. Please wait...")
            st.write(f" Processing... first run can take a while, ~{est_time:.1f}  min. Please wait...")

            try:
                cos_mat = np.asarray(bee_corr(src_text, tgt_text))
            except Exception as exc:
                st.write("exc: ", exc)
                raise SystemExit(1) from Exception

            st.markdown("### cosine similarity matrix")
            st.dataframe(pd.DataFrame(cos_mat))

            fig, ax = plt.subplots()
            # fig = plt.figure()  # (figsize= (10,7))

            # cmap = sns.diverging_palette(20, 220, as_cmap=True)
            sns.heatmap(cos_mat, vmin=0, vmax=1)

            plt.xlabel("file2")
            plt.ylabel("file1")
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
                st.subheader("para alignment in detail")
                if st.checkbox(f"tick to show detailed alignment (threhold={thr})", value=0, key="para_df"):
                        st.table(df)

                s_df = color_table_applymap(df)

                st.sidebar.subheader("aligned paras xlsx file for downloading")
                st.sidebar.markdown(get_table_download_link(s_df), unsafe_allow_html=True)

    # align sents
    if st.sidebar.checkbox(
        "tick to proceed with sent alignment",
        value=False,
        key="sent-align",
    ):
        if file1_flag and file2_flag:
            st.info(" to be implemented ")
        else:
            st.info("Pick two files first")

    back_cover()


main()

"""
[Discuss 1445](https://discuss.streamlit.io/t/uploading-multiple-files-with-file-uploader/1445)
"""
# pylint: enable=line-too-long
from typing import Dict

import cchardet
import streamlit as st

from logzero import logger


@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}


@st.cache(allow_output_mutation=True)
def get_src_file() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}

@st.cache(allow_output_mutation=True)
def get_files() -> list:
    """This list is initialized once and can be used to store the files uploaded."""
    return []


@st.cache(allow_output_mutation=True)
def get_tgt_file() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded."""
    return {}

st.set_page_config(
     page_title="Ex-stream-ly Cool App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)


def main():
    """Run this function to run the app."""

    logger.info(" ----main entry.----")

    static_store = get_static_store()
    files = get_files()

    logger.info("len: %s, static_store: %s", len(static_store), static_store)

    st.info(__doc__)
    # result = st.file_uploader("Upload", type="py")

    src_f =  st.file_uploader(
        "Source file",
        type="txt",
        accept_multiple_files=True
    )
    if src_f:
        src_filename = src_f[-1].name
        src_file = src_f[-1].getvalue()
        src_file = src_file.decode(cchardet.detect(src_file).get('encoding', 'utf8'))

    tgt_f =  st.file_uploader(
        "Target file",
        type="txt",
        accept_multiple_files=True
    )
    if tgt_f:
        tgt_filename = tgt_f[-1].name
        tgt_file = tgt_f[-1].getvalue()
        tgt_file = tgt_file.decode(cchardet.detect(tgt_file).get('encoding', 'utf8'))

    if not (src_f and tgt_f):
        return

    files = st.empty()
    # either of both empty
    if not (src_file.strip() and tgt_file.strip()):
        # files.write("At least one of the files contains nothing.")
        if not src_file.strip() and not tgt_file.strip():
            files.write("Both files empty!")
        elif not src_file.strip():
            files.write("Source file empty!")
        else:
            files.write("Target file empty!")

        return

    # overwrite possible previous message
    files.write("")

    result = st.file_uploader(
        "Upload",
        type="py",
        accept_multiple_files=True
    )
    if result:
        # Process you file here

        logger.debug("result: %s", result)
        logger.debug("[elm.name for elm in result]: %s", [elm.name for elm in result])

        # names = [elm.name for elm in result]
        # values = [elm.getvalue() for elm in result]
        names = []
        values = []
        for elm in result:
            if elm.getvalue() in values:
                continue
            names.append(elm.name)
            values.append(elm.getvalue())

        # And add it to the static_store if not already in
        # result must be hashable?

        # if not value in static_store.values(): static_store[result] = value
    else:
        static_store.clear()  # Hack to clear list if the user clears the cache and reloads the page
        # files.clear()
        st.info("Upload one or more `.py` files.")

    if st.button("Clear file list"):
        static_store.clear()
        # files.clear()
    if st.checkbox("Show file list?", True):
        # st.write(list(static_store.keys()))
        # st.write([[k, v] for k, v in static_store. items()])

        # st.write(static_store)
        if "names" in locals():
            st.write(names)

    if st.checkbox("Show content of files?"):
        # for value in static_store.values(): st.code(value)
        if "values" in locals():
            for value in values:
                st.code(value)

    # logger.info("len: %s, static_store: %s", len(static_store), static_store)

    logger.info("main exit.")
    # logger.info("len: %s", len(static_store))

    if "names" in locals() and "values" in locals():
        logger.debug("%s, %s", len(names), len(values))

    if "src_filename" in locals():
        if st.checkbox("Show content of src_file?"):
            st.code(src_file)

    if "tgt_filename" in locals():
        if st.checkbox("Show content of tgt_file?"):
            st.code(tgt_file)

    reset_btn = st.button("Reset")
    if reset_btn:
        logger.debug(" Reset pressed")
        return None
    else:
        logger.debug(" Reset **not** pressed!")
        st.stop()

main()


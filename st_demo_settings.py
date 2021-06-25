"""Bumblebee-ng using state.

https://gist.github.com/okld/0aba4869ba6fdc8d49132e6974e2e662#file-st_demo_settings-py
"""
__version__ = "0.1.1b"

import streamlit as st
from streamlit.hashing import _CodeHasher

from get_state import get_state

_ = """
try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server
# """

st.set_page_config(
     page_title=f"Bumblebee-ng v{__version__}",
     page_icon="🧊",
     layout="wide",
     # initial_sidebar_state="expanded",
)


def main():
    # state = _get_state()
    state = get_state()
    pages = {
        "Dashboard": page_dashboard,
        "Settings": page_settings,
    }

    st.sidebar.title(":floppy_disk: Page states")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def page_dashboard(state):
    st.title(":chart_with_upwards_trend: Dashboard page")
    sb_pick_filels = st.beta_expander("Pick two files", expanded=True)
    with sb_pick_filels:
        state.src_fileio = st.file_uploader("Upload source file (utf8 txt)", type=['txt',], key="src_text", accept_multiple_files=True)
        state.tgt_fileio = st.file_uploader("Upload target file (utf8 txt)", type=['txt',], key="tgt_text", accept_multiple_files=True)
    if state.src_fileio:
        state.src_file = state.src_fileio[-1].getvalue().decode()
        state.src_filename = state.src_fileio[-1].name
    if state.tgt_fileio:
        state.tgt_file = state.tgt_fileio[-1].getvalue().decode()
        state.tgt_filename = state.tgt_fileio[-1].name
        
    display_state_values(state)


def page_settings(state):
    st.title(":wrench: Settings")
    display_state_values(state)

    # state.file1 = ""

    # options = ["Hello", "World", "Goodbye"]
    # options = ["src_file", "tgt_file", "World", "Hello", "Goodbye"]

    state.input = st.text_input("Set input value.", state.input or "")
    state.slider = st.slider("Set slider value.", 1, 10, state.slider)

    options = ["src_file", "tgt_file", ]
    _ = """
    # src_fileio tgt_fileio
    sb_pick_filels = st.beta_expander("Pick two files", expanded=True)
    with sb_pick_filels:
        state.radio = st.radio("Select", options, options.index(state.radio) if state.radio else 0)
        if state.radio in ["src_file"]:
            state.src_fileio = st.file_uploader("Upload source file (utf8 txt)", type=['txt',], key="src_text", accept_multiple_files=True)
        else:
            state.tgt_fileio = st.file_uploader("Upload target file (utf8 txt)", type=['txt',], key="tgt_text", accept_multiple_files=True)
    # """

    sb_pick_filels = st.beta_expander("Pick two files", expanded=True)
    with sb_pick_filels:
        state.src_fileio = st.file_uploader("Upload source file (utf8 txt)", type=['txt',], key="src_text", accept_multiple_files=True)
        state.tgt_fileio = st.file_uploader("Upload target file (utf8 txt)", type=['txt',], key="tgt_text", accept_multiple_files=True)

    if state.src_fileio:
        state.src_file = state.src_fileio[-1].getvalue().decode()
        state.src_filename = state.src_fileio[-1].name
    if state.tgt_fileio:
        state.tgt_file = state.tgt_fileio[-1].getvalue().decode()
        state.tgt_filename = state.tgt_fileio[-1].name


    st.write("---")

    state.checkbox = st.checkbox("Set checkbox value.", state.checkbox)
    state.selectbox = st.selectbox("Select value.", options, options.index(state.selectbox) if state.selectbox else 0)
    state.multiselect = st.multiselect("Select value(s).", options, state.multiselect)

    # Dynamic state assignments
    for i in range(3):
        key = f"State value {i}"
        state[key] = st.slider(f"Set value {i}", 1, 10, state[key])


def display_state_values(state):
    st.write("Source filename:", state.src_filename)
    st.write("Target filename:", state.tgt_filename)

    if state.src_file:
        st.write("src_file[:100]:", state.src_file[:100])

    if state.tgt_file:
        st.write("tgt_file[:100]:", state.tgt_file[:100])

    st.write("Input state:", state.input)
    # _ = """
    st.write("Slider state:", state.slider)
    st.write("Radio state:", state.radio)
    st.write("Checkbox state:", state.checkbox)
    st.write("Selectbox state:", state.selectbox)
    st.write("Multiselect state:", state.multiselect)

    for i in range(3):
        st.write(f"Value {i}:", state[f"State value {i}"])
    # """
    if st.button("Clear state"):
        state.clear()




if __name__ == "__main__":
    main()
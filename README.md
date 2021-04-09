<!---
https://www.analyticsvidhya.com/blog/2021/01/an-ode-to-model-deployment-using-streamlit-open-sourcing-typing-tutor-for-programmers/
   setup streamlit on colab
    codes: https://github.com/faizankshaikh/AV_Article_Codes/tree/master/Typing_Tutor/original
        from pyngrok import ngrok

        public_url = ngrok.connect(port='8501')
        public_url
--->
# streamlit bumblebee aligner ![build](https://github.com/ffreemt/st-bumblebee-aligner/workflows/build/badge.svg)[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/ffreemt/st-bumblebee-aligner/dev/st_app.py)

a streamlit powered bumblebee aligner (refined and production-ready)

## Pre-installation of `pyicu`, `pycld2` and `morfessor`

### For Linux/OSX

1.   Install `libicu`
E.g.
*   Ubuntu: `sudo apt install libicu-dev`
*   Centos: `yum install libicu`
*   OSX: `brew install icu4c`

2.   Then install `pyicu`, `pycld2` and `morfessor`, e.g.,
```
  pip install pyicu pycld2 morfessor
```

### For Windows

Download and install the `pyicu`, `pycld2` and `morfessor` whl packages for your OS and Python versions from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyicu and https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycld2 and https://www.lfd.uci.edu/~gohlke/pythonlibs/#morfessor e.g.
```
pip install pyicu...whl pyicld2....whl morfessro...whl
```

## Installation
From a command prompt, run

```bash
git clone https://github.com/ffreemt/st-bumblebee-aligner
cd st-bumblebee-aligner
```

with `pip`, run
```
pip install -r requirements.txt
```
<details>
<summary>or with ``poetry``, run</summary>

```
poetry install
```
</details>

## Usage
```bash
streamlit run st-bumblebee-aligner.py
```
<details><summary>or</summary>

```
python -m streamlit run st-bumblebee-aligner.py
```
</details>

* `st-bumblebee-aligner-v0.1.0.py` is the same as `st-bumblebee-aligner.py`
   * Simple para alignment or sent alignment

* `st-bumblebee-aligner-v0.1.1.py`
  * introduced sent alignment within paras aligned

To try `st-bumblebee-aligner-v0.1.1.py`, do
```
streamlit run st-bumblebee-aligner-v0.1.1.py
```

Point your browser to `http://127.0.0.1:8501`
 and follow instructions.

## Notes
`numpy` 1.19.4 is known to cause some problems, refer to [https://github.com/numpy/numpy/wiki/FMod-Bug-on-Windows](https://github.com/numpy/numpy/wiki/FMod-Bug-on-Windows). Pin `numpy` to version `1.19.3` may get rid of the problems, e.g.,
```
pip install -U numpy==1.19.3
```
or
```
poetry add numpy==1.19.3
```
or try newer `numpy`.

<!---
https://docs.streamlit.io/en/stable/deploy_streamlit_app.html
# Matplotlib doesn’t work well with threads. So if you’re using Matplotlib you should wrap your code with locks as shown in the snippet below.

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

with _lock:
  fig.title('This is a figure)')
  fig.plot([1,20,3,40])
  st.pyplot(fig)

st.beta_columns
st.beta_expander
st.beta_container

--->
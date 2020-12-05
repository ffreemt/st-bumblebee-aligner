# streamlit bumblebee aligner
a streamlit powered bumblebee aligner

### Pre-installation of `pyicu`, `pycld2` and `morfessor`

###### For Linux/OSX

1. Install `libicu`
E.g.
* Ubuntu: `sudo apt install libicu-dev`
* Centos: `yum install libicu`
* OSX: `brew install icu4c`

2. Then install `pyicu`, `pycld2` and `morfessor`, e.g.,
```
  pip install pyicu pycld2 morfessor
```

###### For Windows

Download and install the `pyicu`, `pycld2` and `morfessor` whl packages for your OS and Python versions from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyicu and https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycld2 and https://www.lfd.uci.edu/~gohlke/pythonlibs/#morfessor e.g.
```
pip install pyicu...whl pyicld2....whl morfessro...whl
```

### Installation
From a command prompt, run

```bash
git clone https://github.com/ffreemt/st-bumblebee-aligner
cd st-bumblebee-aligner
```

with `pip`, run
```
pip install -r requirements.txt
```
or with `poetry`, run
```
poetry install
```

### Usage
```bash
streamlit run st-bumblebee-aligner.py
```

Point your browser to `http://127.0.0.1:8501`

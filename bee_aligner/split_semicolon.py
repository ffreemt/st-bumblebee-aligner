"""Split text at semicolons."""
from typing import List, Union
import re


def split_semicolon(text: Union[str, List[str]]) -> List[str]:
    """Split text at semicolons.

    Args:
        text: string or list of strings

    Returns:
        List of strings
    """
    if isinstance(text, str):
        text = [text]

    res = []
    for elm in text:
        # _ = [el for el in re.split(r"(.*?;)", elm) if el]
        # _ = re.findall(r"[\s\S]*?;|[^;]+$", elm)
        # _ = re.findall(r"[^;]+(?:;|$)",  elm)
        _ = re.split(r"(?<=;)(?!$)", elm)  # need py3.7
        res.extend(_)

    return res

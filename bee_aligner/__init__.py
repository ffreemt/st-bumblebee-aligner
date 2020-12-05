''' init '''
import warnings

from bee_aligner.find_aligned_pairs import find_aligned_pairs  # noqa: F401
from bee_aligner.gen_row_alignment import gen_row_alignment  # noqa: F401

warnings.simplefilter(action='ignore', category=FutureWarning)
__date__ = '2020.12.05'
__version__ = '0.1.0'
VERSION = tuple(__version__.split('.'))

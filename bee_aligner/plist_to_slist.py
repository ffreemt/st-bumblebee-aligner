""" p_list to sent_list.

p_list = [('Wuthering Heights', '呼啸山庄', ''), ('--------------------------------------------------------------------------------', '--------------------------------------------------------------------------------', '1.00'), ('Chapter 2', '第二章', ''), ('', '英文', ''), ('Chinese', '昨天下午又冷又有雾。我想就在书房炉边消磨一下午，不想踩着杂草污泥到呼啸山庄了。', ''), ("Yesterday afternoon set in misty and cold. I had half a mind to spend it by my study fire, instead of wading through heath and mud to Wuthering Heights. On coming up from dinner, however (N.B. I dine between twelve and one o'clock; the housekeeper, a matronly lady, taken as a fixture along with the house, could not, or would not, comprehend my request that I might be served at five), on mounting the stairs with this lazy intention, and stepping into the room, I saw a servant girl on her knees surrounded by brushes and coal-scuttles, and raising an infernal dust as she extinguished the flames with heaps of cinders. This spectacle drove me back immediately; I took my hat, and, after a four-miles' walk, arrived at Heathcliff's garden gate just in time to escape the first feathery flakes of a snow shower.", '但是，吃过午饭（注意——我在十二点与一点钟之间吃午饭，而可以当作这所房子的附属物的管家婆，一位慈祥的太太却不能，或者并不愿理解我请求在五点钟开饭的用意），在我怀着这个懒惰的想法上了楼，迈进屋子的时候，看见一个女仆跪在地上，身边是扫帚和煤斗。她正在用一堆堆煤渣封火，搞起一片弥漫的灰尘。这景象立刻把我赶回头了。我拿了帽子，走了四里路，到达了希刺克厉夫的花园口口，刚好躲过了一场今年初降的鹅毛大雪。', '0.93'), ('On that bleak hill top the earth was hard with a black frost, and the air made me shiver through every limb. Being unable to remove the chain, I jumped over, and, running up the flagged causeway bordered with straggling gooseberry bushes, knocked vainly for admittance, till my knuckles tingled and the dogs howled.', '在那荒凉的山顶上，土地由于结了一层黑冰而冻得坚硬，冷空气使我四肢发抖。我弄不开门链，就跳进去，顺着两边种着蔓延的醋栗树丛的石路跑去。我白白地敲了半天门，一直敲到我的手指骨都痛了，狗也狂吠起来。', '0.91'), ]  # pylint: disable=line-too-long
"""

from typing import List, Tuple

from threading import currentThread

from polyglot.text import Text, Detector
from sentence_splitter import split_text_into_sentences
import blinker

import logzero
from logzero import logger

# fmt: off
# use sentence_splitter if supported
LANG_S = ["ca", "cs", "da", "nl", "en", "fi", "fr", "de", "el", "hu", "is", "it", "lv", "lt", "no", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv", "tr"]
# fmt: on

SIG_ALIGNER = blinker.signal("aligner")


def split_text(text: str, lang: str) -> List[str]:
    """ split text to sentences.

    use sentence_splitter if supported,
    else use polyglot.text.Text
    """
    if lang in LANG_S:
        return split_text_into_sentences(text, lang)

    return [elm.string for elm in Text(text, lang).sentences]


# fmt: off
def plist_to_slist(
        p_list: List[Tuple[str, str, float]],
        lang0: str = "",
        lang1: str = "",
) -> List[Tuple[str, str, float]]:
    # ):
    # fmt: on
    """ para_list to sent_list. """

    c_th = currentThread()

    # logzero.loglevel(10)
    logzero.loglevel(20)

    # convert 3rd column's nonempty str to float
    prob = [elm[2] if elm[2] == "" else float(elm[2]) for elm in p_list]

    logger.debug("prob: %s", prob)

    if lang0 in [""]:
        lang0 = Detector(" ".join([elm[0] for elm in p_list[:100]])).language.code
    if lang1 in [""]:
        lang1 = Detector(" ".join([elm[1] for elm in p_list[:100]])).language.code

    # lang0 = Detector(" ".join([elm[0] for elm in p_list[:100]])).language.code
    # lang1 = Detector(" ".join([elm[1] for elm in p_list[:100]])).language.code

    idx_lst = [[]]
    # t_f = [*map(lambda x: not isinstance(x, float), prob[:])]; idx, elm = i, t_f[i]
    for idx, elm in enumerate(map(lambda x: not isinstance(x, float), prob[:])):
        idx_lst = (idx_lst[:-1] + [idx_lst[-1] + [idx]]) if elm else (idx_lst + [[idx]])

    logger.debug("idx_lst: %s", str(idx_lst))

    # p_list[idx_lst[3][0]: idx_lst[3][-1]+1]
    # p_list[idx_lst[idx][0]: idx_lst[idx][-1]+1]

    sent_lst = []
    for elm in idx_lst:
        if not elm:  # bypass possible first empty list
            continue

        left, right = [], []
        for idx in range(elm[0], elm[-1] + 1):
            if p_list[idx][0]:
                # left0 = [sent.string for sent in Text(p_list[idx][0], lang0).sentences]
                left0 = split_text(p_list[idx][0], lang0)
                left.extend([s.strip() for s in left0 if s.strip()])
            if p_list[idx][1]:
                # right0 = [sent.string for sent in Text(p_list[idx][1], lang1).sentences]
                right0 = split_text(p_list[idx][1], lang1)
                right.extend([s.strip() for s in right0 if s.strip()])

            # supply "" string if nothing exists
            if not left:
                left = [""]
            if not right:
                right = [""]

        sent_lst.append([left, right])

    c_th.sent_lst = sent_lst  # type: ignore
    # SIG_ALIGNER.send("plist_to_slist", **{"sent_lst":sent_lst})
    # SIG_ALIGNER.send("plist_to_slist", sent_lst=sent_lst)

    return sent_lst

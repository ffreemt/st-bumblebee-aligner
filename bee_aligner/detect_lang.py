# coding: utf-8
r'''
Detect language using longid.classify.

detct as Chinese if chinese_char_ratio>= threthold else dectect_language()

based on mypython\detect_lang
'''

import langid
import pycountry

from logzero import logger as LOGGER

# pycountry.languages.get(alpha_2='ja').lower()
LANG_DICT = dict(
    zh='chinese',
    en='english',
    fr='french',
    it='italian',
    de='german',
    pt='portuguese',
    es='spanish',
    ko='korean',
    ja='japanese',
    ru='russian',
)


def detect_lang(text1: str, checklen: int = 3000) -> str:
    '''
    Detect Chinese and other languages.

    return a string lang for use in seg_sent(text, lang=lang)
    '''

    if not text1:
        return 'english'

    text0 = text1[:checklen]

    detected = 'en'
    try:
        detected = langid.classify(text0)[0]
    except Exception as exc:
        LOGGER.warning(" langid.classify failed: %s, falls back eo english", exc)

    # if detected not in ['zh', 'en', 'fr', 'it', 'de', 'pt', 'es']: detected = 'en'

    # lang_dict[detected]

    try:
        lang = pycountry.languages.get(alpha_2=detected).name.lower()
    except Exception as exc:
        LOGGER.warning('Unable to detect, %s, setting to english', exc)
        lang = 'english'

    return lang

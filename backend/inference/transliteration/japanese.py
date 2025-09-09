from inference.transliteration.abstract import TransliterationStrategy
import spacy
import pykakasi
from spacy.cli import download
from spacy.util import is_package

class JapaneseStrategy(TransliterationStrategy):
    def __init__(self):
        # Load heavy models once
        pkg = 'ja_core_news_lg'
        if not is_package(pkg):
                print(f"{pkg} not found — downloading…")
                download(pkg)
        self.nlp = spacy.load(pkg)
        self.kks = pykakasi.kakasi()

    def transliterate(self, sentence: str) -> str:
        doc = self.nlp(sentence)
        romaji = []
        for word in doc:
            if word.text.isascii():
                romaji.append(word.text)
            elif word.text in ("、", "。"):
                romaji.append({"、": ",", "。": "."}[word.text])
            else:
                kana = word.morph.to_dict().get('Reading', word.text)
                conv = self.kks.convert(kana)
                hepburn = " ".join(item['hepburn'] for item in conv)
                romaji.append(hepburn)
        return " ".join(romaji)
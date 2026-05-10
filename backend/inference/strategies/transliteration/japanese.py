import re
import pykakasi
from inference.strategies.abstract_strategy import AbstractStrategy

JP_BLOCK = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fffー々]+')


class JapaneseTransliterationStrategy(AbstractStrategy):
    def __init__(self):
        super().__init__(language_code="ja")

    def load_model(self) -> None:
        self.kks = pykakasi.kakasi()

    def _run_one(self, s: str) -> str:
        out = []
        i = 0
        while i < len(s):
            m = JP_BLOCK.match(s, i)
            if m:
                chunk = s[m.start():m.end()]
                pieces = self.kks.convert(chunk)
                romaji = " ".join((p.get("hepburn") or p.get("orig") or "").strip()
                                  for p in pieces if p.get("orig"))
                out.append(romaji)
                i = m.end()
            else:
                out.append(s[i])
                i += 1

        text = "".join(out)
        # Normalize Japanese punctuation to Western and tidy spaces
        text = text.replace("、", ",").replace("。", ".")
        text = re.sub(r"\s+([,.\!\?\:\;])", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

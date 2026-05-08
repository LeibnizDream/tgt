import re
from utils.functions import load_glossing_rules
from inference.strategies.glossing.spacy import SpaCyGlossingStrategy


LEIPZIG_GLOSSARY = load_glossing_rules("LEIPZIG_GLOSSARY.json")

class JapaneseGlossingStrategy(SpaCyGlossingStrategy):
    """
    A glossing strategy that either uses a default spaCy model
    or a custom one in models/glossing/, plus optional translation.
    """

    def gloss(self, sentence: str) -> str:
        doc = self.nlp(sentence)
        out_tokens = []

        # particle -> gloss mapping (surface form)
        case_gloss = {
            'が': 'NOM', 'は': 'TOP', 'の': 'GEN', 'を': 'ACC',
            'に': 'DAT', 'へ': 'ALL', 'から': 'ABL', 'で': 'INS',
        }

        # ASCII + full-width digits, common JP brackets/quotes
        SKIP_RE = re.compile(r"[()\[\]{}0-9\uFF10-\uFF19\u300C\u300D\u300E\u300F\u3010\u3011\uFF08\uFF09\u3014\u3015]")

        for token in doc:
            text = token.text
            pos = token.pos_ or "X"

            # pass through punctuation and bracket/number-like tokens
            if pos == "PUNCT" or SKIP_RE.search(text):
                out_tokens.append(text)
                continue

            # detect case particles by surface form (most reliable for JP)
            is_case_particle = text in case_gloss or (pos in {"ADP", "PART", "SCONJ"} and text in case_gloss)

            if is_case_particle:
                norm = text 
            else:
                norm = text.lower().replace(" ", ".")
                rule_feat = None

            out_tokens.append(f"{norm}-{pos}-{rule_feat}" if rule_feat else f"{norm}-{pos}")
            output = " ".join(out_tokens)
           

        return output

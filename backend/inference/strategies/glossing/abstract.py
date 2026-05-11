"""
Abstract base for all glossing strategies.

GlossingStrategy defines the interface (load_model, gloss) and provides the
Leipzig mapping utilities used by NLP-based subclasses (spaCy, Stanza).

Why load a translation model here
----------------------------------
NLP-based glossers (spaCy/Stanza) work with English dependency trees, so they
need to translate the source text before parsing.  The Marian translation model
is loaded eagerly in __init__ so that failures surface at construction time
rather than mid-batch.  LLM-based subclasses (LLMGlossingStrategy) ignore the
translation model entirely because the LLM handles both understanding and
glossing in one step.

Leipzig mapping utilities
--------------------------
map_feature and map_morph_to_leipzig convert spaCy/Stanza UD morphology dicts
into the Leipzig Glossing Rules notation (e.g. {'Case':'Nom','Number':'Sing'}
→ 'SG.NOM').  These helpers live here because every NLP-based strategy needs
them and they depend on the shared LEIPZIG_GLOSSARY constant.
"""
from inference.strategies.abstract_strategy import AbstractStrategy
from utils.functions import load_glossing_rules

LEIPZIG_GLOSSARY = load_glossing_rules("LEIPZIG_GLOSSARY.json")

# (category, UD) -> Leipzig
UD2LEIPZIG_MAP = {
    (v["category"], v["UD"]): k
    for k, v in LEIPZIG_GLOSSARY.items()
    if "category" in v and "UD" in v
}

GLOSSARY_CATEGORIES = {v["category"] for v in LEIPZIG_GLOSSARY.values() if "category" in v}

class GlossingStrategy(AbstractStrategy):
    """Interface for all glossing strategies.

    Subclasses must implement load_model() and gloss(sentence) -> str.
    Leipzig mapping helpers (map_feature, map_morph_to_leipzig) are provided
    for NLP-based subclasses; LLM-based ones can ignore them.
    """
    def __init__(self, language_code, model=None):
        self.model = model  # must be set before super().__init__ calls load_model()
        super().__init__(language_code)

    # ---------- Leipzig mapping helpers ----------
    @staticmethod
    def _map_one_atom(category: str, ud_val_atom: str) -> str:
        """Map ONE UD atom (no commas) for a given category to Leipzig; fallback to upper."""
        return UD2LEIPZIG_MAP.get((category, ud_val_atom), ud_val_atom.upper())

    @staticmethod
    def map_feature(category: str, ud_val: str | None) -> str | None:
        """
        Map a SINGLE UD feature value (possibly comma-separated) to Leipzig.
        Handles exact combined matches first (e.g., Case='Acc,Dat,Nom'), else per-atom join with '/'.
        """
        if not ud_val or ud_val == "None":
            return None

        # exact combined value (e.g., ('Case','Acc,Dat,Nom'))
        code = UD2LEIPZIG_MAP.get((category, ud_val))
        if code:
            return code

        # comma-separated atoms
        if "," in ud_val:
            atoms = [a.strip() for a in ud_val.split(",") if a.strip()]
            mapped = [GlossingStrategy._map_one_atom(category, a) for a in atoms]
            return "/".join(mapped)

        # single atom
        return GlossingStrategy._map_one_atom(category, ud_val)

    def map_morph_to_leipzig(self, morph: dict) -> str:
        """
        Convert a whole spaCy morph dict (e.g., {'Case':'Nom','Gender':'Masc','Number':'Sing'})
        into a Leipzig string (e.g., 'M-SG-NOM') using a stable feature order.
        """
        features_in_order = [
            # Core order
            "PronType", "Definite", "Gender", "Person", "Number", "Case",
            # Lexical
            "NumType", "Other", "Abbr", "ExtPos", "Clusivity",
            # Nominal
            "Animacy", "NounClass", "Deixis", "DeixisRef", "Degree",
            # Verbal
            "VerbForm", "Mood", "Tense", "Aspect", "Voice",
            "Evident", "Polarity", "Polite", "Foreign", "Abbr", "Typo",
            "Poss", 'Nomzr', 'Pos'
        ]
        # sanity check (optional)
        missing = GLOSSARY_CATEGORIES - set(features_in_order)
        if missing:
            raise ValueError(
                f"Missing categories in features_in_order: {sorted(missing)} "
                f"(update order or glossary)"
            )

        parts = []
        seen = set()

        for cat in features_in_order:
            val = morph.get(cat)
            code = self.map_feature(cat, val)
            if code:
                parts.append(code)
            seen.add(cat)

        # include any remaining features (rare)
        for cat, val in morph.items():
            if cat in seen:
                continue
            code = self.map_feature(cat, val)
            if code:
                parts.append(code)
            else:
                # fallback: keep raw for debugging visibility
                parts.append(f"{cat}={val}")

        return "-".join(parts)

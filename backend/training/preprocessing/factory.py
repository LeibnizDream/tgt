"""
Factory for selecting the correct training preprocessor.

:class:`PreProcessorFactory` maps an (action, language) pair to the
appropriate :class:`~training.preprocessing.abstract.BasePreprocessor`
subclass.  Currently only the ``"gloss"`` action is supported, which
returns a :class:`~training.preprocessing.UD.UDPreprocessor` for all
supported language codes.
"""
from training.preprocessing.UD import UDPreprocessor


class PreProcessorFactory:
    """Factory that returns a :class:`~training.preprocessing.abstract.BasePreprocessor` for a given action and language."""
    @staticmethod
    def get_preprocessor(action, language: str, study: str):
        """
        Return the correct preprocessor for the given action and language.

        Args:
            action (str): Task action, e.g. ``"gloss"``.
            language (str): ISO 639-1 language code, e.g. ``"de"``.
            study (str): Study identifier appended to output file names.

        Returns:
            BasePreprocessor: A configured preprocessor instance.

        Raises:
            ValueError: When no preprocessor exists for *action* / *language*.
        """
        if action == "gloss":
            if language in ["ca", "zh", "hr", "da", "nl", "en", "fi", "fr", "de", "el", "it", "ja", "ko",
                            "lt", "mk", "xx", "nb", "pl", "pt", "ro", "ru", "sl", "es", "sv", "uk", "af",
                            #No pretrained models for these languages
                            "sq", "am", "grc", "ar", "hy", "az", "eu", "bn", "bg", "cs", "et", "fo", "gu",
                            "he", "hi", "hu", "is", "id", "ga", "kn", "ky", "la", "lv", "lij", "dsb", "lg",
                            "lb", "ms", "ml", "mr", "ne", "nn", "fa", "sa", "sr", "tn", "si", "sk", "tl",
                            "ta", "tt", "te", "th", "ti", "tr", "hsb", "ur", "vi", "yo"
                            ]:
                return UDPreprocessor(language, study)
            else:
                raise ValueError(f"No preprocessor available for language: {language}")
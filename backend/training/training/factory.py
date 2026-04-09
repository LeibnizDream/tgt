"""
Factory for selecting the correct model trainer.

:class:`TrainerFactory` maps an (action, language) pair to the appropriate
:class:`~training.training.abstract.AbstractTrainer` subclass.  Currently
only the ``"gloss"`` action is supported, which returns a
:class:`~training.training.SpacyTrainer.SpacyTrainer` for all supported
language codes.
"""
from training.training.SpacyTrainer import SpacyTrainer


class TrainerFactory:
    """Factory that returns an :class:`~training.training.abstract.AbstractTrainer` for a given action and language."""
    @staticmethod
    def get_trainer(action, language: str, study: str):
        """
        Return the correct trainer for the given action and language.

        Args:
            action (str): Task action, e.g. ``"gloss"``.
            language (str): ISO 639-1 language code, e.g. ``"de"``.
            study (str): Study identifier appended to saved model names.

        Returns:
            AbstractTrainer: A configured trainer instance.

        Raises:
            ValueError: When no trainer exists for *action* / *language*.
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
                return SpacyTrainer(language, study)
            else:
                raise ValueError(f"No trainer available for language: {language}")
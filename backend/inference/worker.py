import argparse
from inference.abstract_worker import AbstractInferenceWorker

class LocalWorker(AbstractInferenceWorker):
    """
    Local implementation of AbstractInferenceWorker for CLI or API usage.
    Defines simple folder iteration and messaging behavior.
    """

    def _initial_message(self) -> None:
        """
        Print or queue the initial job start message.
        """
        self._put(f"Starting job {self.job_id} – action: {self.action}")

    def _after_process(self) -> None:
        """
        Print or queue a message after processing a folder.
        """
        self._put(f"Processed folder {self.current_folder} for job {self.job_id}")


def main() -> None:
    """
    Command-line interface entry point for running inference workers.

    Positional arguments:
        action      : transcribe | translate | transliterate | gloss
        language    : language name/code
        base_dir    : directory to process

    Optional arguments:
        --format
        --instruction
        --translation-model
        --glossing-model
    """

    parser = argparse.ArgumentParser(
        description="Run inference worker from the command line."
    )

    # Positional arguments
    parser.add_argument(
        "action",
        choices=["transcribe", "translate", "transliterate", "gloss"],
        help="Action to perform"
    )

    parser.add_argument(
        "language",
        help="Language name or code"
    )

    parser.add_argument(
        "base_dir",
        help="Directory to process"
    )

    # Optional arguments
    parser.add_argument(
        "--format",
        default="plain",
        choices=["labvanced", "plain"],
        help="Input/output format"
    )

    parser.add_argument(
        "--instruction",
        default=None,
        choices=["automatic", "corrected", "sentences"],
        help="Required for labvanced format"
    )

    parser.add_argument(
        "--translation-model",
        default=None,
        help="Translation model name"
    )

    parser.add_argument(
        "--glossing-model",
        default=None,
        help="Glossing model name"
    )

    args = parser.parse_args()

    if args.format == "labvanced" and not args.instruction:
        parser.error("--instruction is required when --format is labvanced")

    worker = LocalWorker(
        base_dir=args.base_dir,
        action=args.action,
        language=args.language,
        format=args.format,
        instruction=args.instruction,
        translationModel=args.translation_model,
        glossingModel=args.glossing_model,
    )

    worker.run()

if __name__ == "__main__":
    main()
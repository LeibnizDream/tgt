import argparse

from training.abstract_worker import AbstractTrainingWorker


class TrainingWorker(AbstractTrainingWorker):
    """
    Concrete training worker for local CLI or API execution.

    Implements abstract methods for local usage context.
    """
    def __init__(self, base_dir, language, action, study, publisher=None):
        super().__init__(base_dir, language, action, study, publisher)

    def _initial_message(self):
        return f"Initializing local training worker for {self.language}"

    def _folder_to_process(self):
        """Return the base directory for processing."""
        return self.base_dir

    def _after_preprocess(self):
        self.inform(f"Local preprocessing completed for job {self.job_id} – action: {self.action}")

    def _after_train(self):
        self.inform(f"Local training completed for job {self.job_id} – action: {self.action}")


def main():
    """
    CLI entry point for running the training preprocessing.

    Parses command-line arguments and executes the TrainingWorker.
    """
    parser = argparse.ArgumentParser(
        description="Run the UD training preprocessing for a given study."
    )
    parser.add_argument(
        "base_dir",
        help="Path to the directory containing annotated files"
    )
    parser.add_argument(
        "language",
        help="Language code for preprocessing (e.g., 'en', 'de')"
    )
    parser.add_argument(
        "action",
        help="Action to perform (e.g., 'preprocess', 'train', 'gloss')"
    )
    parser.add_argument(
        "study",
        help="Study identifier or name"
    )

    args = parser.parse_args()

    worker = TrainingWorker(
        base_dir=args.base_dir,
        language=args.language,
        action=args.action,
        study=args.study
    )
    worker.run()


if __name__ == "__main__":
    main()
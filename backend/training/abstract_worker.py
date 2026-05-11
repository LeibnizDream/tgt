"""
Abstract and concrete training worker classes for the TGT backend.

A *training worker* orchestrates two sequential phases:

1. **Preprocessing** – delegates to a :class:`~training.preprocessing.abstract.BasePreprocessor`
   subclass (chosen by :class:`~training.preprocessing.factory.PreProcessorFactory`)
   to convert annotated Excel files into a clean pandas DataFrame.
2. **Training**      – delegates to an :class:`~training.training.abstract.AbstractTrainer`
   subclass (chosen by :class:`~training.training.factory.TrainerFactory`) to
   train a morphological model and persist it under ``models/glossing/``.

Usage:
    - In production the router instantiates an :class:`OneDriveWorker` and
      calls :meth:`AbstractTrainingWorker.run` in a separate process.
    - From the CLI the ``main`` function instantiates a :class:`TrainingWorker`
      directly.
"""
import traceback
from abc import ABC, abstractmethod

import pandas as pd
from training.preprocessing.factory import PreProcessorFactory
from training.training.factory import TrainerFactory
from utils.functions import find_language, set_global_variables

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()


class AbstractTrainingWorker(ABC):
    """
    Abstract base class for training workers that handle preprocessing and training tasks.

    Subclasses must implement methods to produce initial messages, determine the folder to process,  
    and define actions after preprocessing and after training.
    """
    def __init__(self, base_dir, language, action, study, job=None):
        """
        Initialize the worker with job configuration.

        Args:
            base_dir (str): Root directory containing study data.
            language (str): Language code (e.g., 'en', 'de', 'yoruba').
            action (str): Task action (e.g., 'preprocess', 'train', 'gloss').
            study (str): Study identifier or name.
            job (optional): Optional job object when running in a queued environment.
        """
        self.base_dir = base_dir
        self.current_folder = base_dir
        self.language = find_language(language, LANGUAGES)
        self.action = action
        self.study = study
        self.job = job
        self.preprocessor = PreProcessorFactory.get_preprocessor(self.action, self.language, self.study)
        self.trainer = TrainerFactory.get_trainer(self.action, self.language, self.study)

        # Determine job-related identifiers and messaging queue
        self.job_id = job.id if job else 'local_job'
        self.q = job.queue if job else None
        self.cancel = job.cancel_event if job else None
    
    def _put(self, msg):
        """
        Send a message to the job queue or print to stdout if no queue is provided.

        Args:
            msg (str): Message to send or print.
        """
        if self.q:
            self.q.put(msg)
        else:
            print(msg)

    @abstractmethod
    def _initial_message(self):
        """Return the initial start message for the worker."""
        pass

    @abstractmethod
    def _folder_to_preprocess(self):
        """Determine and return the folder path to be processed."""
        yield self.current_folder

    def _preprocess(self):
        """
        Execute the preprocessing workflow using UDPreprocessor.

        This method sends status messages before and after processing.
        """
        for folder in self._folder_to_preprocess():
            self._put(f"Starting preprocessing for job {self.job_id} – action: {self.action}")
            data_df = self.preprocessor.preprocess(folder)
            self._put(f"Preprocessing completed for job {self.job_id}")
            return data_df

    def _train(self, data_df: pd.DataFrame):
        self._put(f"Starting training for job {self.job_id} – action: {self.action}")
        self.trainer.train(data_df)
        self._put(f"Training completed for job {self.job_id}")

    @abstractmethod
    def _after_preprocess(self):
        """Actions to perform immediately after preprocessing step completes.
        use self.current_folder to access the folder being processed."""
        pass

    @abstractmethod
    def _after_train(self):
        """Actions to perform immediately after training step completes."""
        pass

    def run(self):
        """
        Entry point to run preprocessing and training in sequence.

        Captures exceptions and ensures a final completion message.
        """
        self._put(f"Starting job {self.job_id} – action: {self.action}")
        try:
            data_df = self._preprocess()
            if data_df.empty:
                self._put(f"[WARNING] dataframe is empty after preprocessing for job {self.job_id} – action: {self.action}")
                return
            self._after_preprocess()
            self._train(data_df)
            self._after_train()
        except Exception as e:
            self._put(f"[ERROR] {str(e)}")
            traceback.print_exc()
        finally:
            self._put("[DONE ALL]")
 
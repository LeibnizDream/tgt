"""
Abstract training worker for the TGT backend.

Orchestrates two sequential phases:
1. Preprocessing — delegates to a BasePreprocessor subclass.
2. Training      — delegates to an AbstractTrainer subclass.
"""
import traceback
from abc import ABC, abstractmethod

import pandas as pd
from routers.helpers.job_manager import JobPublisher
from training.preprocessing.factory import PreProcessorFactory
from training.training.factory import TrainerFactory
from utils.functions import find_language, set_global_variables

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()


class AbstractTrainingWorker(ABC):
    def __init__(self, base_dir, language, action, study, publisher: JobPublisher | None = None):
        self.base_dir = base_dir
        self.current_folder = base_dir
        self.language = find_language(language, LANGUAGES)
        self.action = action
        self.study = study
        self.publisher = publisher if publisher is not None else JobPublisher()
        self.job_id = self.publisher.job_id
        self.cancel = self.publisher.cancel
        self.preprocessor = PreProcessorFactory.get_preprocessor(self.action, self.language, self.study)
        self.trainer = TrainerFactory.get_trainer(self.action, self.language, self.study)

    @property
    def _is_cancelled(self) -> bool:
        return bool(self.cancel and self.cancel.is_set())

    def inform(self, msg: str, level: str = "info") -> None:
        self.publisher.inform(msg, level)

    @abstractmethod
    def _initial_message(self):
        pass

    @abstractmethod
    def _folder_to_preprocess(self):
        yield self.current_folder

    def _preprocess(self):
        for folder in self._folder_to_preprocess():
            self.publisher.inform(f"Starting preprocessing for job {self.job_id} – action: {self.action}")
            data_df = self.preprocessor.preprocess(folder)
            self.publisher.inform(f"Preprocessing completed for job {self.job_id}")
            return data_df

    def _train(self, data_df: pd.DataFrame):
        self.publisher.inform(f"Starting training for job {self.job_id} – action: {self.action}")
        self.trainer.train(data_df)
        self.publisher.inform(f"Training completed for job {self.job_id}")

    @abstractmethod
    def _after_preprocess(self):
        pass

    @abstractmethod
    def _after_train(self):
        pass

    def run(self):
        self.publisher.inform(f"Starting job {self.job_id} – action: {self.action}")
        try:
            data_df = self._preprocess()
            if data_df.empty:
                self.publisher.inform(
                    f"Dataframe is empty after preprocessing for job {self.job_id} – action: {self.action}",
                    level="warning",
                )
                return
            self._after_preprocess()
            self._train(data_df)
            self._after_train()
        except Exception as e:
            self.publisher.inform(str(e), level="error")
            traceback.print_exc()
        finally:
            if self._is_cancelled:
                self.publisher.cancelled()
            else:
                self.publisher.done()

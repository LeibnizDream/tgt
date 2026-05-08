import os
import re
import warnings
import pandas as pd
from tqdm import tqdm
from inference.strategies.transcription.factory import TranscriptionStrategyFactory
from utils.functions import (
    set_global_variables,
    clean_german_transcription,
    find_ffmpeg,
    format_excel_output,
)
from inference.processors.labvanced.base import LabvancedBaseProcessor

# Global setup
LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()
warnings.filterwarnings("ignore")
ffmpeg_path = find_ffmpeg()


class TranscriptionProcessor(LabvancedBaseProcessor):
    """
    Processes directories of audio files, transcribing them into a trials-and-sessions sheet.
    """

    def __init__(self, language: str, instruction: str, device: str | None = None):
        super().__init__(language, instruction, device)
        self.strategy = TranscriptionStrategyFactory.get_strategy(self.language)
        print('initialized transcription strategy:', self.strategy.__class__.__name__)
        self.filename_regexp = re.compile(
            r'blockNr_(?P<block>\d+)_taskNr_(?P<task>\d+)_trialNr_(?P<trial>\d+).*'
        )

    def _find_files(self, base_dir: str) -> list[str]:
        # find parent dirs containing a 'binaries' subfolder
        bases = set()
        for subdir, _, files in os.walk(base_dir):
            if 'binaries' in os.path.basename(subdir):
                bases.add(os.path.abspath(os.path.join(subdir, '..')))
        return sorted(bases)

    def _read_file(self, base_dir: str) -> pd.DataFrame:
        # load trials-and-sessions sheet
        df, out_file = self.load_trials_data(base_dir)
        self._current_out_file = out_file
        self._current_base_dir = base_dir
        return df

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # iterate over audio files in 'binaries' and append transcriptions
        bin_dir = os.path.join(self._current_base_dir, 'binaries')
        audio_files = [
            f for f in sorted(os.listdir(bin_dir))
            if f.lower().endswith(('.mp3', '.mp4', '.m4a'))
        ]
        total = len(audio_files)
        progress_cb = getattr(self, '_progress_callback', None)
        count = 0
        for file in tqdm(audio_files, desc="Transcribing audio"):
            count += 1
            path = os.path.join(bin_dir, file)
            try:
                text = self.strategy.transcribe(path)
                if self.language == 'de':
                    text = clean_german_transcription(text)
                self.add_transcription_to_df(
                    df,
                    file,
                    text,
                    count,
                    self.filename_regexp,
                )
            except Exception as e:
                self.logger.info(f"Error processing file '{file}': {e}")
            if progress_cb:
                progress_cb(count, total)
        return df

    def _write_file(self, _: str, df: pd.DataFrame):
        # write out annotated sheet and apply formatting
        df.to_excel(self._current_out_file, index=False)
        highlight_col = (
            'transcription_original_script'
            if self.language in NO_LATIN
            else 'latin_transcription_everything'
        )
        format_excel_output(self._current_out_file, highlight_col)

    def load_trials_data(self, base_dir: str):
        csv_file = os.path.join(base_dir, 'trials_and_sessions.csv')
        excel_file = os.path.join(base_dir, 'trials_and_sessions_annotated.xlsx')
        excel_out = os.path.join(base_dir, 'trials_and_sessions_annotated.xlsx')

        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
        elif os.path.exists(csv_file):
            df = pd.read_csv(csv_file, encoding='utf-8')
        else:
            raise FileNotFoundError(
                "No trials_and_sessions file found in the directory."
            )

        for col in OBLIGATORY_COLUMNS:
            if col not in df.columns:
                df[col] = ""
            else:
                df[col] = df[col].fillna("").astype(str).replace("nan", "")

        if self.language not in NO_LATIN:
            df["transcription_original_script"] = ""
            df["transcription_original_script_utterance_used"] = ""

        return df, excel_out

    def _append_to_cell(self, df, idx, column, text):
        old = df.at[idx, column]
        df.at[idx, column] = ("" if pd.isna(old) else old) + text

    def add_transcription_to_df(
        self, df, file, transcription, count, filename_regexp
    ):
        series = df[df.isin([file])].stack()
        series = series[series == file]
        print(f"[DEBUG] file='{file}' | series length={len(series)} | empty={series.empty}")
        if not series.empty:
            for (row_idx, col_name_found), val in series.items():
                print(f"  [DEBUG] row={row_idx}, col='{col_name_found}', value='{val}'")
            unique_rows = series.index.get_level_values(0).unique()
            print(f"  [DEBUG] unique rows matched: {list(unique_rows)} (count={len(unique_rows)})")
            if len(unique_rows) > 1:
                raise ValueError(f"File '{file}' matched multiple unique rows in the DataFrame, which should not happen.")
        text_auto = f"{count}: {transcription}"
        suffix = " - " if series.empty else " "
        col_name = (
            'transcription_original_script'
            if self.language in NO_LATIN
            else 'latin_transcription_everything'
        )

        if series.empty:
            match = filename_regexp.search(file)
            if not match:
                self.logger.info(
                    f"File '{file}' does not match block/task/trial pattern. Skipping."
                )
                return
            blk = int(match['block'])
            tsk = int(match['task'])
            trl = int(match['trial'])
            cond = (
                (df['Block_Nr'] == blk)
                & (df['Task_Nr'] == tsk)
                & (df['Trial_Nr'] == trl)
            )
            if df.loc[cond].empty:
                self.logger.info(
                    f"No row for block {blk}, task {tsk}, trial {trl}. Skipping '{file}'."
                )
                return
            miss_col = next(
                f"missing_filename_{i}"
                for i in range(1, 10)
                if f"missing_filename_{i}" not in df.columns
                or df.loc[cond, f"missing_filename_{i}"].isna().all()
            )
            df.loc[cond, miss_col] = file
            for idx in df.loc[cond].index:
                self._append_to_cell(df, idx, 'automatic_transcription', text_auto + suffix)
                self._append_to_cell(df, idx, col_name,      text_auto + suffix)
        else:
            for (row_idx, _), _ in series.items():
                self._append_to_cell(df, row_idx, 'automatic_transcription', text_auto + suffix)
                self._append_to_cell(df, row_idx, col_name,      text_auto + suffix)
    

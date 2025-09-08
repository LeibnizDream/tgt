import os
import pandas as pd
from inference.processors.abstract import DataProcessor
from utils.reorder_columns import create_columns
from utils.functions import (
    set_global_variables,
    format_excel_output
)
from pathlib import Path

LANGUAGES, NO_LATIN, OBLIGATORY_COLUMNS = set_global_variables()

class ColumnCreationProcessor(DataProcessor):
    def _find_files(self, base_dir: str) -> list[str]:
        matches = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith("trials_and_sessions.csv"):
                    matches.append(os.path.join(root, f))
        self.logger.info(f"Found {len(matches)} matching files in {base_dir}")
        return sorted(matches)
    
    def _read_file(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)
    
    def _write_file(self, path: str, df: pd.DataFrame):
        # reorder columns as in base class
        extra_cols = [c for c in df.columns if c not in OBLIGATORY_COLUMNS]
        df = df[extra_cols + [c for c in OBLIGATORY_COLUMNS if c in df.columns]]

        in_path = Path(path)
        # write alongside the CSV as an .xlsx
        out_path = in_path.with_name(in_path.stem + "_annotated.xlsx")

        # make sure you have an Excel engine installed (openpyxl or xlsxwriter)
        df.to_excel(out_path, index=False)
        self.logger.info(f"Wrote output to {out_path}")

        if self.columns_to_highlight:
            format_excel_output(str(out_path), self.columns_to_highlight)

    def _process_dataframe(self, df: pd.DataFrame):
        new_df = create_columns(df, self.language)
        return new_df
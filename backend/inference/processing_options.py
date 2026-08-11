# Copyright (C) 2026 Leibniz-Zentrum Allgemeine Sprachwissenschaft
# Developed as part of the ERC-funded LeibnizDream project.
# Licensed under the GNU General Public License v3.0 or later.
# See LICENSE for details.


from pydantic import BaseModel


class ProcessingOptions(BaseModel):
    language: str
    action: str
    format: str | None = None
    instruction: str | None = None
    model: str | None = None
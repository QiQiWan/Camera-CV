# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping


class DailyLogger:
    def __init__(self, log_dir: str = 'Data', file_extension: str = 'csv'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file_extension = file_extension.lstrip('.') or 'csv'
        self.headers: list[str] = []

    def set_headers(self, headers: Iterable[str]):
        self.headers = list(headers)

    def _current_path(self) -> Path:
        today = datetime.now().strftime('%Y%m%d')
        return self.log_dir / f'log_{today}.{self.file_extension}'

    def log(self, row: Mapping[str, object]):
        path = self._current_path()
        file_exists = path.exists()
        with path.open('a', newline='', encoding='utf-8-sig') as f:
            fieldnames = self.headers or list(row.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, '') for key in fieldnames})

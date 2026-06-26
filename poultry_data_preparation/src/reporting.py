from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import markdown_codeblock


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dataframe_summary_block(dataframe: pd.DataFrame, title: str) -> str:
    if dataframe.empty:
        return f"## {title}\n\nNo rows available.\n"
    return f"## {title}\n\n{markdown_codeblock(dataframe.to_string())}\n"

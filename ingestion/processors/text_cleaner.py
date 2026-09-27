import re
from typing import List


class TextCleaner:
    """
    Utility class for cleaning and normalizing raw text extracted from Telecom PDFs.
    """

    @staticmethod
    def clean_raw_text(text: str) -> str:
        """
        Cleans general text extracted from PDF pages.
        Removes orphan headers, footers, page numbers, and excess whitespace.
        """
        if not text:
            return ""

        # Strip repetitive page numbers (e.g., "Page 12 of 45", "Page 12")
        text = re.sub(r"(?i)page\s+\d+(\s+of\s+\d+)?", "", text)

        # Strip internal corporate footer noise
        text = re.sub(r"(?i)strictly confidential\s*-\s*internal use only", "", text)

        # Replace multiple horizontal spaces/tabs with a single space
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize multiple vertical line breaks (keep max 2 newlines for paragraph breaks)
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()

    @staticmethod
    def format_table_as_markdown(table_data: List[List[str]]) -> str:
        """
        Converts a 2D list (extracted table from pdfplumber) into a clean Markdown table string.
        """
        if not table_data or len(table_data) < 2:
            return ""

        cleaned_table = []
        for row in table_data:
            cleaned_row = [
                (cell.replace("\n", " ").strip() if cell else "")
                for cell in row
            ]
            if any(cleaned_row):
                cleaned_table.append(cleaned_row)

        if not cleaned_table:
            return ""

        headers = cleaned_table[0]
        rows = cleaned_table[1:]

        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

        row_lines = []
        for row in rows:
            while len(row) < len(headers):
                row.append("")
            row_lines.append("| " + " | ".join(row[:len(headers)]) + " |")

        return "\n".join([header_line, separator_line] + row_lines)
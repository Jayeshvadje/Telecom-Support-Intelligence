from pathlib import Path
from typing import Dict, Any, List
from docx import Document

from ingestion.processors.text_cleaner import TextCleaner


class DOCXDocumentLoader:
    """
    Loads and parses enterprise Word (.docx) policy documents.
    Extracts structured headings, paragraphs, bullet lists, and tables as Markdown.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"DOCX file not found at: {file_path}")

    def load_and_parse(self) -> Dict[str, Any]:
        """
        Parses the DOCX file and extracts content sequentially while preserving structure.
        """
        doc = Document(str(self.file_path))
        extracted_elements = []

        # 1. Iterate through body elements (paragraphs and tables)
        for element in doc.element.body:
            # Handle Paragraphs (including Headings and Bullet Lists)
            if element.tag.endswith('p'):
                para = [p for p in doc.paragraphs if p._element == element]
                if para and para[0].text.strip():
                    p_obj = para[0]
                    text = TextCleaner.clean_raw_text(p_obj.text)
                    style_name = p_obj.style.name if p_obj.style else ""

                    extracted_elements.append({
                        "type": "heading" if "Heading" in style_name else "paragraph",
                        "style": style_name,
                        "text": text
                    })

            # Handle Tables
            elif element.tag.endswith('tbl'):
                table_obj = [t for t in doc.tables if t._element == element]
                if table_obj:
                    raw_table_data = []
                    for row in table_obj[0].rows:
                        row_data = [cell.text.strip() for cell in row.cells]
                        raw_table_data.append(row_data)

                    markdown_table = TextCleaner.format_table_as_markdown(raw_table_data)
                    if markdown_table:
                        extracted_elements.append({
                            "type": "table",
                            "style": "Table",
                            "text": markdown_table
                        })

        # Combine extracted elements into a coherent document string
        full_content = "\n\n".join([item["text"] for item in extracted_elements])

        return {
            "file_name": self.file_path.name,
            "content": full_content,
            "elements": extracted_elements,
            "metadata": {
                "file_name": self.file_path.name,
                "file_path": str(self.file_path),
            }
        }
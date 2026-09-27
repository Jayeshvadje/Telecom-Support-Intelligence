from pathlib import Path
from typing import List, Dict, Any

from ingestion.loaders.docx_loader import DOCXDocumentLoader


class IngestionPipeline:
    """
    Orchestrates the ingestion process for raw DOCX policy files.
    """

    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)

    def run(self) -> List[Dict[str, Any]]:
        if not self.input_dir.exists():
            print(f"Directory {self.input_dir} does not exist. Creating it...")
            self.input_dir.mkdir(parents=True, exist_ok=True)
            return []

        docx_files = list(self.input_dir.glob("*.docx"))
        print(f"Found {len(docx_files)} DOCX policy documents in '{self.input_dir}'")

        all_parsed_documents = []

        for docx_path in docx_files:
            # Skip temporary Office lock files (files starting with ~$)
            if docx_path.name.startswith("~$"):
                continue

            print(f"Parsing: {docx_path.name}...")
            try:
                loader = DOCXDocumentLoader(str(docx_path))
                parsed_doc = loader.load_and_parse()
                all_parsed_documents.append(parsed_doc)
                print(f" Successfully parsed {docx_path.name} ({len(parsed_doc['elements'])} structural elements)")
            except Exception as e:
                print(f"❌ Failed to parse {docx_path.name}: {str(e)}")

        return all_parsed_documents


if __name__ == "__main__":
    input_directory = "ingestion/data"
    pipeline = IngestionPipeline(input_dir=input_directory)
    results = pipeline.run()
    print(f"\nIngestion Pipeline Complete. Total files processed: {len(results)}")
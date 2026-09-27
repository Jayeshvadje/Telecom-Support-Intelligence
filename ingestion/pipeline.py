from pathlib import Path
from typing import List, Dict, Any

from ingestion.loaders.docx_loader import DOCXDocumentLoader
from ingestion.chunking.policy_chunker import PolicyChunker
from ingestion.metadata.extractor import MetadataExtractor


class IngestionPipeline:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.chunker = PolicyChunker(target_tokens=600, overlap_tokens=100)

    def run(self) -> List[Dict[str, Any]]:
        if not self.input_dir.exists():
            print(f"Directory {self.input_dir} does not exist.")
            return []

        docx_files = [p for p in self.input_dir.glob("*.docx") if not p.name.startswith("~$")]
        print(f"Found {len(docx_files)} DOCX policy documents in '{self.input_dir}'\n")

        all_processed_chunks = []

        for docx_path in docx_files:
            print(f"📄 Processing: {docx_path.name}")
            try:
                # 1. Load DOCX
                loader = DOCXDocumentLoader(str(docx_path))
                parsed_doc = loader.load_and_parse()

                # 2. Chunking
                chunks = self.chunker.create_chunks(parsed_doc)

                # 3. Metadata Enrichment
                for chunk in chunks:
                    enriched_chunk = MetadataExtractor.enrich_chunk(chunk)
                    all_processed_chunks.append(enriched_chunk)

                print(f"   └── Extracted {len(chunks)} policy chunks")

            except Exception as e:
                print(f"❌ Failed processing {docx_path.name}: {str(e)}")

        print(f"\n Pipeline Complete. Total chunks generated across all files: {len(all_processed_chunks)}")
        return all_processed_chunks


if __name__ == "__main__":
    pipeline = IngestionPipeline(input_dir="ingestion/data")
    chunks = pipeline.run()
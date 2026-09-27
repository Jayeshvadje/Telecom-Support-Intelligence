from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from ingestion.loaders.docx_loader import DOCXDocumentLoader
from ingestion.chunking.policy_chunker import PolicyChunker
from ingestion.metadata.extractor import MetadataExtractor
from ingestion.embeddings.gemini_embedder import GeminiEmbedder
from ingestion.vectorstore.pinecone_manager import PineconeManager

load_dotenv()


class IngestionPipeline:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.chunker = PolicyChunker(target_tokens=600, overlap_tokens=100)
        self.embedder = GeminiEmbedder()
        self.vector_store = PineconeManager()

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
                # 1. Parse DOCX
                loader = DOCXDocumentLoader(str(docx_path))
                parsed_doc = loader.load_and_parse()

                # 2. Chunking
                chunks = self.chunker.create_chunks(parsed_doc)

                # 3. Metadata Enrichment
                enriched_chunks = [MetadataExtractor.enrich_chunk(c) for c in chunks]

                # 4. Generate Embeddings via Gemini API
                embedded_chunks = self.embedder.embed_chunks(enriched_chunks)

                all_processed_chunks.extend(embedded_chunks)
                print(f"   └── Extracted and embedded {len(embedded_chunks)} chunks")

            except Exception as e:
                print(f"❌ Failed processing {docx_path.name}: {str(e)}")

        # 5. Upsert to Pinecone
        if all_processed_chunks:
            print("\nUpserting vectors into Pinecone...")
            self.vector_store.upsert_chunks(all_processed_chunks)

        print(f"\n Pipeline Complete. Total chunks indexed: {len(all_processed_chunks)}")
        return all_processed_chunks


if __name__ == "__main__":
    pipeline = IngestionPipeline(input_dir="ingestion/data")
    pipeline.run()
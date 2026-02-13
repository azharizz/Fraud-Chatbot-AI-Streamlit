import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


def main() -> None:
    from src.data.database import FraudDatabase

    logger.info("=== FRAUD Q&A CHATBOT - DATA INGESTION ===")

    # Step 1: CSV -> DuckDB
    logger.info("[1/2] Loading CSV files into DuckDB...")
    db = FraudDatabase.connect(read_only=False)
    row_count = db.ingest_csv()
    logger.info("[1/2] Complete: %s transactions loaded", f"{row_count:,}")

    # Step 2: PDF -> FAISS
    logger.info("[2/2] Processing PDFs into FAISS...")
    try:
        from src.data.vectorstore import VectorStore
        vs = VectorStore.from_pdfs()
        logger.info("[2/2] Complete: %d chunks indexed", len(vs.chunks))
    except ImportError:
        logger.warning("[2/2] vectorstore not yet implemented, skipping")

    logger.info("=== INGESTION COMPLETE ===")


if __name__ == "__main__":
    main()

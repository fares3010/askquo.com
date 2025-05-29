import os
from pinecone import Pinecone, ServerlessSpec
import getpass
import openai
import langchain
from langchain_openai import OpenAIEmbeddings
import pinecone
from langchain.chat_models import init_chat_model
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from typing import Optional, List, Tuple, Any
import logging
import dotenv
from pathlib import Path

# Load environment variables
dotenv.load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str, api_key: str, index_name: str):
        """Initialize the EmbeddingModel with required parameters."""
        self.model_name = model_name
        self.api_key = api_key
        self.index_name = index_name
        self._validate_initialization()

    def _validate_initialization(self) -> None:
        """Validate initialization parameters."""
        if not all([self.model_name, self.index_name]):
            raise ValueError("model_name and index_name are required")
        if not self.api_key and not os.environ.get("OPENAI_API_KEY"):
            logger.warning("No API key provided during initialization")

    def get_api_key(self) -> str:
        """Safely retrieve OpenAI API key from environment or user input."""
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                self.api_key = getpass.getpass("Enter API key for OpenAI: ")
        os.environ["OPENAI_API_KEY"] = self.api_key
        return self.api_key

    def initialize_models(self) -> Tuple[Any, OpenAIEmbeddings]:
        """Initialize and return the chat model and embeddings model."""
        try:
            model = init_chat_model(self.model_name, model_provider="openai")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            logger.info(f"Successfully initialized models: {self.model_name}")
            return model, embeddings
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise

    def connect_to_pinecone(self, embeddings: OpenAIEmbeddings) -> Pinecone:
        """Establish connection to Pinecone database with proper error handling."""
        try:
            pinecone_api_key = os.environ.get("PINECONE_API_KEY")
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")

            pc = Pinecone(api_key=pinecone_api_key)
            
            if not pc.has_index(self.index_name):
                logger.info(f"Creating new index: {self.index_name}")
                pc.create_index_for_model(
                    name=self.index_name,
                    cloud="aws",
                    region="us-east-1",
                    embed={
                        "model": "text-embedding-3-large",
                        "field_map": {"text": "chunk_text"}
                    }
                )
            else:
                logger.info(f"Using existing index: {self.index_name}")
                index = pc.Index(self.index_name)
                if embeddings:
                    index.upsert(vectors=embeddings)
            
            return pc.from_existing_index(self.index_name, embeddings)
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {str(e)}")
            raise

def process_documents(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    """Load and process documents with configurable chunking parameters."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        loader = TextLoader(str(path))
        docs = loader.load()
            
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
            length_function=len,
            is_separator_regex=False
        )
            
        split_docs = splitter.split_documents(docs)
        logger.info(f"Successfully split {len(docs)} documents into {len(split_docs)} chunks")
        return split_docs
    except FileNotFoundError as e:
        logger.error(f"Document file not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise

def main() -> Tuple[Any, OpenAIEmbeddings, Pinecone, List[Any]]:
    """Main execution function with proper error handling and logging."""
    try:
        embedding_model = EmbeddingModel(
            model_name="gpt-3.5-turbo",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            index_name="test-index"
        )
        
        # Set API key
        embedding_model.get_api_key()
        
        # Initialize models
        model, embeddings = embedding_model.initialize_models()
        
        # Connect to Pinecone
        db = embedding_model.connect_to_pinecone(embeddings)
        
        # Test query
        query = "Tell me about modular furniture"
        response = model.invoke(query)
        logger.info(f"Query response: {response}")
        
        # Process documents
        split_docs = process_documents("sereno_docs.txt")
        
        return model, embeddings, db, split_docs
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()

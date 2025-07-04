from create_agent.vdb.pinecone_vdb import PineconeVDB
from create_agent.embedding.openai import OpenAIEmbeddingModel, EmbeddingConfig
from create_agent.models import Agent, AgentEmbeddings, AgentVectorsDatabase
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import openai
from typing import Optional, Dict, Any
import time

load_dotenv()

import logging

logger = logging.getLogger(__name__)

def initialize_chat_agent(
    user_id: int, 
    agent_id: int, 
    query: str, 
    chat_history: str = "",
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> str:
    """Initialize chat agent components with proper error handling and logging.
    
    Args:
        user_id: ID of the user
        agent_id: ID of the agent
        query: User's question/query
        chat_history: Previous conversation history
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Delay between retries in seconds
        
    Returns:
        str: AI agent's response to the query
        
    Raises:
        ValueError: For invalid inputs or configuration errors
        Exception: For unexpected errors during initialization or API calls
    """
    def retry_with_backoff(func, *args, **kwargs):
        """Helper function to retry operations with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay * (2 ** attempt))

    try:
        # Get user and agent with specific error messages
        agent = retry_with_backoff(get_object_or_404, Agent, agent_id=agent_id)

        # Get embeddings and vector DB configurations with validation
        vector_db_config = retry_with_backoff(get_object_or_404, AgentVectorsDatabase, agent=agent)
        
        if not vector_db_config.namespace:
            raise ValueError("Incomplete vector DB configuration")

        # Initialize embedding model with optimized config
        config = EmbeddingConfig(
            chunk_size=700,  # Optimized for most use cases
            chunk_overlap=100,  # 15% overlap for better context
            batch_size=100,  # Balanced between speed and memory
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        embedding_model = OpenAIEmbeddingModel(config=config)

        # Initialize vector DB
        vector_db = PineconeVDB(
            index_name=vector_db_config.database_index,
            namespace=vector_db_config.namespace
        )

        # Get query embedding and retrieve relevant context
        query_embedding = retry_with_backoff(
            embedding_model.embeddings.embed_query,
            query
        )
        
        matches = retry_with_backoff(
            vector_db.query_vectors,
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        ).matches

        # Build context from matches
        context = "\n".join(
            match["metadata"]["text"] 
            for match in matches 
            if match.get("metadata", {}).get("text")
        )

        # Prepare and send prompt to OpenAI
        prompt = PromptTemplate(
            template="""You are a helpful Customer Support Agent. Use the following context to answer the question.
            If you don't know the answer, just say that you don't know.
            
            Context: {context}
            Chat History: {chat_history}
            Question: {question}
            
            Please provide a clear, concise, and helpful response.""",
            input_variables=["context", "chat_history", "question"]
        )
        
        prompt_value = prompt.format(
            context=context,
            chat_history=chat_history,
            question=query
        )
        
        response = retry_with_backoff(
            embedding_model.chat_model.invoke,
            input=prompt_value
        )
        
        return response.content

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat agent initialization: {str(e)}")
        raise

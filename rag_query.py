# rag_query.py

import psycopg2
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch

class RIBRAGSystem:
    def __init__(self, db_config: Dict[str, str], model_name: str = None):
        """
        Initialize RAG system with HuggingFace model
        
        Args:
            db_config: Database configuration
            model_name: HuggingFace model name (must match the one used for embedding)
        """
        self.db_config = db_config
        
        # Use the same model as used for embedding!
        if model_name is None:
            model_name = "BAAI/bge-large-en-v1.5"
        
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        
        print(f"✓ Model loaded on {device}")
        
    def connect_db(self):
        """Establish database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            port=self.db_config.get('port', 5432)
        )
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query"""
        # For BGE models, use query instruction
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    def search_similar_chunks(
        self, 
        query: str, 
        top_k: int = 5, 
        similarity_threshold: float = 0.7,
        category_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query: User query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            category_filter: Optional category to filter by
            
        Returns:
            List of similar chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            # Build query with optional category filter
            sql = """
                SELECT 
                    c.id,
                    c.category,
                    c.title,
                    c.content,
                    1 - (c.embedding <=> %s::vector) AS similarity,
                    m.section_number,
                    m.regulations,
                    m.keywords,
                    m.risk_type
                FROM rib_chunks c
                LEFT JOIN chunk_metadata m ON c.id = m.chunk_id
                WHERE 1 - (c.embedding <=> %s::vector) > %s
            """
            params = [query_embedding, query_embedding, similarity_threshold]
            
            if category_filter:
                sql += " AND c.category = %s"
                params.append(category_filter)
            
            sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
            params.extend([query_embedding, top_k])
            
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'category': row[1],
                    'title': row[2],
                    'content': row[3],
                    'similarity': float(row[4]),
                    'section_number': row[5],
                    'regulations': row[6],
                    'keywords': row[7],
                    'risk_type': row[8]
                })
            
            return results
        
        finally:
            cursor.close()
            conn.close()
    
    def format_results(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks for display"""
        if not chunks:
            return "No relevant information found."
        
        formatted = "Retrieved relevant sections:\n\n"
        for i, chunk in enumerate(chunks, 1):
            formatted += f"{i}. Section {chunk['section_number']}: {chunk['title']}\n"
            formatted += f"   Category: {chunk['category']}\n"
            formatted += f"   Similarity: {chunk['similarity']:.3f}\n"
            formatted += f"   Risk Type: {chunk['risk_type']}\n"
            if chunk['regulations']:
                formatted += f"   Regulations: {', '.join(chunk['regulations'][:2])}\n"
            formatted += "\n"
        
        return formatted
    
    def query(
        self, 
        question: str, 
        top_k: int = 5,
        category_filter: str = None,
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Search for relevant information
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            category_filter: Optional category filter
            show_sources: Whether to include source chunks in response
            
        Returns:
            Dictionary with results and metadata
        """
        print(f"Searching for: {question}")
        
        # Search for similar chunks
        chunks = self.search_similar_chunks(
            query=question,
            top_k=top_k,
            category_filter=category_filter
        )
        
        if not chunks:
            return {
                'results': "I couldn't find relevant information in the RIB document.",
                'sources': []
            }
        
        print(f"✓ Found {len(chunks)} relevant sections")
        
        # Format results
        formatted_results = self.format_results(chunks)
        
        # Prepare response
        response = {
            'results': formatted_results,
            'chunks': chunks,
            'num_sources': len(chunks)
        }
        
        if show_sources:
            response['sources'] = [
                {
                    'section': chunk['section_number'],
                    'title': chunk['title'],
                    'category': chunk['category'],
                    'similarity': round(chunk['similarity'], 3),
                    'risk_type': chunk['risk_type'],
                    'content_preview': chunk['content'][:1000] + "..."
                }
                for chunk in chunks
            ]
        
        return response


def main():
    """Example usage"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'tsdsrd',
        'user': 'amir',
        'password': 'amir123',
        'port': 5432
    }
    
    # Use the same model as embedding!
    model_name = "BAAI/bge-large-en-v1.5"
    
    # Initialize RAG system
    rag = RIBRAGSystem(db_config, model_name=model_name)
    
    # Example queries
    queries = [
        "What are the recommendations for preventing subsidence?",
        "How should I handle electrical panel safety?",
        "What regulations apply to LPG gas cylinder storage?",
        "What should I do about lightning protection?",
        "How to manage dust extraction systems?",
    ]
    
    print("="*80)
    print("RIB RAG SYSTEM - DEMO")
    print("="*80)
    
    for query in queries:
        print(f"\n\nQuery: {query}")
        print("-"*80)
        
        result = rag.query(query, top_k=3, show_sources=True)
        
        print(result['results'])
        
        if result['sources']:
            print("\nDetailed Sources:")
            for source in result['sources']:
                print(f"\n{source['section']}: {source['title']}")
                print(f"Similarity: {source['similarity']}")
                print(f"Preview: {source['content_preview']}")


if __name__ == "__main__":
    main()
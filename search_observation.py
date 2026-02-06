# search_rib_observations.py

import psycopg2
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any

class RIBSemanticSearch:
    def __init__(self, db_config: Dict[str, str], model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize semantic search with database and model
        
        Args:
            db_config: Database connection config
            model_name: Same model used for creating embeddings
        """
        self.db_config = db_config
        self.model_name = model_name
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✓ Model loaded on {device}")
    
    def connect_db(self):
        """Connect to database"""
        return psycopg2.connect(
            host=self.db_config['host'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            port=self.db_config.get('port', 5432)
        )
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for search query
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding
        """
        # For BGE models, use query-specific instruction
        if "bge" in self.model_name.lower():
            query = f"Represent this query for searching relevant observations: {query}"
        
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    def search_observations(
        self, 
        query: str, 
        limit: int = 5,
        category_filter: str = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for observations semantically similar to the query
        
        Args:
            query: Search query
            limit: Number of results to return
            category_filter: Optional category to filter by (e.g., "Process", "Electrical")
            similarity_threshold: Optional minimum similarity score (0-1)
            
        Returns:
            List of matching chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Connect to database
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            # Build SQL query
            sql = """
                SELECT 
                    r.id,
                    r.section_number,
                    r.title,
                    r.category,
                    r.observation,
                    r.recommendation,
                    r.regulation,
                    m.observation_length,
                    m.recommendation_length,
                    1 - (r.observation_embedding <=> %s::vector) as similarity
                FROM tsd_rib r
                JOIN tsd_metadata m ON r.id = m.chunk_id
                WHERE m.has_observation = TRUE
            """
            
            params = [query_embedding]
            
            # Add category filter if specified
            if category_filter:
                sql += " AND r.category = %s"
                params.append(category_filter)
            
            # Add similarity threshold if specified
            if similarity_threshold:
                sql += " AND (1 - (r.observation_embedding <=> %s::vector)) >= %s"
                params.extend([query_embedding, similarity_threshold])
            
            sql += " ORDER BY r.observation_embedding <=> %s::vector LIMIT %s"
            params.extend([query_embedding, limit])
            
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'section_number': row[1],
                    'title': row[2],
                    'category': row[3],
                    'observation': row[4],
                    'recommendation': row[5],
                    'regulation': row[6],
                    'observation_length': row[7],
                    'recommendation_length': row[8],
                    'similarity': float(row[9])
                })
            
            return results
            
        finally:
            cursor.close()
            conn.close()
    
    def print_results(self, results: List[Dict[str, Any]], show_full_text: bool = False):
        """
        Pretty print search results
        
        Args:
            results: List of result dictionaries
            show_full_text: Whether to show full observation/recommendation text
        """
        if not results:
            print("No results found.")
            return
        
        print(f"\n{'='*100}")
        print(f"Found {len(results)} results")
        print(f"{'='*100}\n")
        
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Section: {result['section_number']}")
            print(f"  Category: {result['category']}")
            print(f"  Title: {result['title']}")
            print(f"  Similarity: {result['similarity']:.4f}")
            
            if show_full_text:
                print(f"\n  OBSERVATION:")
                print(f"  {result['observation'][:500]}{'...' if len(result['observation']) > 500 else ''}")
                
                if result['recommendation']:
                    print(f"\n  RECOMMENDATION:")
                    print(f"  {result['recommendation'][:500]}{'...' if len(result['recommendation']) > 500 else ''}")
            else:
                print(f"  Observation: {result['observation'][:200]}...")
            
            print(f"\n{'-'*100}\n")
    
    def get_categories(self) -> List[str]:
        """Get list of all categories in the database"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT DISTINCT category 
                FROM tsd_rib 
                ORDER BY category
            """)
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()


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
    
    # Initialize search
    search = RIBSemanticSearch(db_config, model_name="BAAI/bge-large-en-v1.5")
    
    # Show available categories
    print("Available categories:")
    categories = search.get_categories()
    for cat in categories:
        print(f"  - {cat}")
    
    print("\n" + "="*100)
    
    # Example searches
    example_queries = [
        "gas pipeline maintenance",
        "electrical hazards and safety",
        "fire prevention systems",
        "corrosion in pipelines"
    ]
    
    for query in example_queries:
        print(f"\n🔍 Searching for: '{query}'")
        print("="*100)
        
        results = search.search_observations(
            query=query,
            limit=3,
            # category_filter="Process",  # Optional: filter by category
            # similarity_threshold=0.5     # Optional: minimum similarity
        )
        
        search.print_results(results, show_full_text=False)
    
    # Interactive search
    print("\n" + "="*100)
    print("INTERACTIVE SEARCH MODE")
    print("="*100)
    print("Enter your search queries (or 'quit' to exit)")
    print("You can optionally specify category like: query | category")
    print("Example: 'electrical safety | Electrical'\n")
    
    while True:
        try:
            user_input = input("\nEnter query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Check if category filter is specified
            category = None
            if '|' in user_input:
                query, category = user_input.split('|', 1)
                query = query.strip()
                category = category.strip()
            else:
                query = user_input
            
            # Perform search
            results = search.search_observations(
                query=query,
                limit=5,
                category_filter=category
            )
            
            search.print_results(results, show_full_text=True)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
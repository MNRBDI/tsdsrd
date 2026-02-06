# interactive_rag_vllm_improved.py

from multimodal_rag_vllm_serve import MultimodalRAGSystemVLLM
import os
from pathlib import Path

def print_header():
    """Print application header"""
    print("\n" + "="*80)
    print(" "*20 + "RIB MULTIMODAL RAG ASSISTANT (VLLM)")
    print("="*80)
    print("\n📖 How to use:")
    print("  1. Set an image (optional): /image /path/to/image.jpg")
    print("  2. Ask your question: What safety issues do you see?")
    print("  3. The system will use the image + question together")
    print("\n⚙️  Commands:")
    print("  /image <path>     - Set an image for the next query")
    print("  /clear            - Clear the current image")
    print("  /category <name>  - Filter by category (Perils, Electrical, etc.)")
    print("  /reset            - Reset category filter")
    print("  /topk <number>    - Set number of chunks to retrieve (default: 5)")
    print("  /temp <number>    - Set temperature (0.0-1.0, default: 0.3)")
    print("  /threshold <num>  - Set similarity threshold (0.0-1.0, default: 0.3)")
    print("  /stats            - Show current settings")
    print("  /help             - Show this help message")
    print("  /quit or /exit    - Exit the application")
    print("="*80)
    print("\n💡 Tip: Set an image first, then ask questions about it!")
    print("="*80 + "\n")

def print_performance_box(result: dict):
    """Print a nice performance metrics box"""
    gen_time = result.get('generation_time', 0)
    tps = result.get('tokens_per_second', 0)
    usage = result.get('usage', {})
    
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    total_tokens = usage.get('total_tokens', 0)
    
    print("\n┌" + "─"*78 + "┐")
    print(f"│ {'⚡ PERFORMANCE METRICS':^78} │")
    print("├" + "─"*78 + "┤")
    print(f"│ ⏱️  Generation Time:     {gen_time:>8.2f}s{' '*43}│")
    print(f"│ ⚡ Tokens/Second:        {tps:>8.2f} tok/s{' '*39}│")
    print("├" + "─"*78 + "┤")
    print(f"│ 📊 Token Usage:{' '*62}│")
    print(f"│    • Prompt Tokens:      {prompt_tokens:>8,}{' '*43}│")
    print(f"│    • Completion Tokens:  {completion_tokens:>8,}{' '*43}│")
    print(f"│    • Total Tokens:       {total_tokens:>8,}{' '*43}│")
    print("└" + "─"*78 + "┘")

def print_current_settings(current_image, category_filter, top_k, temperature, similarity_threshold):
    """Print current settings in a nice format"""
    print("\n" + "─"*80)
    print("⚙️  CURRENT SETTINGS")
    print("─"*80)
    print(f"🖼️  Image: {Path(current_image).name if current_image else 'None'}")
    print(f"🏷️  Category Filter: {category_filter if category_filter else 'None (all categories)'}")
    print(f"📊 Top-K Retrieval: {top_k}")
    print(f"🌡️  Temperature: {temperature}")
    print(f"🎯 Similarity Threshold: {similarity_threshold}")
    print("─"*80 + "\n")

def main():
    """Interactive CLI for RAG system with VLLM"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'tsdsrd',
        'user': 'amir',
        'password': 'amir123',
        'port': 5432
    }
    
    print("Initializing RAG system with VLLM server...")
    
    # Initialize RAG system
    try:
        rag = MultimodalRAGSystemVLLM(
            db_config=db_config,
            vllm_url="http://localhost:8000",
            embedding_model_name="BAAI/bge-large-en-v1.5"
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize RAG system: {e}")
        return
    
    print_header()
    
    # Default settings
    current_image = None
    category_filter = None
    top_k = 5
    temperature = 0.3
    similarity_threshold = 0.3
    
    while True:
        try:
            # Show compact settings
            settings = []
            if current_image:
                settings.append(f"🖼️  {Path(current_image).name}")
            if category_filter:
                settings.append(f"🏷️  {category_filter}")
            settings.append(f"K={top_k}")
            settings.append(f"T={temperature}")
            settings.append(f"Th={similarity_threshold}")
            
            print(f"⚙️  [{' | '.join(settings)}]")
            
            # Get user input
            user_input = input("💬 Your question: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command in ['/quit', '/exit']:
                    print("\n👋 Goodbye!")
                    break
                
                elif command == '/help':
                    print_header()
                    continue
                
                elif command == '/stats':
                    print_current_settings(current_image, category_filter, top_k, temperature, similarity_threshold)
                    continue
                
                elif command == '/image':
                    if not args:
                        print("❌ Usage: /image <path>")
                        print("   Example: /image /home/amir/Downloads/klcc.jpg")
                    else:
                        image_path = args.strip()
                        
                        # Expand ~ to home directory
                        image_path = os.path.expanduser(image_path)
                        
                        if Path(image_path).exists():
                            current_image = image_path
                            print(f"✓ Image set: {image_path}")
                            print("   Now ask your question about this image!")
                        else:
                            print(f"❌ Image not found: {image_path}")
                            print(f"   Make sure the file exists and the path is correct")
                            
                            # Try to suggest corrections
                            if Path(image_path.lower()).exists():
                                print(f"   💡 Did you mean: {image_path.lower()}")
                    continue
                
                elif command == '/clear':
                    if current_image:
                        print(f"✓ Image cleared: {Path(current_image).name}")
                        current_image = None
                    else:
                        print("ℹ️  No image was set")
                    continue
                
                elif command == '/category':
                    if not args:
                        print("❌ Usage: /category <name>")
                        print("   Available categories:")
                        print("     • Perils")
                        print("     • Electrical")
                        print("     • Housekeeping")
                        print("     • Human Element")
                        print("     • Process")
                    else:
                        category_filter = args.strip()
                        print(f"✓ Category filter set: {category_filter}")
                    continue
                
                elif command == '/reset':
                    if category_filter:
                        print(f"✓ Category filter reset (was: {category_filter})")
                        category_filter = None
                    else:
                        print("ℹ️  No category filter was set")
                    continue
                
                elif command == '/topk':
                    if not args:
                        print("❌ Usage: /topk <number>")
                        print("   Example: /topk 10")
                    else:
                        try:
                            new_topk = int(args)
                            if new_topk > 0:
                                top_k = new_topk
                                print(f"✓ Top-K set to: {top_k}")
                            else:
                                print("❌ Top-K must be positive")
                        except ValueError:
                            print("❌ Invalid number")
                    continue
                
                elif command == '/temp':
                    if not args:
                        print("❌ Usage: /temp <number>")
                        print("   Example: /temp 0.7")
                        print("   Range: 0.0 (deterministic) to 1.0 (creative)")
                    else:
                        try:
                            new_temp = float(args)
                            if 0 <= new_temp <= 1:
                                temperature = new_temp
                                print(f"✓ Temperature set to: {temperature}")
                            else:
                                print("❌ Temperature must be between 0.0 and 1.0")
                        except ValueError:
                            print("❌ Invalid number")
                    continue
                
                elif command == '/threshold':
                    if not args:
                        print("❌ Usage: /threshold <number>")
                        print("   Example: /threshold 0.5")
                        print("   Range: 0.0 (any match) to 1.0 (exact match)")
                    else:
                        try:
                            new_threshold = float(args)
                            if 0 <= new_threshold <= 1:
                                similarity_threshold = new_threshold
                                print(f"✓ Similarity threshold set to: {similarity_threshold}")
                            else:
                                print("❌ Threshold must be between 0.0 and 1.0")
                        except ValueError:
                            print("❌ Invalid number")
                    continue
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("   Type /help for available commands")
                    continue
            
            # Process query
            print("\n" + "─"*80)
            
            if current_image:
                print(f"🖼️  Using image: {Path(current_image).name}")
            
            result = rag.query(
                question=user_input,
                image_path=current_image,
                category_filter=category_filter,
                top_k=top_k,
                temperature=temperature,
                similarity_threshold=similarity_threshold,
                show_sources=True
            )
            
            # Check for errors
            if 'error' in result:
                print(f"\n⚠️  Error: {result.get('error')}")
            
            print(f"\n💡 Answer:\n{result['answer']}\n")
            
            # Show performance metrics in a nice box
            print_performance_box(result)
            
            # Show sources
            if result.get('sources'):
                print(f"\n📚 Sources ({result['num_sources']}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. [{source['category']}] Section {source['section']}: {source['title']}")
                    print(f"     Similarity: {source['similarity']} | Risk: {source['risk_type']}")
                    if source.get('regulations'):
                        regs = source['regulations'][:2]  # Show first 2 regulations
                        print(f"     Regulations: {', '.join(regs)}")
            elif result['num_sources'] == 0:
                print(f"\n⚠️  No relevant sources found")
                print(f"   💡 Try: /threshold 0.2 (to lower the threshold)")
            
            print("─"*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
# ```

# ## Usage Examples:

# ### Example 1: Text-only query
# ```
# 💬 Your question: What are the LPG safety recommendations?
# ```

# ### Example 2: Image + text query (correct way)
# ```
# 💬 Your question: /image /home/amir/Downloads/klcc.jpg
# ✓ Image set: /home/amir/Downloads/klcc.jpg

# 💬 Your question: What safety risks do you see in this building?
# ```

# ### Example 3: Using category filter
# ```
# 💬 Your question: /category Electrical
# ✓ Category filter set: Electrical

# 💬 Your question: What electrical inspections are needed?
# ```

# ### Example 4: Adjust settings
# ```
# 💬 Your question: /topk 10
# ✓ Top-K set to: 10

# 💬 Your question: /temp 0.7
# ✓ Temperature set to: 0.7

# 💬 Your question: Tell me about fire safety
# ```

# ## Quick Reference Card:
# ```
# ┌────────────────────────────────────────────────────────────┐
# │                    QUICK REFERENCE                         │
# ├────────────────────────────────────────────────────────────┤
# │ Setting an image:                                          │
# │   /image /path/to/image.jpg                               │
# │   Then: What do you see?                                  │
# │                                                            │
# │ Filtering results:                                         │
# │   /category Electrical                                     │
# │   /reset (to clear filter)                                │
# │                                                            │
# │ Adjusting retrieval:                                       │
# │   /topk 10      (get more context)                        │
# │   /threshold 0.2 (lower = more results)                   │
# │                                                            │
# │ Adjusting creativity:                                      │
# │   /temp 0.1     (precise, deterministic)                  │
# │   /temp 0.7     (creative, varied)                        │
# └────────────────────────────────────────────────────────────┘
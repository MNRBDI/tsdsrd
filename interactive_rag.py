# interactive_rag_vllm_fixed_path.py

from multimodal_rag import MultimodalRAGSystemVLLM
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
            
            # Get user input - PRESERVE CASE!
            user_input = input("💬 Your question: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands - ONLY lowercase the command, not the arguments!
            if user_input.startswith('/'):
                # Split only on first space to preserve path
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()  # Only lowercase the command
                args = parts[1] if len(parts) > 1 else ""  # Keep args with original case
                
                if command in ['/quit', '/exit']:
                    print("\n👋 Goodbye!")
                    break
                
                elif command == '/help':
                    print_header()
                    continue
                
                elif command == '/image':
                    if not args:
                        print("❌ Usage: /image <path>")
                        print("   Example: /image /home/amir/Desktop/MRE\\ TSD/rib_chunks/klcc.jpg")
                        print("   Or use quotes: /image \"/home/amir/Desktop/MRE TSD/rib_chunks/klcc.jpg\"")
                    else:
                        # Remove quotes if present
                        image_path = args.strip().strip('"').strip("'")
                        
                        # Expand ~ to home directory
                        image_path = os.path.expanduser(image_path)
                        
                        # Debug output
                        print(f"   Searching for: {image_path}")
                        print(f"   File exists: {Path(image_path).exists()}")
                        
                        if Path(image_path).exists():
                            current_image = image_path
                            print(f"✓ Image set: {image_path}")
                            print("   Now ask your question about this image!")
                        else:
                            print(f"❌ Image not found: {image_path}")
                            
                            # Try to help
                            parent_dir = Path(image_path).parent
                            filename = Path(image_path).name
                            
                            if parent_dir.exists():
                                print(f"\n   📁 Directory exists. Looking for image files...")
                                # Case-insensitive search
                                all_files = list(parent_dir.glob("*"))
                                matching_files = [f for f in all_files if f.name.lower() == filename.lower()]
                                
                                if matching_files:
                                    print(f"   💡 Found with different case:")
                                    for f in matching_files:
                                        print(f"      {f}")
                                else:
                                    print(f"   Available image files:")
                                    image_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']]
                                    for f in image_files[:10]:
                                        print(f"      {f.name}")
                            else:
                                print(f"   ❌ Directory doesn't exist: {parent_dir}")
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
            
            # Show performance metrics
            print_performance_box(result)
            
            # Show sources
            if result.get('sources'):
                print(f"\n📚 Sources ({result['num_sources']}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. [{source['category']}] Section {source['section']}: {source['title']}")
                    print(f"     Similarity: {source['similarity']} | Risk: {source['risk_type']}")
                    if source.get('regulations'):
                        regs = source['regulations'][:2]
                        print(f"     Regulations: {', '.join(regs)}")
            elif result['num_sources'] == 0:
                print(f"\n⚠️  No relevant sources found")
                print(f"   💡 Try: /threshold 0.2")
            
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
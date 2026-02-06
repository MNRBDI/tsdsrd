# interactive_rag_tsd_rib.py

from multimodal_rag_observation import MultimodalRAGSystemTSDRIB
import os
from pathlib import Path

def print_header():
    """Print application header"""
    print("\n" + "="*80)
    print(" "*15 + "TSD RIB MULTIMODAL RAG ASSISTANT")
    print("="*80)
    print("\n📖 How to use:")
    print("  MODE 1 - Text Query:")
    print("    Just type your description of the safety/structural issue")
    print("    Example: 'I see cracks in the foundation with ground settlement'")
    print("\n  MODE 2 - Image Analysis:")
    print("    Use /image command followed by the image path")
    print("    Example: /image /path/to/image.jpg")
    print("\n  The system will:")
    print("    • Text mode: Search RIB → Generate recommendations (2 stages)")
    print("    • Image mode: Describe → Search → Recommend (3 stages)")
    print("\n⚙️  Commands:")
    print("  /image <path>     - Analyze an image with the 3-stage pipeline")
    print("  /text             - Switch to text query mode (default)")
    print("  /category <name>  - Filter by category (Perils, Electrical, etc.)")
    print("  /reset            - Reset category filter")
    print("  /topk <number>    - Set number of observations to retrieve (default: 3)")
    print("  /temp <number>    - Set temperature (0.0-1.0, default: 0.2)")
    print("  /threshold <num>  - Set similarity threshold (0.0-1.0, default: 0.4)")
    print("  /stats            - Show current settings")
    print("  /help             - Show this help message")
    print("  /quit or /exit    - Exit the application")
    print("="*80)
    print("\n💡 Tips:")
    print("   • Be specific in your text descriptions for better matches")
    print("   • Use technical terms like 'subsidence', 'ground settlement', etc.")
    print("   • Lower the threshold if you're not finding matches")
    print("="*80 + "\n")

def print_performance_box(result: dict):
    """Print a nice performance metrics box"""
    total_time = result.get('total_time', 0)
    tps = result.get('tokens_per_second', 0)
    num_sources = result.get('num_sources', 0)
    
    print("\n┌" + "─"*78 + "┐")
    print(f"│ {'⚡ PERFORMANCE METRICS':^78} │")
    print("├" + "─"*78 + "┤")
    print(f"│ ⏱️  Total Pipeline Time: {total_time:>8.2f}s{' '*43}│")
    print(f"│ ⚡ Generation Speed:     {tps:>8.2f} tok/s{' '*39}│")
    print(f"│ 📚 Sources Retrieved:    {num_sources:>8}{' '*43}│")
    print("└" + "─"*78 + "┘")

def print_stage_separator(stage_num: int, stage_name: str):
    """Print a visual separator for pipeline stages"""
    print(f"\n{'='*80}")
    print(f"  STAGE {stage_num}: {stage_name}")
    print(f"{'='*80}\n")

def process_text_query(rag, query_text: str, top_k: int, category_filter, 
                       similarity_threshold: float, temperature: float):
    """Process a text query through the 2-stage pipeline"""
    print(f"\n🔍 Query: {query_text[:100]}{'...' if len(query_text) > 100 else ''}")
    print("\n🚀 Starting 2-stage text query pipeline...")
    
    result = rag.query_with_text(
        query_text=query_text,
        top_k=top_k,
        category_filter=category_filter,
        similarity_threshold=similarity_threshold,
        temperature=temperature,
        show_sources=True
    )
    
    print("\n" + "="*80)
    
    if result.get('success'):
        # Display recommendations
        print_stage_separator(2, "RIB RECOMMENDATIONS & REGULATIONS")
        print(result['answer'])
        
        # Show performance metrics
        print_performance_box(result)
        
        # Show matched sources
        if result.get('sources'):
            print(f"\n📚 Matched RIB Sections ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. Section {source['section']}: {source['title']}")
                print(f"     Category: {source['category']} | Similarity: {source['similarity']}")
        elif result['num_sources'] == 0:
            print(f"\n⚠️  No relevant RIB sections found")
            print(f"   💡 Try: /threshold 0.3 or rephrase your query")
        
        print("="*80 + "\n")
        
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        print()

def process_image_query(rag, image_path: str, top_k: int, category_filter,
                       similarity_threshold: float, temperature: float):
    """Process an image through the 3-stage pipeline"""
    print(f"✓ Image found: {image_path}")
    print("\n🚀 Starting 3-stage image analysis pipeline...")
    
    result = rag.query_with_image(
        image_path=image_path,
        top_k=top_k,
        category_filter=category_filter,
        similarity_threshold=similarity_threshold,
        temperature=temperature,
        show_sources=True
    )
    
    print("\n" + "="*80)
    
    if result.get('success'):
        # Display Stage 1 output
        print_stage_separator(1, "IMAGE DESCRIPTION")
        print(result['image_description'])
        
        # Display Stage 3 output
        print_stage_separator(3, "RIB RECOMMENDATIONS & REGULATIONS")
        print(result['answer'])
        
        # Show performance metrics
        print_performance_box(result)
        
        # Show matched sources
        if result.get('sources'):
            print(f"\n📚 Matched RIB Sections ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. Section {source['section']}: {source['title']}")
                print(f"     Category: {source['category']} | Similarity: {source['similarity']}")
        elif result['num_sources'] == 0:
            print(f"\n⚠️  No relevant RIB sections found")
            print(f"   💡 Try: /threshold 0.3")
        
        print("="*80 + "\n")
        
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        if result.get('image_description'):
            print(f"\n📝 Image was described as:")
            print(result['image_description'])
        print()

def main():
    """Interactive CLI for TSD RIB RAG system"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'tsdsrd',
        'user': 'amir',
        'password': 'amir123',
        'port': 5432
    }
    
    print("Initializing TSD RIB RAG system with VLLM server...")
    
    # Initialize RAG system
    try:
        rag = MultimodalRAGSystemTSDRIB(
            db_config=db_config,
            vllm_url="http://localhost:8000",
            embedding_model_name="BAAI/bge-large-en-v1.5"
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize RAG system: {e}")
        return
    
    print_header()
    
    # Default settings
    category_filter = None
    top_k = 3
    temperature = 0.2
    similarity_threshold = 0.4
    query_mode = "text"  # "text" or "image"
    
    while True:
        try:
            # Show compact settings
            settings = []
            mode_icon = "📝" if query_mode == "text" else "🖼️"
            settings.append(f"{mode_icon} {query_mode.upper()}")
            if category_filter:
                settings.append(f"🏷️  {category_filter}")
            settings.append(f"K={top_k}")
            settings.append(f"T={temperature}")
            settings.append(f"Th={similarity_threshold}")
            
            print(f"⚙️  [{' | '.join(settings)}]")
            
            # Get user input with appropriate prompt
            if query_mode == "text":
                user_input = input("💬 Describe the issue (or use /image or /help): ").strip()
            else:
                user_input = input("💬 Enter query or command: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                # Split only on first space to preserve path
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command in ['/quit', '/exit']:
                    print("\n👋 Goodbye!")
                    break
                
                elif command == '/help':
                    print_header()
                    continue
                
                elif command == '/text':
                    query_mode = "text"
                    print("✓ Switched to text query mode")
                    continue
                
                elif command == '/stats':
                    print("\n📊 Current Settings:")
                    print(f"  🎯 Query Mode: {query_mode.upper()}")
                    print(f"  🏷️  Category Filter: {category_filter if category_filter else 'None'}")
                    print(f"  📊 Top-K: {top_k}")
                    print(f"  🌡️  Temperature: {temperature}")
                    print(f"  🎯 Similarity Threshold: {similarity_threshold}")
                    print()
                    continue
                
                elif command == '/image':
                    if not args:
                        print("❌ Usage: /image <path>")
                        print("   Example: /image /home/amir/Desktop/MRE\\ TSD/1.1\\ subsidence.png")
                        print("   Or use quotes: /image \"/home/amir/Desktop/MRE TSD/1.1 subsidence.png\"")
                    else:
                        # Remove quotes if present
                        image_path = args.strip().strip('"').strip("'")
                        
                        # Expand ~ to home directory
                        image_path = os.path.expanduser(image_path)
                        
                        if Path(image_path).exists():
                            process_image_query(
                                rag, image_path, top_k, category_filter,
                                similarity_threshold, temperature
                            )
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
            
            else:
                # Not a command - treat as text query
                if query_mode == "text":
                    # Process as text query
                    process_text_query(
                        rag, user_input, top_k, category_filter,
                        similarity_threshold, temperature
                    )
                else:
                    print("💡 In image mode. Use /image <path> to analyze an image")
                    print("   Or use /text to switch to text query mode")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
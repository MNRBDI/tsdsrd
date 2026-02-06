import pdfplumber
from pathlib import Path

def extract_images_from_page_range(pdf_path, output_folder, start_page=None, end_page=None):
    """
    Extract ALL images from a specified page range.
    
    Parameters:
    - pdf_path: Path to the PDF file
    - output_folder: Folder to save extracted images
    - start_page: Starting page number (1-indexed). If None, will prompt user.
    - end_page: Ending page number (1-indexed). If None, will prompt user.
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages in document: {total_pages}")
        
        # Get page range from user if not provided
        if start_page is None:
            while True:
                try:
                    start_page = int(input(f"Enter starting page (1-{total_pages}): "))
                    if 1 <= start_page <= total_pages:
                        break
                    else:
                        print(f"Please enter a number between 1 and {total_pages}")
                except ValueError:
                    print("Please enter a valid number")
        
        if end_page is None:
            while True:
                try:
                    end_page = int(input(f"Enter ending page ({start_page}-{total_pages}): "))
                    if start_page <= end_page <= total_pages:
                        break
                    else:
                        print(f"Please enter a number between {start_page} and {total_pages}")
                except ValueError:
                    print("Please enter a valid number")
        
        # Validate page range
        if not (1 <= start_page <= total_pages):
            print(f"Error: Start page must be between 1 and {total_pages}")
            return
        
        if not (start_page <= end_page <= total_pages):
            print(f"Error: End page must be between {start_page} and {total_pages}")
            return
        
        print(f"\nWill extract images from page {start_page} to page {end_page}")
        print(f"{'='*60}")
        
        # Convert to 0-indexed for pdfplumber
        start_idx = start_page - 1
        end_idx = end_page
        
        # Extract images from specified page range
        total_images = 0
        for page_num in range(start_idx, end_idx):
            page = pdf.pages[page_num]
            images = page.images
            
            print(f"Processing page {page_num + 1}/{total_pages}: Found {len(images)} images")
            
            # Process all images on this page
            for img_idx, image_obj in enumerate(images):
                try:
                    # Extract image from bounding box
                    img_bbox = (image_obj['x0'], image_obj['top'], 
                                image_obj['x1'], image_obj['bottom'])
                    cropped = page.crop(img_bbox)
                    
                    # Create filename with page number and image index
                    img_name = f"page{page_num + 1}_img{img_idx + 1}.png"
                    img_path = output_path / img_name
                    
                    # Convert and save the image
                    pil_img = cropped.to_image(resolution=150)
                    pil_img.save(img_path)
                    
                    print(f"  -> Saved: {img_path}")
                    total_images += 1
                    
                except Exception as e:
                    print(f"  -> Error saving image {img_idx + 1} on page {page_num + 1}: {e}")
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE!")
        print(f"Extracted from page {start_page} to page {end_page}")
        print(f"Total images extracted: {total_images}")
        print(f"Images saved to: {output_path.absolute()}")
        print(f"{'='*60}")

# Usage
if __name__ == "__main__":
    pdf_file = "/home/amir/Desktop/MRE TSD/5. Risk Improvement Benchmark 1.pdf"
    output_dir = "RIB_images"
    
    # Option 1: Interactive mode - will prompt for page numbers
    extract_images_from_page_range(pdf_file, output_dir)
    
    # Option 2: Specify page range directly (e.g., pages 5 to 10)
    # extract_images_from_page_range(pdf_file, output_dir, start_page=5, end_page=10)
    
    # Option 3: Extract from a single page (e.g., page 7 only)
    # extract_images_from_page_range(pdf_file, output_dir, start_page=7, end_page=7)
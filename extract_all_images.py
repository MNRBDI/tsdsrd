import pdfplumber
from pathlib import Path

def extract_all_images_from_pdf(pdf_path, output_folder):
    """
    Extract ALL images from the entire PDF document.
    
    Parameters:
    - pdf_path: Path to the PDF file
    - output_folder: Folder to save extracted images
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages in document: {total_pages}")
        print(f"Extracting images from all pages...")
        print(f"{'='*60}")
        
        # Extract images from all pages
        total_images = 0
        for page_num in range(total_pages):
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
        print(f"Processed all {total_pages} pages")
        print(f"Total images extracted: {total_images}")
        print(f"Images saved to: {output_path.absolute()}")
        print(f"{'='*60}")

# Usage
if __name__ == "__main__":
    pdf_file = "/home/amir/Desktop/MRE TSD/5. Risk Improvement Benchmark 1.pdf"
    output_dir = "RIB_images"
    
    extract_all_images_from_pdf(pdf_file, output_dir)
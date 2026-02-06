import pdfplumber
from pathlib import Path

def extract_all_images_from_photograph_page_to_end(pdf_path, output_folder):
    """
    Extract ALL images from the page containing PHOTOGRAPH section to the end of document
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages in document: {len(pdf.pages)}")
        
        # Find the first page with PHOTOGRAPH section
        photograph_page = -1
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text = page.extract_text()
            
            if text and "PHOTOGRAPH" in text.upper():
                photograph_page = page_num
                print(f"Found PHOTOGRAPH section on page {page_num + 1}")
                break
        
        if photograph_page == -1:
            print("No PHOTOGRAPH section found in the document")
            return
        
        print(f"Will extract images from page {photograph_page + 1} to page {len(pdf.pages)}")
        
        # Extract images from photograph_page to the end of document
        total_images = 0
        for page_num in range(photograph_page, len(pdf.pages)):
            page = pdf.pages[page_num]
            images = page.images
            
            print(f"Processing page {page_num + 1}/{len(pdf.pages)}: Found {len(images)} images")
            
            # Process all images on this page (regardless of content)
            for img_idx, image_obj in enumerate(images):
                try:
                    # Extract image from bounding box
                    img_bbox = (image_obj['x0'], image_obj['top'], 
                               image_obj['x1'], image_obj['bottom'])
                    cropped = page.crop(img_bbox)
                    
                    # Create filename with page number
                    img_name = f"page{page_num + 1}_img{img_idx + 1}.png"
                    img_path = output_path / img_name
                    
                    # Convert and save the image
                    pil_img = cropped.to_image(resolution=150)
                    pil_img.save(img_path)
                    
                    print(f"  -> Saved: {img_path}")
                    total_images += 1
                    
                except Exception as e:
                    print(f"  -> Error saving image {img_idx + 1} on page {page_num + 1}: {e}")
        
        print(f"\nEXTRACTION COMPLETE!")
        print(f"Started from page {photograph_page + 1} and processed until page {len(pdf.pages)}")
        print(f"Total images extracted: {total_images}")

# Usage
if __name__ == "__main__":
    pdf_file = "/home/amir/Desktop/MRE TSD/4. Application Form, Survey Report, Photos-1.pdf"
    output_dir = "extracted_photographs"
    
    extract_all_images_from_photograph_page_to_end(pdf_file, output_dir)
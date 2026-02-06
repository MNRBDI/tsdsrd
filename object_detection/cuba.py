import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# Load model
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# Load image
image_path = "/home/amir/Desktop/MRE TSD/RIB_images/page87_img2.png"
image = Image.open(image_path).convert("RGB")

# Text prompts to search for
texts = [["aerosol can", "spray paint can"]]

# Prepare inputs
inputs = processor(text=texts, images=image, return_tensors="pt")

# Run model
with torch.no_grad():
    outputs = model(**inputs)

# Post-process detections
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.15)[0]

# Plot results
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(image)

found = False

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    found = True
    box = box.tolist()
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1 - 5,
            f"{texts[0][label]}: {score:.2f}",
            color='red', fontsize=12, backgroundcolor="white")

plt.axis("off")
output_path = "spray_can_detection_result.png"
plt.savefig(output_path, bbox_inches="tight")
print(f"Result saved to {output_path}")


if not found:
    print("No spray/aerosol cans detected with current threshold.")

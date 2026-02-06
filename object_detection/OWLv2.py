from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

def get_red_circle_mask(pil_image):
    """Detect a red circle in the image and return a mask (None if not found)."""
    image = np.array(pil_image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Red color ranges in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # Clean noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # No red region found

    # Pick largest contour
    c = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(c)

    if radius < 20:  # Too small = probably noise
        return None

    center = (int(x), int(y))
    radius = int(radius)

    # Create circular mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    return mask

def plot_detections(image, results, texts, threshold=0.1):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    roi_mask = get_red_circle_mask(image)

    i = 0
    boxes = results[i]["boxes"]
    scores = results[i]["scores"]
    labels = results[i]["labels"]

    detections = list(zip(boxes, scores, labels))
    detections = [d for d in detections if d[1] >= threshold]
    detections.sort(key=lambda x: x[1].item(), reverse=True)

    for box, score, label in detections:
        xmin, ymin, xmax, ymax = box.tolist()

        # 🔥 If red circle exists, only keep boxes inside it
        if roi_mask is not None:
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)

            if roi_mask[cy, cx] == 0:
                continue  # Skip boxes outside the circle

        width = xmax - xmin
        height = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        ax.text(
            xmin, ymin - 10,
            f'{texts[0][label]}: {score:.2f}',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1),
            fontsize=10, weight='bold'
        )

    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Load model and processor
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# Load and prepare image
image = Image.open("/home/amir/Desktop/MRE TSD/RIB_images/page32_img2.png").convert("RGB")

# Define text queries
texts = [[
    "subsidence", "lightning arrester and lightning event counter", "falling tree", "combustible materials near electrical board",
    "cardboard boxes near electrical panel", "electrical board is not properly closed", "electrical panels are left open, exposing live wires",
    "electrical board is not closed", "exposed wiring", "circuit breaker is burnt", "kill switch is not locked",
    "storage of LPG gas cylinders", "storage of liquid drums without containment", "storage of combustible materials",
    "combustible materials stored at smoking area", "cooking facilities", "massive wastepaper bales near fire control panel",
    "spray coating booth is not equipped with explosion proof type electrical installation", 
    "battery charging socket stored close to combustible materials", "overhead crane", "spray painting within showroom space",

]]

# Process inputs
inputs = processor(text=texts, images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process the results
target_sizes = torch.Tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(
    outputs=outputs,
    target_sizes=target_sizes,
    threshold=0.1
)

# 🔥 Sort detections before printing
i = 0
text_targets = texts[i]
boxes = results[i]["boxes"]
scores = results[i]["scores"]
labels = results[i]["labels"]

detections = list(zip(boxes, scores, labels))
detections.sort(key=lambda x: x[1].item(), reverse=True)

print(f"Found {len(detections)} objects (sorted by confidence):")
for box, score, label in detections:
    box = [round(v, 2) for v in box.tolist()]
    print(f"Detected {text_targets[label]} with confidence {round(score.item(), 3)} at location {box}")

# Visualize results (also sorted)
plot_detections(image, results, texts, threshold=0.1)

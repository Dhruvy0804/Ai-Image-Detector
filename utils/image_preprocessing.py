"""
Image Preprocessing Module
Handles image loading, validation, resizing, and normalization
for the AI vs Real image detection pipeline.
"""

import numpy as np
from PIL import Image, ImageOps
import io


# EfficientNetV2 input size
IMG_SIZE = (224, 224)

# ImageNet normalization values
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG", "WEBP", "BMP", "TIFF"}


def validate_image(uploaded_file):
    """
    Validate that the uploaded file is a supported image format.
    Returns (is_valid, error_message).
    """
    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        
        fmt = img.format
        if fmt and fmt.upper() in SUPPORTED_FORMATS:
            return True, None
        else:
            return False, f"Unsupported format: {fmt}. Supported: {', '.join(SUPPORTED_FORMATS)}"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def load_image(uploaded_file):
    """
    Load an image from an uploaded file and convert to RGB.
    Returns a PIL Image in RGB mode.
    """
    img = Image.open(uploaded_file)
    
    # Convert RGBA, P, L, or other modes to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    return img


def preprocess_for_model(img, target_size=IMG_SIZE):
    """
    Preprocess a PIL image for EfficientNetV2 model input.
    - Resize to target size
    - Convert to numpy float32 array
    - Keep in [0, 255] range (EfficientNetV2 has built-in preprocessing)
    
    IMPORTANT: Do NOT normalize to [0,1] or apply ImageNet mean/std!
    EfficientNetV2 with include_preprocessing=True handles normalization
    internally, expecting raw [0, 255] pixel values.
    
    Returns numpy array of shape (1, H, W, 3).
    """
    # Resize with high-quality resampling
    img_resized = img.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array — keep [0, 255] range!
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch


def preprocess_for_display(img, max_size=800):
    """
    Resize image for display purposes (preserving aspect ratio).
    """
    img_copy = img.copy()
    img_copy.thumbnail((max_size, max_size), Image.LANCZOS)
    return img_copy


def preprocess_for_frequency(img, target_size=(256, 256)):
    """
    Preprocess image for frequency analysis.
    Converts to grayscale and resizes.
    Returns numpy array.
    """
    img_gray = ImageOps.grayscale(img)
    img_resized = img_gray.resize(target_size, Image.LANCZOS)
    return np.array(img_resized, dtype=np.float32)


def get_image_info(img):
    """
    Extract basic image information.
    Returns dict with size, mode, format details.
    """
    return {
        "Width": img.size[0],
        "Height": img.size[1],
        "Mode": img.mode,
        "Format": getattr(img, 'format', 'Unknown'),
        "Aspect Ratio": f"{img.size[0]/img.size[1]:.2f}",
        "Total Pixels": f"{img.size[0] * img.size[1]:,}",
    }

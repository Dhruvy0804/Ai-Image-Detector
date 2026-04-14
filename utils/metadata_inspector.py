"""
Metadata Inspector Module
Extracts and analyzes EXIF metadata from images to detect
signs of AI generation vs authentic camera capture.
Also analyzes image properties (JPEG quality, dimensions) as fallback.
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import os


# EXIF tags that indicate a real camera-captured photo
CAMERA_INDICATORS = [
    'Make', 'Model', 'DateTime', 'DateTimeOriginal',
    'ExposureTime', 'FNumber', 'ISOSpeedRatings',
    'FocalLength', 'Flash', 'WhiteBalance',
    'ExposureProgram', 'MeteringMode', 'LensModel',
    'ShutterSpeedValue', 'ApertureValue', 'BrightnessValue',
    'GPSInfo', 'Software'
]

# Software signatures that indicate AI generation
AI_SOFTWARE_SIGNATURES = [
    'stable diffusion', 'midjourney', 'dall-e', 'dall·e',
    'openai', 'comfyui', 'automatic1111', 'novelai',
    'adobe firefly', 'canva ai', 'playground ai',
    'leonardo ai', 'nightcafe', 'artbreeder',
    'deepai', 'craiyon', 'bing image creator',
    'gemini', 'imagen', 'flux'
]

# Common phone camera aspect ratios
PHONE_ASPECT_RATIOS = [
    (4, 3), (3, 4),     # Standard phone camera
    (16, 9), (9, 16),   # Widescreen
    (1, 1),             # Square (Instagram)
    (3, 2), (2, 3),     # Some DSLRs
    (18.5, 9), (9, 18.5),  # Modern phones
    (19.5, 9), (9, 19.5),
    (20, 9), (9, 20),
]

# Common AI image resolutions
AI_RESOLUTIONS = [
    (512, 512), (768, 768), (1024, 1024), (1536, 1536),
    (2048, 2048), (512, 768), (768, 512), (1024, 768),
    (768, 1024), (1024, 1536), (1536, 1024),
    (1280, 720), (1920, 1080),
]


def extract_exif_data(img):
    """
    Extract EXIF metadata from a PIL Image using multiple methods.
    Returns a dict of readable EXIF tags.
    """
    exif_data = {}
    
    # Method 1: _getexif() (traditional)
    try:
        raw_exif = img._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == 'GPSInfo':
                    gps_data = {}
                    for gps_tag_id in value:
                        gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag_name] = value[gps_tag_id]
                    exif_data[tag_name] = gps_data
                else:
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='replace')
                        except Exception:
                            value = str(value)
                    exif_data[str(tag_name)] = value
    except (AttributeError, Exception):
        pass
    
    # Method 2: getexif() (newer PIL versions)
    if not exif_data:
        try:
            exif_obj = img.getexif()
            if exif_obj:
                for tag_id, value in exif_obj.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='replace')
                        except Exception:
                            value = str(value)
                    exif_data[str(tag_name)] = value
        except (AttributeError, Exception):
            pass
    
    # Method 3: Try exifread library
    if not exif_data:
        try:
            import exifread
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=img.format or 'JPEG')
            img_bytes.seek(0)
            tags = exifread.process_file(img_bytes, details=False)
            if tags:
                for key, value in tags.items():
                    clean_key = key.split(' ')[-1] if ' ' in key else key
                    exif_data[clean_key] = str(value)
        except (ImportError, Exception):
            pass
    
    return exif_data


def analyze_image_properties(img):
    """
    Analyze image properties when EXIF is not available.
    Uses resolution, aspect ratio, and JPEG characteristics
    to guess if it's a real photo or AI-generated.
    
    Returns a score 0.0 (likely AI) to 1.0 (likely real).
    """
    width, height = img.size
    score = 0.5
    details = []
    
    # Check 1: Exact AI resolution match → likely AI
    is_ai_resolution = False
    for ai_w, ai_h in AI_RESOLUTIONS:
        if (width == ai_w and height == ai_h):
            is_ai_resolution = True
            break
    
    if is_ai_resolution:
        score -= 0.20
        details.append("Resolution matches common AI generator output")
    
    # Check 2: Perfect square → often AI
    if width == height:
        score -= 0.10
        details.append("Square aspect ratio (common in AI generators)")
    
    # Check 3: Resolution analysis
    max_dim = max(width, height)
    if max_dim >= 2000:
        score += 0.15
        details.append("High resolution (typical of camera photos)")
    elif max_dim >= 1000:
        score += 0.10
        details.append("Medium-high resolution (consistent with photos)")
    
    # Check 4: Aspect ratio close to camera standard (4:3, 16:9, 3:2)
    ratio = width / height if height > 0 else 1
    phone_ratios = [4/3, 3/4, 16/9, 9/16, 3/2, 2/3]
    ratio_match = False
    for pr in phone_ratios:
        if abs(ratio - pr) < 0.05:
            score += 0.15
            details.append(f"Aspect ratio {ratio:.2f} matches camera standard ({pr:.2f})")
            ratio_match = True
            break
    
    if not ratio_match and not (width == height):
        # Non-standard non-square ratio — could be cropped real photo
        score += 0.05
        details.append(f"Non-standard aspect ratio {ratio:.2f}")
    
    # Check 5: Non-AI resolution (not 512, 768, 1024, etc.) → likely real
    ai_dims = [256, 512, 768, 1024, 1536, 2048]
    if width not in ai_dims and height not in ai_dims:
        score += 0.10
        details.append("Non-standard resolution (not typical AI output)")
    
    # Check 6: JPEG format and quantization
    try:
        fmt = getattr(img, 'format', None)
        if fmt and fmt.upper() == 'JPEG':
            score += 0.05
            details.append("JPEG format (common for real photos)")
    except Exception:
        pass
    
    try:
        if hasattr(img, 'quantization') and img.quantization:
            score += 0.05
            details.append("JPEG quantization tables present")
    except Exception:
        pass
    
    score = max(0.0, min(1.0, score))
    return score, details


def analyze_metadata(img):
    """
    Analyze image metadata to determine likelihood of being
    a real photo vs AI-generated.
    """
    exif_data = extract_exif_data(img)
    
    camera_indicators_found = []
    details = []
    ai_signatures_found = []
    editing_software_found = []
    
    # Check for camera indicators
    for indicator in CAMERA_INDICATORS:
        if indicator in exif_data:
            camera_indicators_found.append(indicator)
    
    # Check software field for AI signatures
    software = str(exif_data.get('Software', '')).lower()
    image_description = str(exif_data.get('ImageDescription', '')).lower()
    
    for sig in AI_SOFTWARE_SIGNATURES:
        if sig in software or sig in image_description:
            ai_signatures_found.append(sig)
    
    # Calculate metadata score
    score = 0.5
    
    if len(exif_data) == 0:
        # No EXIF — use image property analysis as fallback
        prop_score, prop_details = analyze_image_properties(img)
        score = prop_score
        details.extend(prop_details)
        details.append("No EXIF metadata — using image property analysis")
    else:
        # Has EXIF — analyze it
        if len(camera_indicators_found) >= 5:
            score += 0.35
            details.append(f"{len(camera_indicators_found)} camera-related EXIF tags found")
        elif len(camera_indicators_found) >= 2:
            score += 0.2
            details.append(f"{len(camera_indicators_found)} camera-related EXIF tags found")
        elif len(camera_indicators_found) == 0:
            score -= 0.1
            details.append("EXIF data present but no camera-specific tags")
        
        if 'Make' in exif_data and 'Model' in exif_data:
            score += 0.1
            details.append(f"Camera: {exif_data.get('Make', '')} {exif_data.get('Model', '')}")
        
        if 'GPSInfo' in exif_data:
            score += 0.1
            details.append("GPS location data present - strong indicator of real photo")
        
        if ai_signatures_found:
            score -= 0.4
            details.append(f"AI generation software detected: {', '.join(ai_signatures_found)}")
        
        if 'DateTimeOriginal' in exif_data:
            details.append(f"Original capture date: {exif_data['DateTimeOriginal']}")
            score += 0.05
        
        if 'ExposureTime' in exif_data and 'FNumber' in exif_data:
            details.append(f"Exposure: {exif_data.get('ExposureTime')}s, f/{exif_data.get('FNumber')}")
            score += 0.05
    
    score = max(0.0, min(1.0, score))
    
    # Generate verdict
    if score >= 0.7:
        verdict = "Metadata strongly suggests a real camera-captured photo"
    elif score >= 0.5:
        verdict = "Metadata partially consistent with a real photo"
    elif score >= 0.3:
        verdict = "Metadata is inconclusive — limited camera information"
    else:
        verdict = "Metadata suggests this may be AI-generated or heavily processed"
    
    if len(exif_data) == 0:
        verdict = "No EXIF metadata — image properties analysis used as fallback"
    
    return {
        "exif_data": exif_data,
        "camera_indicators_found": camera_indicators_found,
        "camera_indicator_count": len(camera_indicators_found),
        "ai_signatures": ai_signatures_found,
        "editing_software": editing_software_found,
        "metadata_score": score,
        "verdict": verdict,
        "details": details,
        "has_exif": len(exif_data) > 0,
    }


def format_exif_for_display(exif_data, max_items=20):
    """
    Format EXIF data for display in Streamlit.
    Returns a list of (key, value) tuples.
    """
    display_data = []
    
    priority_keys = [
        'Make', 'Model', 'DateTime', 'DateTimeOriginal',
        'ExposureTime', 'FNumber', 'ISOSpeedRatings',
        'FocalLength', 'Flash', 'Software',
        'ImageWidth', 'ImageLength', 'WhiteBalance',
        'LensModel', 'ExposureProgram'
    ]
    
    for key in priority_keys:
        if key in exif_data:
            value = exif_data[key]
            if not isinstance(value, (dict, list, tuple)):
                display_data.append((key, str(value)))
    
    for key, value in exif_data.items():
        if key not in priority_keys and not isinstance(value, (dict, list, tuple)):
            if len(display_data) >= max_items:
                break
            display_data.append((str(key), str(value)[:100]))
    
    return display_data


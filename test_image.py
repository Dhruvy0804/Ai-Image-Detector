"""
Quick diagnostic: Test what scores the app gives for an image.
Usage: python test_image.py <path_to_image>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from PIL import Image
from utils.metadata_inspector import analyze_metadata, extract_exif_data
from utils.image_preprocessing import preprocess_for_model, preprocess_for_frequency
from utils.frequency_analysis import compute_spectral_features

if len(sys.argv) < 2:
    print("Usage: python test_image.py <path_to_image>")
    sys.exit(1)

img_path = sys.argv[1]
print(f"\nTesting: {img_path}\n")

img = Image.open(img_path).convert('RGB')
print(f"   Size: {img.size}, Mode: {img.mode}")

# 1. EXIF
print("\n--- METADATA ANALYSIS ---")
exif = extract_exif_data(Image.open(img_path))
print(f"   Total EXIF tags found: {len(exif)}")
if exif:
    for k, v in list(exif.items())[:15]:
        val_str = str(v)[:80]
        print(f"   {k}: {val_str}")
else:
    print("   WARNING: NO EXIF DATA AT ALL!")

meta = analyze_metadata(Image.open(img_path))
print(f"\n   Camera indicators: {meta['camera_indicator_count']}")
print(f"   Camera tags found: {meta['camera_indicators_found']}")
print(f"   AI signatures: {meta['ai_signatures']}")
print(f"   Has EXIF: {meta['has_exif']}")
print(f"   Metadata score: {meta['metadata_score']:.3f}")
print(f"   Verdict: {meta['verdict']}")

# 2. Frequency
print("\n--- FREQUENCY ANALYSIS ---")
img_gray = preprocess_for_frequency(img)
freq = compute_spectral_features(img_gray)
print(f"   Anomaly score: {freq['anomaly_score']:.3f}")

# 3. Model
print("\n--- MODEL PREDICTION ---")
try:
    import keras
    model = keras.saving.load_model('models/saved_model/ai_vs_real_efficientnet.keras')
    img_pre = preprocess_for_model(img)
    pred = model.predict(img_pre, verbose=0)[0][0]
    print(f"   Model raw output: {pred:.4f}")
    print(f"   (Close to 1 = Real, Close to 0 = AI)")
except Exception as e:
    print(f"   Model error: {e}")
    pred = None

# 4. Ensemble decision
print("\n--- ENSEMBLE DECISION ---")
if meta['ai_signatures']:
    print("   -> AI signature override -> AI-Generated (95%)")
elif meta['camera_indicator_count'] >= 5:
    print(f"   -> Strong EXIF override -> Real ({max(85, meta['metadata_score']*100):.0f}%)")
elif 'Make' in meta.get('exif_data', {}) and 'Model' in meta.get('exif_data', {}):
    print(f"   -> Camera Make/Model override -> Real ({max(78, meta['metadata_score']*100):.0f}%)")
else:
    print("   -> No override, using weighted ensemble")
    if not meta['has_exif']:
        print("   -> No EXIF = 30% penalty applied (likely AI)")

print("\nDone!")

"""
EfficientNetV2-B0 Training Script for AI vs Real Image Detection
================================================================

This script can be run directly on Google Colab or locally with GPU.

Usage (Colab):
    1. Upload this file to Colab or paste into a notebook
    2. Run: !pip install tensorflow kaggle
    3. Upload your kaggle.json API key
    4. Run the script

Usage (Local):
    python train_model.py --data_dir ./data --epochs 20 --batch_size 32

The trained model will be saved as 'ai_vs_real_efficientnet.keras'
"""

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetV2B0
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Configuration
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 5       # Frozen base, train head only
FINE_TUNE_EPOCHS = 15    # Unfreeze top layers
FINE_TUNE_AT = 100       # Unfreeze layers from this index
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
LABEL_SMOOTHING = 0.1
DROPOUT_RATE = 0.4
SEED = 42

CLASS_NAMES = ['AI-Generated', 'Real']


def setup_gpu():
    """Configure GPU memory growth."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU setup error: {e}")
    else:
        print("⚠️ No GPU found. Training will be slow on CPU.")


def download_cifake_dataset(data_dir='./data'):
    """
    Download CIFAKE dataset from Kaggle.
    Requires kaggle.json API credentials.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    print("📥 Downloading CIFAKE dataset from Kaggle...")
    print("   Make sure your kaggle.json is configured.")
    print("   Run: pip install kaggle")
    print("   Place kaggle.json in ~/.kaggle/")
    
    try:
        os.system(f'kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p {data_dir} --unzip')
        print("✅ Dataset downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n📋 Manual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        print("2. Download and extract to:", data_dir)
        print("3. Expected structure:")
        print(f"   {data_dir}/train/REAL/")
        print(f"   {data_dir}/train/FAKE/")
        print(f"   {data_dir}/test/REAL/")
        print(f"   {data_dir}/test/FAKE/")


def create_data_augmentation():
    """Create data augmentation pipeline."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")


def load_dataset(data_dir, subset='training', validation_split=0.2):
    """
    Load dataset using tf.keras.utils.image_dataset_from_directory.
    
    Expected directory structure:
        data_dir/train/REAL/
        data_dir/train/FAKE/
        data_dir/test/REAL/  (or test/)
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if subset == 'training':
        ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='binary',
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            seed=SEED,
            validation_split=validation_split,
            subset='training',
            shuffle=True,
        )
    elif subset == 'validation':
        ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='binary',
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            seed=SEED,
            validation_split=validation_split,
            subset='validation',
            shuffle=False,
        )
    elif subset == 'test':
        ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels='inferred',
            label_mode='binary',
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    else:
        raise ValueError(f"Unknown subset: {subset}")
    
    # Optimize pipeline performance
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds


def build_model():
    """
    Build EfficientNetV2-B0 model with custom classification head.
    
    Architecture:
        Input [0,255] → Augmentation → EfficientNetV2-B0 (has built-in preprocessing)
        → GlobalAvgPool → Dense(256) → Dropout → Dense(1, sigmoid)
    
    NOTE: EfficientNetV2 includes its own preprocessing layer that normalizes
    [0, 255] inputs to [-1, 1]. Do NOT add extra Rescaling layers!
    """
    # Load pre-trained base
    base_model = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_preprocessing=True,  # Built-in normalization [0,255] → [-1,1]
    )
    base_model.trainable = False
    
    # Data augmentation
    data_augmentation = create_data_augmentation()
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Augmentation (only active during training) — images stay in [0, 255]
    x = data_augmentation(inputs)
    
    # Base model handles its own preprocessing internally
    # DO NOT add Rescaling(1./255) — EfficientNetV2 expects [0, 255] input
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(DROPOUT_RATE / 2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model


def compile_model(model, learning_rate=LEARNING_RATE):
    """Compile model with optimizer and loss."""
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]
    )
    return model


def get_callbacks(output_dir='./models/saved_model'):
    """Create training callbacks."""
    os.makedirs(output_dir, exist_ok=True)
    
    return [
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]


def train_model(data_dir, output_dir='./models/saved_model'):
    """
    Full training pipeline:
    Phase 1: Train classification head with frozen base (5 epochs)
    Phase 2: Fine-tune top layers of base model (15 epochs)
    """
    setup_gpu()
    
    print("\n" + "="*60)
    print("🚀 AI vs Real Image Detector - Training Pipeline")
    print("="*60)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_ds = load_dataset(data_dir, subset='training')
    val_ds = load_dataset(data_dir, subset='validation')
    test_ds = load_dataset(data_dir, subset='test')
    
    print(f"  Training batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"  Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"  Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")
    
    # Build model
    print("\n🏗️ Building EfficientNetV2-B0 model...")
    model, base_model = build_model()
    model = compile_model(model, LEARNING_RATE)
    model.summary()
    
    cb = get_callbacks(output_dir)
    
    # ==========================================
    # Phase 1: Train classification head only
    # ==========================================
    print("\n" + "-"*60)
    print("📌 Phase 1: Training classification head (base frozen)")
    print(f"   Epochs: {INITIAL_EPOCHS}, LR: {LEARNING_RATE}")
    print("-"*60)
    
    history1 = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=cb,
        verbose=1
    )
    
    # ==========================================
    # Phase 2: Fine-tune top layers
    # ==========================================
    print("\n" + "-"*60)
    print(f"📌 Phase 2: Fine-tuning (unfreezing from layer {FINE_TUNE_AT})")
    print(f"   Epochs: {FINE_TUNE_EPOCHS}, LR: {FINE_TUNE_LR}")
    print("-"*60)
    
    # Unfreeze base model from FINE_TUNE_AT
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    
    print(f"  Total layers: {len(base_model.layers)}")
    print(f"  Trainable layers: {len([l for l in base_model.layers if l.trainable])}")
    print(f"  Frozen layers: {len([l for l in base_model.layers if not l.trainable])}")
    
    # Recompile with lower learning rate
    model = compile_model(model, FINE_TUNE_LR)
    
    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    
    history2 = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=cb,
        verbose=1
    )
    
    # Merge histories
    history = {}
    for key in history1.history:
        history[key] = history1.history[key] + history2.history[key]
    
    # ==========================================
    # Evaluation
    # ==========================================
    print("\n" + "="*60)
    print("📊 Evaluating on test set...")
    print("="*60)
    
    results = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test Loss:      {results[0]:.4f}")
    print(f"  Test Accuracy:  {results[1]:.4f}")
    print(f"  Test Precision: {results[2]:.4f}")
    print(f"  Test Recall:    {results[3]:.4f}")
    print(f"  Test AUC:       {results[4]:.4f}")
    
    # Get predictions for classification report
    y_true = []
    y_pred_proba = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_proba.extend(preds.flatten())
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\n📋 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Save model (native .keras format — required for modern TF/Keras)
    model_path = os.path.join(output_dir, 'ai_vs_real_efficientnet.keras')
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save training metrics
    metrics = {
        'test_loss': float(results[0]),
        'test_accuracy': float(results[1]),
        'test_precision': float(results[2]),
        'test_recall': float(results[3]),
        'test_auc': float(results[4]),
        'classification_report': report,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'config': {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'initial_epochs': INITIAL_EPOCHS,
            'fine_tune_epochs': FINE_TUNE_EPOCHS,
            'fine_tune_at': FINE_TUNE_AT,
            'learning_rate': LEARNING_RATE,
            'fine_tune_lr': FINE_TUNE_LR,
            'label_smoothing': LABEL_SMOOTHING,
            'dropout_rate': DROPOUT_RATE,
        }
    }
    
    metrics_path = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📈 Metrics saved to: {metrics_path}")
    
    # Generate plots
    save_training_plots(history, y_true, y_pred, y_pred_proba, output_dir)
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    
    return model, history, metrics


def save_training_plots(history, y_true, y_pred, y_pred_proba, output_dir):
    """Generate and save training visualization plots."""
    
    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 0].plot(history.get('precision', []), label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.get('val_precision', []), label='Val Precision', linewidth=2)
    axes[1, 0].plot(history.get('recall', []), label='Train Recall', linewidth=2, linestyle='--')
    axes[1, 0].plot(history.get('val_recall', []), label='Val Recall', linewidth=2, linestyle='--')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 1].plot(history.get('auc', []), label='Train AUC', linewidth=2)
    axes[1, 1].plot(history.get('val_auc', []), label='Val AUC', linewidth=2)
    axes[1, 1].set_title('AUC-ROC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Plots saved to:", output_dir)


def is_notebook():
    """Detect if running inside Jupyter/Colab notebook."""
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is not None:
            return True
        return False
    except (ImportError, AttributeError, NameError):
        return False


# ============================================================
# 🚀 RUN TRAINING
# ============================================================

if is_notebook():
    # ============================
    # ⬇️ COLAB / JUPYTER CONFIG ⬇️
    # Change these paths as needed
    # ============================
    DATA_DIR = './data'         # Path to extracted CIFAKE dataset
    OUTPUT_DIR = './output'     # Where to save model & metrics
    BATCH_SIZE = 32
    FINE_TUNE_EPOCHS = 15

    print("📓 Running in Notebook mode (Colab / Jupyter)")
    print(f"   Data dir:   {DATA_DIR}")
    print(f"   Output dir: {OUTPUT_DIR}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Fine-tune epochs: {FINE_TUNE_EPOCHS}")
    print()

    # Uncomment the next line to download the dataset first:
    # download_cifake_dataset(DATA_DIR)

    train_model(DATA_DIR, OUTPUT_DIR)

else:
    # Command-line mode — safe to use argparse
    if __name__ == '__main__':
        import sys
        # Safety: clear any stray Jupyter args
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if not a.startswith('/root')]

        parser = argparse.ArgumentParser(description='Train AI vs Real Image Detector')
        parser.add_argument('--data_dir', type=str, default='./data',
                            help='Path to dataset directory')
        parser.add_argument('--output_dir', type=str, default='./models/saved_model',
                            help='Path to save model and metrics')
        parser.add_argument('--epochs', type=int, default=FINE_TUNE_EPOCHS,
                            help='Number of fine-tuning epochs')
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                            help='Batch size for training')
        parser.add_argument('--download', action='store_true',
                            help='Download CIFAKE dataset first')

        args = parser.parse_args()

        if args.download:
            download_cifake_dataset(args.data_dir)

        BATCH_SIZE = args.batch_size
        FINE_TUNE_EPOCHS = args.epochs

        train_model(args.data_dir, args.output_dir)

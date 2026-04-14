"""
Grad-CAM Visualization Module
Generates Gradient-weighted Class Activation Maps to visualize
which regions of the image influenced the model's prediction.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image


def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None, pred_index=None):
    """
    Generate Grad-CAM heatmap for the given image.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (1, H, W, 3)
        last_conv_layer_name: Name of the last conv layer (auto-detected if None)
        pred_index: Index of the predicted class (auto-detected if None)
    
    Returns:
        heatmap: numpy array of the Grad-CAM heatmap
    """
    # Auto-detect last conv layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Conv layer
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        raise ValueError("Could not find a convolutional layer in the model")
    
    # Create a model that outputs both the conv layer output and the final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        # For binary classification with single output neuron
        if predictions.shape[-1] == 1:
            class_output = predictions[:, 0]
        else:
            class_output = predictions[:, pred_index]
    
    # Get gradients of the class output with respect to the conv layer output
    grads = tape.gradient(class_output, conv_outputs)
    
    # Global average pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4, colormap='jet'):
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        original_img: PIL Image (original size)
        heatmap: Grad-CAM heatmap array
        alpha: Overlay transparency
        colormap: Matplotlib colormap name
    
    Returns:
        overlaid_img: PIL Image with heatmap overlay
    """
    # Resize heatmap to match original image
    img_width, img_height = original_img.size
    
    # Get colormap
    cm = plt.get_cmap(colormap)
    
    # Resize heatmap
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (img_width, img_height), Image.LANCZOS
        )
    ) / 255.0
    
    # Apply colormap
    heatmap_colored = cm(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Convert original image to array
    original_array = np.array(original_img)
    
    # Blend
    overlaid = (original_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    
    return Image.fromarray(overlaid)


def create_gradcam_figure(original_img, heatmap, prediction_label, confidence):
    """
    Create a publication-quality Grad-CAM visualization figure.
    
    Returns matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0F0F1A')
    
    for ax in axes:
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', color='#E2E8F0', fontsize=13, fontweight='bold')
    
    # Heatmap
    heatmap_display = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            original_img.size, Image.LANCZOS
        )
    ) / 255.0
    axes[1].imshow(heatmap_display, cmap='jet')
    axes[1].set_title('Attention Heatmap', color='#E2E8F0', fontsize=13, fontweight='bold')
    
    # Overlay
    overlaid = overlay_gradcam(original_img, heatmap, alpha=0.5)
    axes[2].imshow(overlaid)
    color = '#10B981' if prediction_label == 'Real' else '#EF4444'
    axes[2].set_title(
        f'Grad-CAM Overlay ({prediction_label}: {confidence:.1%})', 
        color=color, fontsize=13, fontweight='bold'
    )
    
    plt.tight_layout(pad=2.0)
    return fig


def generate_dummy_heatmap(img_size=(224, 224)):
    """
    Generate a plausible-looking dummy heatmap for demo purposes
    when no model is loaded.
    """
    h, w = img_size
    
    # Create a gaussian-like pattern
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Random center points
    np.random.seed(42)
    n_spots = np.random.randint(2, 5)
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for _ in range(n_spots):
        cy = np.random.randint(h // 4, 3 * h // 4)
        cx = np.random.randint(w // 4, 3 * w // 4)
        sigma = np.random.randint(20, 60)
        
        gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        heatmap += gaussian
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    
    return heatmap

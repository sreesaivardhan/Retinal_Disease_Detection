# src/models/model_components.py

import tensorflow as tf
from tensorflow.keras import layers, Model

class LightweightConvNeXtV2:
    """
    Ultra-lightweight ConvNeXt-V2 implementation for resource-constrained environments (e.g., 8GB RAM).
    Features reduced 3x3 depthwise convs and minimal block counts.
    """
    @staticmethod
    def convnext_block(x: tf.Tensor, dim: int, drop_rate: float = 0.1) -> tf.Tensor:
        """Lightweight ConvNeXt V2 Block with LayerScale (Implicit in LN/Dropout)"""
        input_tensor = x
        
        # 1. Depthwise Conv (3x3 instead of 7x7 for efficiency)
        x = layers.Conv2D(dim, 3, padding='same', groups=dim, use_bias=True)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # 2. Point-wise conv (Expansion: 1x1)
        x = layers.Conv2D(4 * dim, 1, use_bias=True)(x)
        x = layers.Activation('gelu')(x)
        
        # 3. Point-wise conv (Compression: 1x1) 
        x = layers.Conv2D(dim, 1, use_bias=True)(x)
        
        # Stochastic Depth / Dropout
        if drop_rate > 0:
            x = layers.Dropout(drop_rate)(x)
            
        x = layers.Add()([input_tensor, x])
        return x

    @staticmethod
    def create_mini_convnext(input_shape: tuple = (224, 224, 3)) -> Model:
        """Creates the Mini-ConvNeXt-V2 Model with highly reduced complexity."""
        inputs = layers.Input(shape=input_shape, dtype=tf.float32)
        
        # Stem (Patchify 4x4, 32 channels)
        x = layers.Conv2D(32, 4, strides=4, padding='valid')(inputs)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Stage 1: [56, 56, 32]
        for _ in range(1): 
            x = LightweightConvNeXtV2.convnext_block(x, 32)
            
        # Downsample to [28, 28, 64]
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Conv2D(64, 2, strides=2, padding='valid')(x)
        
        # Stage 2: [28, 28, 64] 
        for _ in range(1):
            x = LightweightConvNeXtV2.convnext_block(x, 64)
            
        # Downsample to [14, 14, 128]
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Conv2D(128, 2, strides=2, padding='valid')(x)
        
        # Stage 3: [14, 14, 128]
        for _ in range(2):
            x = LightweightConvNeXtV2.convnext_block(x, 128)
            
        # Final Pooling & Features
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu', name='convnext_features')(x)
        
        return Model(inputs, x, name='MiniConvNeXtV2')

class LightweightSwinTransformer:
    """
    Ultra-lightweight Swin Transformer variant for 8GB RAM.
    Simplifies window attention to global-like attention on downsampled features 
    to conserve memory while maintaining a transformer structure.
    """
    @staticmethod  
    def window_attention(x: tf.Tensor, dim: int, num_heads: int = 4, drop_rate: float = 0.1) -> tf.Tensor:
        """Simplified Global/Window Attention for reduced complexity."""
        # Use Keras layers to handle shape inference properly
        input_shape = tf.keras.backend.int_shape(x)  # ✅ Keras-compatible shape
        
        x_flat = layers.Reshape((-1, dim))(x)  # ✅ Use dim parameter
        
        # Multi-head attention (no actual window shifting/partitioning for max memory efficiency)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,  
            key_dim=dim // num_heads,
            dropout=drop_rate
        )(x_flat, x_flat)
        
        # Reshape back to spatial
        if len(input_shape) == 4:  # Batch, Height, Width, Channels
            h_dim = input_shape[1] if input_shape[1] is not None else -1
            w_dim = input_shape[2] if input_shape[2] is not None else -1
            x_reshaped = layers.Reshape((h_dim, w_dim, dim))(attention_output)
        else:
            x_reshaped = attention_output  # Keep as sequence if already flattened
            
        return x_reshaped    
    @staticmethod
    def swin_block(x: tf.Tensor, dim: int) -> tf.Tensor:
        """Lightweight Swin Block (Simplified LayerNorm -> Attention -> LayerNorm -> MLP)"""
        
        # 1. Attention path
        skip_attn = x
        x_attn = layers.LayerNormalization(epsilon=1e-6)(x)
        x_attn = LightweightSwinTransformer.window_attention(x_attn, dim=dim, num_heads=2)
        x = layers.Add()([skip_attn, x_attn])
        
        # 2. MLP path
        skip_mlp = x
        x_mlp = layers.LayerNormalization(epsilon=1e-6)(x)
        x_mlp = layers.Dense(dim * 2, activation='gelu')(x_mlp)  # Reduced expansion factor
        x_mlp = layers.Dense(dim)(x_mlp)
        x = layers.Add()([skip_mlp, x_mlp])
        
        return x

    @staticmethod
    def create_mini_swin(input_shape: tuple = (224, 224, 3)) -> Model:
        """Creates the Mini-Swin Transformer Model."""
        inputs = layers.Input(shape=input_shape, dtype=tf.float32)
        
        # Initial Patch Embedding (8x8 patches, 48 channels) -> [28, 28, 48]
        x = layers.Conv2D(48, 8, strides=8, padding='valid')(inputs)        
        # Stage 1: [28, 28, 48]
        for _ in range(2):
            x = LightweightSwinTransformer.swin_block(x, 48)
            
        # Downsample (Patch Merging) to [14, 14, 96]
        x = layers.Conv2D(96, 2, strides=2, padding='valid')(x)
        
        # Stage 2: [14, 14, 96]
        for _ in range(2):
            x = LightweightSwinTransformer.swin_block(x, 96)
            
        # Final Pooling & Features
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu', name='swin_features')(x)
        
        return Model(inputs, x, name='MiniSwinTransformer')


class HybridModelBuilder:
    """
    Builds the Fusion Model (ConvNeXt-V2 + Swin Transformer Hybrid).
    """
    def __init__(self, config: dict):
        self.input_shape = config['input_shape']
        self.num_classes = config['num_classes']
        self.dropout_rate = config['dropout_rate']
    
    def create_hybrid_model(self) -> Model:
        """Creates the feature-fused hybrid model."""
        print("🔄 Building Hybrid ConvNeXt-V2 + Swin Transformer model...")
        
        inputs = layers.Input(shape=self.input_shape)
        
        # 1. ConvNeXt-V2 Branch
        convnext_branch = LightweightConvNeXtV2.create_mini_convnext(self.input_shape)
        convnext_features = convnext_branch(inputs)
        
        # 2. Swin Transformer Branch
        swin_branch = LightweightSwinTransformer.create_mini_swin(self.input_shape) 
        swin_features = swin_branch(inputs)
        
        # 3. Feature Fusion Head
        combined = layers.Concatenate(name='feature_fusion')([convnext_features, swin_features])
        
        # Cross-modal projection / MLP head
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.Dropout(self.dropout_rate)(combined)
        
        # Classification head
        combined = layers.Dense(128, activation='relu', name='hybrid_features')(combined)
        combined = layers.Dropout(self.dropout_rate / 2)(combined)
        
        # Ensure output layer uses float32 when mixed_precision is active
        outputs = layers.Dense(self.num_classes, activation='softmax', name='classification', dtype=tf.float32)(combined)
        
        model = Model(inputs, outputs, name='ConvNeXtV2_Swin_Hybrid')
        print("✅ Hybrid model created successfully!")
        return model

# End of model_components.py
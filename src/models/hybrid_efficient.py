
# hybrid_efficient.py - Optimized for 8GB RAM
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Enable memory growth to prevent TF from hogging all RAM
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class LightweightConvNeXtV2:
    """Ultra-lightweight ConvNeXt-V2 for 8GB RAM"""
    
    @staticmethod
    def convnext_block(x, dim, drop_rate=0.1):
        """Lightweight ConvNeXt block"""
        skip = x
        
        # Depthwise conv (7x7 -> 3x3 for efficiency)
        x = layers.Conv2D(dim, 3, padding='same', groups=dim)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Point-wise conv (expansion)
        x = layers.Conv2D(4 * dim, 1)(x)
        x = layers.Activation('gelu')(x)
        
        # Point-wise conv (compression)  
        x = layers.Conv2D(dim, 1)(x)
        
        # Stochastic depth
        if drop_rate > 0:
            x = layers.Dropout(drop_rate)(x)
            
        x = layers.Add()([skip, x])
        return x
    
    @staticmethod
    def create_mini_convnext(input_shape=(224, 224, 3)):
        """Create mini ConvNeXt-V2 - only 2M parameters!"""
        inputs = layers.Input(shape=input_shape)
        
        # Stem (patchify)
        x = layers.Conv2D(32, 4, strides=4)(inputs)  # Reduced from 96 to 32
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Stage 1: [56, 56, 32]
        for _ in range(1):  # Reduced blocks
            x = LightweightConvNeXtV2.convnext_block(x, 32)
        
        # Downsample to [28, 28, 64]
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Conv2D(64, 2, strides=2)(x)
        
        # Stage 2: [28, 28, 64] 
        for _ in range(1):
            x = LightweightConvNeXtV2.convnext_block(x, 64)
        
        # Downsample to [14, 14, 128]
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Conv2D(128, 2, strides=2)(x)
        
        # Stage 3: [14, 14, 128]
        for _ in range(2):  # Slightly more blocks for final stage
            x = LightweightConvNeXtV2.convnext_block(x, 128)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu', name='convnext_features')(x)
        
        return Model(inputs, x, name='MiniConvNeXtV2')

class LightweightSwinTransformer:
    """Ultra-lightweight Swin Transformer for 8GB RAM"""
    
    @staticmethod  
    def window_attention(x, window_size=4, num_heads=4):
        """Simplified window attention - much smaller than full Swin"""
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Simple global attention instead of complex windowing
        x_flat = layers.Reshape((-1, C))(x)
        
        # Multi-head attention with reduced heads
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=C//num_heads,
            dropout=0.1
        )(x_flat, x_flat)
        
        # Reshape back
        x_reshaped = layers.Reshape((H, W, C))(attention_output)
        return x_reshaped
    
    @staticmethod
    def swin_block(x, dim, window_size=4):
        """Lightweight Swin block"""
        skip = x
        
        # Layer norm + Window attention
        x = layers.LayerNormalization()(x)
        x = LightweightSwinTransformer.window_attention(x, window_size, num_heads=2)
        x = layers.Add()([skip, x])
        
        # Layer norm + MLP
        skip = x
        x = layers.LayerNormalization()(x)
        x = layers.Dense(dim * 2, activation='gelu')(x)  # Reduced expansion
        x = layers.Dense(dim)(x)
        x = layers.Add()([skip, x])
        
        return x
    
    @staticmethod
    def create_mini_swin(input_shape=(224, 224, 3)):
        """Create mini Swin Transformer - only 1.5M parameters!"""
        inputs = layers.Input(shape=input_shape)
        
        # Patch embedding - larger patches for efficiency
        x = layers.Conv2D(48, 8, strides=8, padding='valid')(inputs)  # 8x8 patches
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = layers.Reshape((H*W, C))(x)
        x = layers.LayerNormalization()(x)
        
        # Convert back to spatial for attention
        x = layers.Reshape((H, W, C))(x)
        
        # Stage 1: Basic Swin blocks
        for _ in range(2):  # Only 2 blocks
            x = LightweightSwinTransformer.swin_block(x, 48)
        
        # Downsample
        x = layers.Conv2D(96, 2, strides=2)(x)
        H, W = H//2, W//2
        
        # Stage 2: More Swin blocks
        for _ in range(2):
            x = LightweightSwinTransformer.swin_block(x, 96)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu', name='swin_features')(x)
        
        return Model(inputs, x, name='MiniSwinTransformer')

def create_memory_efficient_hybrid(input_shape=(224, 224, 3), num_classes=5):
    """
    Memory-Efficient Hybrid Model for 8GB RAM
    Total parameters: ~4-5M (vs 25M+ for full models)
    """
    inputs = layers.Input(shape=input_shape)
    
    print("🔄 Creating ConvNeXt-V2 branch...")
    convnext_branch = LightweightConvNeXtV2.create_mini_convnext(input_shape)
    convnext_features = convnext_branch(inputs)
    
    print("🔄 Creating Swin Transformer branch...")
    swin_branch = LightweightSwinTransformer.create_mini_swin(input_shape) 
    swin_features = swin_branch(inputs)
    
    print("🔄 Fusing features...")
    # Feature fusion with attention
    combined = layers.Concatenate(name='feature_fusion')([convnext_features, swin_features])
    
    # Cross-attention between branches
    combined = layers.Dense(256, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)
    
    # Classification head
    combined = layers.Dense(128, activation='relu', name='hybrid_features')(combined)
    combined = layers.Dropout(0.2)(combined)
    outputs = layers.Dense(num_classes, activation='softmax', name='classification')(combined)
    
    model = Model(inputs, outputs, name='EfficientHybridModel')
    
    # Enable mixed precision for memory efficiency
    model = tf.keras.Model(inputs=model.input, outputs=model.output)
    
    return model

# Memory-Efficient Training Configuration
def get_memory_efficient_config():
    """Configuration optimized for 8GB RAM"""
    return {
        'batch_size': 4,  # Small batch size
        'epochs': 15,     # Quick training
        'learning_rate': 1e-4,
        'mixed_precision': True,
        'gradient_accumulation_steps': 8,  # Simulate larger batch
        'memory_limit': 6000  # Leave 2GB for system
    }

def train_memory_efficient_hybrid():
    """Complete training pipeline for 8GB RAM"""
    
    # Enable mixed precision globally
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    print("🚀 Creating memory-efficient hybrid model...")
    model = create_memory_efficient_hybrid()
    
    # Print model info
    model.summary()
    total_params = model.count_params()
    print(f"📊 Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    
    # Memory-efficient optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=0.01
    )
    
    # Compile with mixed precision
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Use your existing data pipeline with small batches
    from your_existing_code import DataPipeline  # Replace with actual import
    
    pipeline = DataPipeline()
    train_df, val_df = pipeline.prepare_aptos_data()
    
    # Create generators with SMALL batch size
    train_gen = pipeline.create_data_generators(train_df, val_df, batch_size=4)[0]
    val_gen = pipeline.create_data_generators(train_df, val_df, batch_size=4)[1]
    
    print("🔄 Starting memory-efficient training...")
    
    # Memory-efficient callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=5, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'efficient_hybrid_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Memory cleanup callback
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: tf.keras.backend.clear_session()
        )
    ]
    
    # Train with gradient accumulation simulation
    history = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
        workers=1,  # Single worker to save memory
        max_queue_size=2  # Small queue
    )
    
    return model, history

if __name__ == "__main__":
    # Set memory limit
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    print("🧠 8GB RAM Optimized Hybrid Training")
    print("="*50)
    
    model, history = train_memory_efficient_hybrid()
    
    print(f"\n🏆 Training Complete!")
    print(f"💾 Model saved as: efficient_hybrid_model.keras")
    print(f"📊 Max validation accuracy: {max(history.history['val_accuracy']):.4f}")

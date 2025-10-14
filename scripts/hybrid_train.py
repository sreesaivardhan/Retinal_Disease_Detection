# scripts/hybrid_train.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import sys
import json
from typing import Tuple, Dict, Any
from sklearn.metrics import accuracy_score 
import os
import sys
import json
from typing import Tuple, Dict, Any

# Ensure project structure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import professional components
try:
    from src.data.load_data import RetinalDataLoader
    from src.data.preprocess import ImagePreprocessor
    from src.models.model_components import HybridModelBuilder
    from src.utils.evaluation_utils import HybridModelEvaluator
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- Configuration Management ---
CONFIG_PATH = 'config/model_config_15epochs.json'
DEFAULT_CONFIG = {
    "data_root": "data/raw",
    "input_shape": [224, 224, 3],
    "num_classes": 5, # APTOS DR grading (0-4)
    "batch_size": 4,  # Crucial for 8GB RAM
    "epochs": 15,
    "initial_lr": 1e-4,
    "weight_decay": 0.01,
    "dropout_rate": 0.3,
    "baseline_accuracy": 0.7394 # For t-test comparison
}

def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    if not os.path.exists(path):
        print(f"⚠️ Config file not found at {path}. Using default configuration.")
        return DEFAULT_CONFIG
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        # Ensure list input_shape is converted to tuple for Keras
        config['input_shape'] = tuple(config.get('input_shape', DEFAULT_CONFIG['input_shape']))
        return config
    except Exception as e:
        print(f"Error loading config: {e}. Using default configuration.")
        return DEFAULT_CONFIG


# --- Data Pipeline Class (Refined) ---

class ResourceOptimizedDataPipeline:
    """Handles data loading and generation optimized for 8GB RAM."""
    
    def __init__(self, config: Dict[str, Any]):
        self.data_root = config['data_root']
        self.target_size = config['input_shape'][:2]
        self.batch_size = config['batch_size']
        self.num_classes = config['num_classes']
        self.loader = RetinalDataLoader(self.data_root)
        self.preprocessor = ImagePreprocessor(self.target_size)
        
    def prepare_aptos_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads and splits the APTOS dataset (multi-class: 0-4)."""
        print("🔄 Preparing APTOS data for multi-class hybrid model...")
        datasets = self.loader.load_all_datasets()
        aptos_df = datasets['aptos']
        
        # Use existing image_path from your working loader - don't recreate!
        print(f"✅ Using existing image paths from working data loader")
        print(f"   Sample path: {aptos_df['image_path'].iloc[0]}")
        # aptos_df already has correct image_path from your working RetinalDataLoader!

        # Split data stratified by 'diagnosis' (0-4)
        train_df, val_df = train_test_split(
            aptos_df, 
            test_size=0.2, 
            random_state=42,
            stratify=aptos_df['diagnosis']
        )
        
        print(f"✅ APTOS split: {len(train_df)} train, {len(val_df)} validation")
        return train_df, val_df
    
    def create_memory_efficient_generators(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Creates data generators optimized for 8GB RAM."""
        print("🔄 Creating memory-efficient data generators...")
        
        # Convert diagnosis to string for Keras flow_from_dataframe
        train_df['diagnosis'] = train_df['diagnosis'].astype(str)
        val_df['diagnosis'] = val_df['diagnosis'].astype(str)
        
        # Robust Data Augmentation (Lighter for memory, more effective)
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # ✅ Simple preprocessing
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        
        # Validation/Test data uses only the preprocessor
        val_datagen = ImageDataGenerator(
            rescale=1./255 )        
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='diagnosis',
            target_size=self.target_size,
            batch_size=self.batch_size, 
            class_mode='sparse', # Used sparse for integer labels
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='image_path', 
            y_col='diagnosis',
            target_size=self.target_size,
            batch_size=self.batch_size, 
            class_mode='sparse',
            shuffle=False
        )
        
        # Store class names for evaluation
        self.class_names = [str(i) for i in range(self.num_classes)]
        
        print(f"✅ Memory-efficient generators created (Batch Size: {self.batch_size})")
        return train_generator, val_generator


# --- Trainer Class (Professional) ---

class HybridTrainer:
    """The main resource-optimized training and evaluation pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = ResourceOptimizedDataPipeline(config)
        self.model_builder = HybridModelBuilder(config)
        self.model = None

    def setup_environment(self):
        """Sets up memory growth and mixed precision for 8GB RAM optimization."""
        # 1. GPU Memory Growth
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"✅ GPU Memory Growth enabled for {physical_devices[0].name}")

        # 2. Mixed Precision (float16) for memory/speed
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision enabled (mixed_float16) for reduced VRAM usage.")
        except Exception as e:
            print(f"⚠️ Mixed precision failed (likely no compatible GPU): {e}. Using float32.")

        # 3. Create results directories
        os.makedirs("results/models", exist_ok=True)
        os.makedirs("results/data", exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)

    def get_callbacks(self, model_name: str = "convnext_swin_hybrid"):
        """Returns memory-efficient training callbacks."""
        # Use .h5 for better compatibility than .keras across TF versions
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', # Often better to monitor loss
                patience=5, 
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=3, 
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'results/models/{model_name}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

    def train_and_evaluate(self):
        """Executes the full training and analysis pipeline."""
        self.setup_environment()
        
        print("\n============= 🚀 STARTING HYBRID TRAINING PIPELINE 🚀 =============")
        print(f"Parameters: Input {self.config['input_shape']}, Batch Size {self.config['batch_size']}")
        
        # 1. Data Preparation
        train_df, val_df = self.pipeline.prepare_aptos_data()
        train_gen, val_gen = self.pipeline.create_memory_efficient_generators(train_df, val_df)
        
        # 2. Model Creation and Compilation
        self.model = self.model_builder.create_hybrid_model()
        
        total_params = self.model.count_params()
        print(f"\n📊 Model Statistics: {total_params:,} parameters ({total_params/1e6:.2f}M)")

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.config['initial_lr'],
            weight_decay=self.config['weight_decay']
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 3. Training
        print("\n🔄 Starting hybrid model training...")
        history = self.model.fit(
            train_gen,
            epochs=self.config['epochs'], 
            validation_data=val_gen,
            callbacks=self.get_callbacks(),
            verbose=1,
        )
        print("✅ Training complete. Loading best weights...")
        self.model.load_weights('results/models/convnext_swin_hybrid.h5') # Load best weights

        # 4. Evaluation and Feature Extraction
        print("\n🔄 Running final model evaluation...")
        
        # Extract all validation images and labels into NumPy arrays (memory permitting)
        X_test = []
        y_test = []
        
        # Reset generator to ensure all samples are fetched exactly once in order
        val_gen.reset()
        for i in range(len(val_gen)):
            batch_data, batch_labels = val_gen[i]
            X_test.append(batch_data)
            y_test.append(batch_labels)
        
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        
        print(f"   -> Extracted {len(X_test)} validation samples for comprehensive analysis.")

        # 5. Comprehensive Analysis (Including all required viz/metrics)
        evaluator = HybridModelEvaluator(
            model=self.model, 
            test_images=X_test, 
            test_labels=y_test.astype(int), 
            class_names=self.pipeline.class_names
        )
        evaluator.set_history(history)
        evaluator.run_full_evaluation(self.config['baseline_accuracy'])
        
        # 6. Interpretability (Implicit in XAI, but also general model structure)
        print("\n[9] INTERPRETABILITY (Model Structure):")
        print("   -> Hybrid structure allows analysis of both ConvNeXt (local features) and Swin (global context) branches.")
        print("   -> SHAP/LIME provide visual interpretability of predictions.")
        
        # Final result
        final_acc = accuracy_score(y_test.astype(int), np.argmax(self.model.predict(X_test, verbose=0), axis=1))
        print(f"\n🏆 Final Hybrid Model Accuracy: {final_acc:.4f}")
        
        return final_acc


if __name__ == "__main__":
    
    config = load_config()
    
    print("🧠 8GB RAM Optimized Hybrid Training Pipeline")
    print(f"🎯 ConvNeXt-V2 + Swin Transformer ({config['num_classes']} classes)")
    print("="*60)
    
    trainer = HybridTrainer(config)
    try:
        final_accuracy = trainer.train_and_evaluate()
        print(f"\n🎉 Pipeline completed successfully! Final Accuracy: {final_accuracy:.4f}")
    except Exception as e:
        print(f"\n❌ A critical error occurred during the pipeline execution: {e}")
        sys.exit(1)

# End of hybrid_train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.load_data import RetinalDataLoader
from data.preprocess import ImagePreprocessor

class DataPipeline:
    def __init__(self, data_root="data/raw", target_size=(224, 224)):
        self.data_root = data_root
        self.target_size = target_size
        self.loader = RetinalDataLoader(data_root)
        self.preprocessor = ImagePreprocessor(target_size)
        
    def prepare_aptos_data(self):
        """Prepare APTOS dataset for DR classification"""
        print("🔄 Preparing APTOS data for DR classification...")
        
        # Load APTOS dataset
        datasets = self.loader.load_all_datasets()
        aptos_df = datasets['aptos']
        
        # Create binary classification (No DR vs Any DR)
        aptos_df['binary_dr'] = aptos_df['diagnosis'].apply(lambda x: 0 if x == 0 else 1)
        
        # Split data stratified by diagnosis
        train_df, val_df = train_test_split(
            aptos_df, 
            test_size=0.2, 
            random_state=42,
            stratify=aptos_df['diagnosis']
        )
        
        print(f"✅ APTOS split: {len(train_df)} train, {len(val_df)} validation")
        print(f"   DR distribution - Train: {train_df['disease_label'].value_counts().to_dict()}")
        
        # ✅ FIXED: Added missing return statement
        return train_df, val_df
    
    def create_data_generators(self, train_df, val_df):
        """Create Keras data generators with augmentation"""
        print("🔄 Creating data generators...")
        
        # ✅ FIXED: Convert diagnosis to string for sparse class_mode
        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df['diagnosis'] = train_df['diagnosis'].astype(str)
        val_df['diagnosis'] = val_df['diagnosis'].astype(str)
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='diagnosis',  # Multi-class (0-4) as strings
            target_size=self.target_size,
            batch_size=32,
            class_mode='sparse',  # For integer labels
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='image_path', 
            y_col='diagnosis',
            target_size=self.target_size,
            batch_size=32,
            class_mode='sparse',
            shuffle=False
        )
        
        print(f"✅ Data generators created")
        print(f"   Train batches: {len(train_generator)}")
        print(f"   Val batches: {len(val_generator)}")
        
        return train_generator, val_generator

# Test the pipeline
if __name__ == "__main__":
    pipeline = DataPipeline()
    train_df, val_df = pipeline.prepare_aptos_data()
    train_gen, val_gen = pipeline.create_data_generators(train_df, val_df)
    print("✅ Data preprocessing pipeline ready!")

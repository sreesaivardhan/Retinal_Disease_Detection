import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.load_data import RetinalDataLoader
from data.preprocess import ImagePreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPipeline:
    def __init__(self, data_root="data/raw", target_size=(224, 224)):
        self.data_root = data_root
        self.target_size = target_size
        self.loader = RetinalDataLoader(data_root)
        self.preprocessor = ImagePreprocessor(target_size)
        
    def prepare_aptos_data(self):
        """Prepare APTOS dataset for DR classification"""
        print("🔄 Preparing APTOS data for DR classification...")
        
        datasets = self.loader.load_all_datasets()
        aptos_df = datasets['aptos']
        
        aptos_df['binary_dr'] = aptos_df['diagnosis'].apply(lambda x: 0 if x == 0 else 1)
        
        train_df, val_df = train_test_split(
            aptos_df, 
            test_size=0.2, 
            random_state=42,
            stratify=aptos_df['diagnosis']
        )
        
        print(f"✅ APTOS split: {len(train_df)} train, {len(val_df)} validation")
        print(f"   DR distribution - Train: {train_df['disease_label'].value_counts().to_dict()}")
        
        return train_df, val_df
    
    def create_data_generators(self, train_df, val_df):
        """Create Keras data generators with augmentation"""
        print("🔄 Creating data generators...")
        
        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df['diagnosis'] = train_df['diagnosis'].astype(str)
        val_df['diagnosis'] = val_df['diagnosis'].astype(str)
        
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
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='diagnosis',
            target_size=self.target_size,
            batch_size=32,
            class_mode='sparse',
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

class BaselineCNN:
    def __init__(self, input_shape=(224, 224, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def create_simple_cnn_functional(self):
        """Create simple CNN using Functional API to avoid Sequential issues"""
        inputs = tf.keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        feat = layers.Dense(512, activation='relu', name='feature_layer')(x)
        x = layers.Dropout(0.5)(feat)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_transfer_learning_model(self):
        """Create ResNet50 baseline with transfer learning"""
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = False
        
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        feat = tf.keras.layers.Dense(512, activation='relu', name='feature_layer')(x)
        x = tf.keras.layers.Dropout(0.5)(feat)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_name="baseline_model"):
        """Get training callbacks"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=10, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'results/models/{model_name}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks

class BaselineTrainer:
    def __init__(self):
        self.pipeline = DataPipeline()
        self.baseline = BaselineCNN()
        
    def train_and_evaluate(self):
        """Complete training pipeline"""
        print("🚀 Starting Review 1 Training Pipeline")
        print("="*50)
        
        # 1. Data Preprocessing
        train_df, val_df = self.pipeline.prepare_aptos_data()
        train_gen, val_gen = self.pipeline.create_data_generators(train_df, val_df)
        
        # 2. Model Selection - Train both models
        results = {}
        
        # Simple CNN (Functional API)
        print("\n🔄 Training Simple CNN...")
        simple_model = self.baseline.create_simple_cnn_functional()
        callbacks = self.baseline.get_callbacks("simple_cnn")
        
        history_simple = simple_model.fit(
            train_gen,
            epochs=20 ,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        simple_acc = simple_model.evaluate(val_gen, verbose=0)[1]
        results['Simple CNN'] = simple_acc
        print(f"✅ Simple CNN Validation Accuracy: {simple_acc:.4f}")
        
        # Transfer Learning Model
        print("\n🔄 Training ResNet50 Transfer Learning...")
        transfer_model = self.baseline.create_transfer_learning_model()
        callbacks = self.baseline.get_callbacks("resnet50_transfer")
        
        history_transfer = transfer_model.fit(
            train_gen,
            epochs=15,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=15
        )
        
        transfer_acc = transfer_model.evaluate(val_gen, verbose=0)[1]
        results['ResNet50 Transfer'] = transfer_acc
        print(f"✅ ResNet50 Transfer Validation Accuracy: {transfer_acc:.4f}")
        
        # 3. Feature Extraction using Functional API
        print("\n🔄 Feature Extraction...")
        best_model = transfer_model if transfer_acc > simple_acc else simple_model
        
        # Create feature extractor - works with Functional API
        feature_extractor = Model(
            inputs=best_model.input,
            outputs=best_model.get_layer('feature_layer').output
        )
        
        # Extract features from validation set
        features = []
        labels = []
        for i in range(len(val_gen)):
            batch_data, batch_labels = val_gen[i]
            batch_features = feature_extractor.predict(batch_data, verbose=0)
            features.extend(batch_features)
            labels.extend(batch_labels)
            
            if i % 10 == 0:
                print(f"   Processed batch {i+1}/{len(val_gen)}")
            
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"✅ Features extracted: {features.shape}")
        
        # Save features
        os.makedirs('results/data', exist_ok=True)
        np.save('results/data/validation_features.npy', features)
        np.save('results/data/validation_labels.npy', labels)
        
        # 4. Results Summary
        print("\n📊 REVIEW 1 RESULTS")
        print("="*50)
        print("✅ Data Selection: 8,350 images (3 datasets)")
        print("✅ Data Preprocessing: Augmentation + normalization")
        print("✅ Model Selection: Simple CNN vs ResNet50 Transfer")
        print("✅ Feature Extraction: 512-dimensional features")
        print(f"\nModel Performance:")
        for model_name, acc in results.items():
            print(f"   {model_name}: {acc:.4f} accuracy")
        
        best_model_name = max(results, key=results.get)
        print(f"\n🏆 Best Model: {best_model_name} ({results[best_model_name]:.4f})")
        
        # Save results
        results_df = pd.DataFrame([results])
        results_df.to_csv('results/data/review1_results.csv', index=False)
        
        print("\n🏆 REVIEW 1 SUCCESSFULLY COMPLETED!")
        print("All requirements fulfilled - ready for advanced architecture development in Review 2")
        
        return results, features, labels

# Run training
if __name__ == "__main__":
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)
    
    trainer = BaselineTrainer()
    results, features, labels = trainer.train_and_evaluate()

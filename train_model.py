import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import json
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
DATA_DIR = "data/raw"
MODEL_SAVE_PATH = "models/skin_model.h5"

def build_model(num_classes):
    """
    Builds a Transfer Learning model using MobileNetV2.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found. Please run utils/generate_data.py first.")
        return

    # Advanced Data Augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    print(f"Detected classes: {train_generator.class_indices}")
    
    model = build_model(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Callbacks
    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    print("Starting training (Phase 1: Frozen Base)...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    # Phase 2: Fine-tuning
    print("Fine-tuning some base layers...")
    base_model = model.layers[0] # This is not quite right if model is Sequential/Functional, but for our Functional model:
    # Actually model.layers[1] is the base model in this specific functional setup
    for layer in model.layers:
        if isinstance(layer, Model): # In case we nested it, but we didn't
            pass 
    
    # Let's just find the MobileNetV2 layer
    base_idx = -1
    for i, l in enumerate(model.layers):
        if l.name.startswith('mobilenetv2'):
            base_idx = i
            break
    
    if base_idx != -1:
        model.layers[base_idx].trainable = True
        # Freeze all but the last 20 layers of the base model
        for layer in model.layers[base_idx].layers[:-20]:
            layer.trainable = False
            
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Starting training (Phase 2: Fine-tuning)...")
    model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        callbacks=[early_stop, checkpoint]
    )

    # Final Save
    print(f"Saving final model and classes...")
    model.save(MODEL_SAVE_PATH)
    
    with open("models/class_indices.json", "w") as f:
        json.dump(train_generator.class_indices, f)
    
    print("Training complete.")

if __name__ == "__main__":
    train_model()


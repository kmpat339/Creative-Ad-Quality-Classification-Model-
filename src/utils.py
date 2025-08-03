
import os
import tensorflow as tf

CLASS_NAMES = ['low_quality', 'ok']

def build_dataset(data_dir, img_size=(224, 224), batch_size=32, shuffle=True):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return ds

def build_model(num_classes=2, input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False,
                                             input_shape=input_shape)
    base.trainable = False  # fine‑tune later if desired
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(base.input, out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

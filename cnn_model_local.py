import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

def load_dataset2(base_path):
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    
    print(f"Scanning {base_path} for images...")
    
    # os.walk automatically goes through every single nested folder
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                
                # 1. Determine the label based on the folder name
                folder_name = root.lower()
                if 'aegypti' in folder_name:
                    label = 0
                elif 'albopictus' in folder_name:
                    label = 1
                else:
                    continue # Skip if it doesn't match either mosquito type
                    
                # 2. Load and resize the image
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (64, 64))
                
                # 3. Route to Train or Test based on folder suffix
                if '_testing' in folder_name:
                    X_test.append(img)
                    Y_test.append(label)
                else:
                    X_train.append(img)
                    Y_train.append(label)
                    
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

# 1. Load the data using the new function
X_train, Y_train, X_test, Y_test = load_dataset2('dataset2')

print(f"Loaded {len(X_train)} training images.")
print(f"Loaded {len(X_test)} testing images.")

# 2. Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. Build the CNN Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    
    # 2 Output classes (Aegypti and Albopictus)
    layers.Dense(2, activation='softmax') 
])

# 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the CNN
print("Training CNN...")
history = model.fit(X_train, Y_train, 
                    epochs=20, 
                    validation_data=(X_test, Y_test))

# 6. Evaluate and Predict
print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
print(f"CNN Accuracy: {test_acc * 100:.2f}%")

predictions = model.predict(X_test)
y_pred_classes = np.argmax(predictions, axis=1) 

print("\nClassification Report:")
print(classification_report(Y_test, y_pred_classes, target_names=['Aegypti', 'Albopictus']))
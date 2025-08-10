import cv2
import os
import numpy as np
import gdown
import zipfile
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image

# Step 1: Download and extract dataset from Google Drive
file_id = "1vwyS3oV4O4vdgfxo9cjc1aysMszfxnJM"
url = f"https://drive.google.com/uc?id={file_id}"
zip_file = "dataset.zip"
extract_dir = "dataset"

if not os.path.exists(os.path.join(extract_dir, "training_set")):
    print("Downloading dataset from Google Drive...")
    gdown.download(url, zip_file, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Step 2: Define dataset path and categories
DATASET_PATH = os.path.join(extract_dir, "training_set")
CATEGORIES = ["cats", "dogs"]
IMG_SIZE = (96, 64)

# Step 3: Feature extraction
def extract_features_and_labels():
    data = []
    labels = []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_PATH, category)
        if not os.path.exists(folder):
            print(f"Warning: Folder '{folder}' does not exist.")
            continue
        for img_name in os.listdir(folder):
            try:
                img_path = os.path.join(folder, img_name)
                img = Image.open(img_path).convert("L").resize(IMG_SIZE)
                img_np = np.array(img)
                features = hog(img_np, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                data.append(features)
                labels.append(label)
            except Exception as e:
                print(f"Skipped {img_name}: {e}")
                continue
    return np.array(data), np.array(labels)

print("Extracting features...")
X, y = extract_features_and_labels()
print(f"Total samples: {len(X)}")

# Step 4: Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(probability=True)
model.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")

# Step 5: Real-time detection
label_map = {0: "Cat", 1: "Dog"}

def detect_frame(frame):
    gray = cv2.cvtColor(cv2.resize(frame, IMG_SIZE), cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    features = features.reshape(1, -1)
    
    prob = model.predict_proba(features)[0]
    pred = model.predict(features)[0]
    
    if prob[pred] > 0.8:
        return label_map[pred]
    else:
        return None

# Step 6: Open webcam
cap = cv2.VideoCapture(0)
print("Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_frame(frame)
    if result:
        cv2.putText(frame, f"{result} Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Dog/Cat Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

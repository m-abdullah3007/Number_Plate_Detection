
# Automatic License Plate Reader (ALPR)

An end-to-end Automatic License Plate Reader (ALPR) system built in Python using Convolutional Neural Networks (CNN) for plate localization and EasyOCR for character recognition.

---

## 🧠 Key Features

- 📁 **Image Preprocessing** – Loads and resizes car images and annotations to a fixed size (200x200)
- 🧾 **Annotation Parsing** – Extracts license plate coordinates from XML annotations using BeautifulSoup
- 🤖 **CNN-Based Detection** – Trains a CNN (or VGG16) to predict bounding boxes for license plates
- 🧪 **Model Training & Evaluation** – Tracks training and validation accuracy over epochs
- 🧠 **Postprocessing** – Enlarges predicted bounding boxes for more accurate coverage
- 🔤 **OCR with EasyOCR** – Extracts alphanumeric license plate numbers from detected plates
- 📊 **Visualization** – Displays results with bounding boxes and OCR text

---

## 🚘 Dataset

This project utilizes the [`Car License Plate Detection`](https://www.kaggle.com/andrewmvd/car-plate-detection) dataset from Kaggle.  
- 📷 433 car images  
- 🏷️ 433 annotation XML files  
- Used to train the CNN model to locate license plates

---

## 🧪 Methodology

### Steps:
1. Read in the vehicle image
2. Locate the license plate using CNN
3. Extract and crop the plate region
4. Apply EasyOCR to recognize and extract plate number

### Why CNN?
CNNs learn spatial hierarchies of features and generalize well to unseen images. Pure OpenCV approaches often fail with lighting or angle variation, but CNNs can handle these with enough data.

---

## 🏗️ Model Architecture

Uses **VGG16** for transfer learning (can also use a custom CNN).

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Functional)           (None, 6, 6, 512)         14714688  
_________________________________________________________________
dropout (Dropout)            (None, 6, 6, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 18432)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               2359424   
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 260       
=================================================================
```

The final output layer has **4 neurons** for the plate bounding box: `[xMax, xMin, yMax, yMin]` (normalized).

---

## 🔧 Preprocessing

- Resize all images to (200, 200, 3)
- Use BeautifulSoup to parse `.xml` files and extract 4 bounding box coordinates
- Normalize pixel values (0–1) and bounding box positions (divided by 200)

---

## 🧠 Training

A dropout layer after VGG16 prevents overfitting.  
The CNN is trained with MSE loss and Adam optimizer for 50 epochs.

```python
cnn.add(VGG16(weights="imagenet", include_top=False, input_shape=(200, 200, 3)))
cnn.add(keras.layers.Dropout(0.1))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dense(128, activation="relu"))
cnn.add(Dense(64, activation="relu"))
cnn.add(Dense(4, activation="sigmoid"))
```

---

## 🧾 CNN Output (Plate Localization)

The model achieves ~80% validation accuracy. However, bounding boxes may only cover part of the plate.

To improve coverage, the bounding boxes are **enlarged** using the following heuristic based on the size ratio of the plate and image:

```python
height_coef = 1 + ((1 / (np.log(box_image_ratio_height))**2) / 2)
width_coef  = 1 + ((1 / (np.log(box_image_ratio_width))**2) / 2)
```

### 🔍 Before and After Enlarging

**Before:**  
![image](https://user-images.githubusercontent.com/25105806/128644180-2f33e36f-6bec-44ab-a1ba-0127874c9a12.png)

**After:**  
![image](https://user-images.githubusercontent.com/25105806/128785525-0eb7e4f1-5d36-4e11-b24a-82ae1a71b9b8.png)

### Large Plate Example:  
<img src="https://user-images.githubusercontent.com/25105806/128644302-9985ccc9-c4c0-4856-8da5-039c2e155754.png" width="50%" height="50%">

### Small Plate Example:  
<img src="https://user-images.githubusercontent.com/25105806/128644346-2559b3b2-3341-4746-bdc2-cf5145c4243a.png" width="50%" height="50%">

---

## 🔤 OCR with EasyOCR

After restoring plate size to original image scale, EasyOCR is used for final character recognition.

**Example Output:**  
![image](https://user-images.githubusercontent.com/25105806/128644438-5b02378e-f29e-41d3-ba9c-e3dd0fc40ce2.png)

---

## 🛠️ Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- EasyOCR
- BeautifulSoup
- Matplotlib
- NumPy, Pandas

---

## ▶️ How to Run

1. Clone the repository  
2. Ensure the dataset is in this structure:
   ```
   rawData/
   ├── images/
   └── annotations/
   ```
3. Install dependencies:
   ```bash
   pip install numpy==1.16.5
   pip install tensorflow opencv-python easyocr beautifulsoup4 matplotlib
   ```
4. Run the notebook or script to train the model and test predictions.

---

## 📄 License

This project is open-source and available under the **MIT License**.  
You are free to use, modify, and distribute this project with proper attribution.

---

## 👨‍💻 Author

**Muhammad Abdullah**  
Automatic License Plate Recognition using Deep Learning and OCR.

---

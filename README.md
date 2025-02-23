# 🕵🏻‍♂️ DLA DEEPFAKE DETECTION 2024/25 - UNICA  
**Deepfake Detection Project using the OpenForensics dataset**  

## 🧑🏻‍🎓 Students  
#### Francesco Congiu  
> Student ID: 60/73/65300  
>  
>> E-Mail: f.congiu38@studenti.unica.it  

#### Simone Giuffrida  
> Student ID: 60/73/65301  
>  
>> E-Mail: s.giuffrida2@studenti.unica.it  

#### Fabio Littera  
> Student ID: 60/73/65310  
>  
>> E-Mail: f.littera3@studenti.unica.it  

---

## 📌 Description  
This repository contains the code for training and evaluating deepfake detection models using the **OpenForensics** dataset. The project follows two approaches:  
1. **Transfer Learning** with pre-trained models (e.g., MobileNet, Xception).  
2. **Training from Scratch** with a custom neural network.  

---

## 📥 Download the Dataset  
The **OpenForensics** dataset required for the project can be downloaded from the following link:  
🔗 **[OpenForensics Dataset - Zenodo](https://zenodo.org/records/5528418)**  

---

## 🚀 Installation  
To run the project locally, follow these steps:

### **1️⃣ Clone the Repository**  
Open the terminal and run:
```bash
git clone git@github.com:wakaflocka17/DLA_DEEPFAKEDETECTION.git
cd DLA_DEEPFAKEDETECTION
```
(Or, if using HTTPS)

```bash
git clone https://github.com/wakaflocka17/DLA_DEEPFAKEDETECTION.git
cd DLA_DEEPFAKEDETECTION
```

### **2️⃣ Create and Activate a Virtual Environment**
It is recommended to create a virtual environment to isolate dependencies:
```bash
python3 -m venv openforensics_env
source openforensics_env/bin/activate  # macOS/Linux
```
(On Windows, use: openforensics_env\Scripts\activate)

### **3️⃣ Install Dependencies**
Install all necessary libraries:
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up the Project Structure**
Run the following script to create the required folders:
```bash
chmod +x setup_folders.sh
```
First, however, we make the script executable with the command:
```bash
setup_folders.sh
```
This will create:
```plaintext
DLA_DEEPFAKEDETECTION/
│── data/
│   ├── Train/
│   ├── Val/
│   ├── Test-Dev/
│   ├── Test-Challenge/
│   ├── dataset/
│
│── processed_data/
│   ├── Train/
│   │   ├── real/
│   │   ├── fake/
│   ├── Val/
│   │   ├── real/
│   │   ├── fake/
│   ├── Test-Dev/
│   │   ├── real/
│   │   ├── fake/
│   ├── Test-Challenge/
│   │   ├── real/
```

### **5️⃣ Download the Dataset**
To automatically download the OpenForensics dataset, use the provided script:
```python
scripts/download_dataset.py
```
💡 Ensure you have a stable internet connection, as the dataset is large (60GB+).

### **6️⃣ Extract the Dataset**
After downloading, extract all ZIP files:
```python
scripts/extract_all_zips.py --input_dir data --output_dir data
```
💡 This will extract all dataset partitions into the data/ directory.

### **7️⃣ Extract Faces from the Dataset**
Run the following script to extract and preprocess faces:
```python
scripts/extract_faces.py
```
This will process Train, Validation, Test-Dev, and Test-Challenge in one go.


### **8️⃣ Verify Installation**
To check if everything works correctly, run:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```
If no errors appear, the setup is complete! 🎯


---

## 📂 Project Structure  
```plaintext
DLA_DEEPFAKEDETECTION/
│── data/               # Dataset OpenForensics (originale, non modificato)
│   ├── Train/          # Training Data
│   ├── Val/            # Evaluation Data
│   ├── Test-Dev/       # Test-Dev Data
│   ├── Test-Challenge/ # Test-Challenge Data
│   ├── dataset/        # How to save the original dataset
│
│── processed_data/     # Preprocessing output (cropped faces)
│   ├── Train/
│   │   ├── real/       # Real faces extracted from the training set
│   │   ├── fake/       # Fake faces extracted from the training set
│   ├── Val/
│   │   ├── real/       # Real faces extracted for evaluation
│   │   ├── fake/       # Fake faces extracted for evaluation
│   ├── Test-Dev/
│   │   ├── real/       # Real faces extracted for Test-Dev
│   │   ├── fake/       # Fake faces extracted for Test-Dev
│   ├── Test-Challenge/
│   │   ├── real/       # Real faces extracted for Test-Challenge
│   │   ├── fake/       # Fake faces extracted for Test-Challenge
│
│── documentation/      # Documentation, reports, extra material
│── models/             # Saved models (es. file .pth)
│── scripts/            # Scripts (training, preprocessing, ecc.)
│── notebooks/          # Jupyter Notebook for debugging and testing
│── utils/              # Generic utilities and support functions
│── requirements.txt    # Project dependencies
│── setup_folders.sh    # Script for automatic creation of folders
│── README.md           # Documentazione del progetto
```

## 📊 Project Goals
✅ **Face extraction** from images using bounding boxes.  
✅ **Binary classification (fake/real)** of extracted faces.  
✅ **Training with transfer learning** using MobileNet or Xception.  
✅ **Development of a custom CNN** for classification.  
✅ **GPU utilization (MPS on MacBook M4 Pro)** to maximize speed  

---

## 🤝 Contributions  
Feel free to contribute to the project! 💡

### 📌 How to Contribute
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-nuova
   ```
3. Commit your changes:
   ```bash
     git commit -m "Aggiunta nuova feature"
   ```
4. Push your changes:
   ```bash
     git push origin feature-nuova
   ```
6. Open a Pull Request on GitHub.




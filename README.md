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

### **2️⃣ Create and Activate a Virtual Environment
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

### **4️⃣ V4️⃣ Verify Installation**
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
│── data/               # Dataset OpenForensics (originale)
│── processed_data/     # Output di preprocessing (volti ritagliati)
│   ├── real/
│   └── fake/
│── documentation/     # Documenti, relazioni, materiale extra
│── models/             # Modelli salvati (es. file .pth)
│── scripts/            # Script Python (training, preprocessing, ecc.)
│── notebooks/          # Jupyter Notebook per debugging e test
│── requirements.txt    # Dipendenze del progetto
│── README.md           # Documentazione del progetto
```

## 📊 Obiettivi del Progetto  
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




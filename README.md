# 🕵🏻‍♂️ DLA DEEPFAKE DETECTION 2024/25 - UNICA  
**Progetto di rilevazione Deepfake con il dataset OpenForensics**  

## 🧑🏻‍🎓 Students  
#### Francesco Congiu  
> Matricola: 60/73/65300  
>  
>> E-Mail: f.congiu38@studenti.unica.it  

#### Simone Giuffrida  
> Matricola: 60/73/65301  
>  
>> E-Mail: s.giuffrida2@studenti.unica.it  

#### Fabio Littera  
> Matricola: 60/73/65310  
>  
>> E-Mail: f.littera3@studenti.unica.it  

---

## 📌 Descrizione  
Questo repository contiene il codice per l'addestramento e la valutazione di modelli di deepfake detection utilizzando il dataset **OpenForensics**. Il progetto prevede due approcci:  
1. **Transfer Learning** con modelli pre-addestrati (es. MobileNet, Xception).  
2. **Training from Scratch** con una rete personalizzata.  

---

## 📥 Download del Dataset  
Il dataset **OpenForensics** necessario per il progetto può essere scaricato dal seguente link:  
🔗 **[OpenForensics Dataset - Zenodo](https://zenodo.org/records/5528418)**  

---

## 🚀 Installazione  
Per eseguire il progetto in locale, segui questi passaggi:

### **1️⃣ Clonare il Repository**  
Apri il terminale e esegui:
```bash
git clone git@github.com:wakaflocka17/DLA_DEEPFAKEDETECTION.git
cd DLA_DEEPFAKEDETECTION
```
(Oppure, se usi HTTPS)
```bash
git clone https://github.com/wakaflocka17/DLA_DEEPFAKEDETECTION.git
cd DLA_DEEPFAKEDETECTION
```

### **2️⃣ Creare e Attivare un Ambiente Virtuale**
Si consiglia di creare un ambiente virtuale per isolare le dipendenze:
```bash
python3 -m venv openforensics_env
source openforensics_env/bin/activate  # macOS/Linux
```
(Su Windows, usa: openforensics_env\Scripts\activate)

### **3️⃣ Installare le Dipendenze**
Installa tutte le librerie necessarie:
```bash
pip install -r requirements.txt
```

### **4️⃣ Verificare l’Installazione**
Per controllare che tutto funzioni correttamente, esegui:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```
Se non ci sono errori, il setup è completato! 🎯

---

## 📂 Struttura del Progetto
```plaintext
DLA_DEEPFAKEDETECTION/
│── data/               # Dataset OpenForensics
│── models/             # Modelli salvati
│── scripts/            # Script Python per training e preprocessing
│── notebooks/          # Jupyter Notebook per debugging e test
│── requirements.txt    # Dipendenze del progetto
│── README.md           # Documentazione del progetto
```

## 📊 Obiettivi del Progetto  
✅ **Estrazione dei volti** dalle immagini tramite bounding box.  
✅ **Classificazione binaria (fake/real)** dei volti estratti.  
✅ **Training con Transfer Learning** usando MobileNet o Xception.  
✅ **Sviluppo di una CNN personalizzata** per la classificazione.  
✅ **Utilizzo della GPU (MPS su MacBook M4 Pro)** per massimizzare la velocità.  

---

## 🤝 Contributi  
Sentiti libero di contribuire al progetto! 💡  

### 📌 Come Contribuire  
1. Fai un fork del repository.  
2. Crea un nuovo branch:  
   ```bash
   git checkout -b feature-nuova
   ```
3. Fai commit delle tue modifiche:
   ```bash
     git commit -m "Aggiunta nuova feature"
   ```
4. Esegui il push:
   ```bash
     git push origin feature-nuova
   ```
6. Apri una Pull Request su GitHub.




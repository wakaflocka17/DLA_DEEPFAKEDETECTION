# 5. BUILDING A NETWORK FROM SCRATCH
## Descrizione generale
In questa sezione illustriamo la struttura e la motivazione dietro la CNN personalizzata sviluppata per rilevare immagini di tipo DeepFake mediante una classificazione binaria, quindi distinzione tra immagini real e fake.

## Architettura della rete

### Blocco convoluzionale 1
- **Livello convoluzionale** (`Conv2d`)
  - **Input channels**: 3 (RGB)
  - **Output channels**: 32
  - **Kernel size**: 3
  - **Padding**: 1

Questo primo livello iniziale ha il compito di catturare e rilevare le feature a basso livello come bordi e colori di base, mantenendo la dimensione spaziale dell'immagine grazie al padding: per quest'ultimo, abbiamo scelto di impostarlo a 1.
In questo primo livello, abbiamo impostato a 32 il numero di filtri utilizzati `out_channels = 32`, utile per catturare caratteristiche diverse delle immagini di input.

Per quanto riguarda la scelta del kernel, abbiamo deciso di utilizzare un 3x3: nel dominio specifico di applicazione, si presta bene alla rilevazione di piccoli particolari che sono fondamentali per il task di DeepFake Detection.
Abbiamo inoltre effettuato un tentativo con un kernel di dimensioni 7x7: i risultati ottenuti, però, non hanno evidenziato miglioramenti.

- **Batch Normalization** (`BatchNorm2d`)
  - Stabilizza e velocizza il training riducendo lo "shift interno della covariata", rendendo il modello più resistente a variazioni delle distribuzioni degli input.
Infatti capita che, nei livelli iniziali della rete, ci siano bruschi cambiamenti dei pesi: il batch normalization ci permette di stabilizzare il training attraverso la normalizzazione in ogni mini-batch per poi, successivamente, applicare scaling e shifting.

- **Max Pooling** (`MaxPool2d`)
  - **Kernel size**: 2
  - **Stride**: 2

  Riduce dimensionalmente la rappresentazione dell'immagine, diminuendo il carico computazionale e catturando feature dominanti.
  Utilizzando un kernel 2x2 andiamo a dimezzare le dimensioni spaziali della feature map, riducendo i calcoli ma senza perdere informazioni utili: per ogni sotto matrice, infatti, andiamo a selezionare il valore massimo. Questa dimensionalità di kernel ci garantisce, dunque, una riduzione spaziale significativa ma comunque un mantenimento delle informazioni fondamentali della nostra immagini.

### Blocco convoluzionale 2
- **Livello convoluzionale** (`Conv2d`)
  - **Input channels**: 32
  - **Output channels**: 64
  - **Kernel size**: 3
  - **Padding**: 1

  Aumenta la profondità della rete, permettendo di catturare caratteristiche più complesse e strutturate dalle immagini: incrementando gradualmente il numero di filtri, la rete impara rappresentazioni sempre più sofisticate senza un sovraccarico eccessivo in termini computazionali.

- **Batch Normalization** (`BatchNorm2d`)
  - Come nel primo blocco, contribuisce alla stabilità del training, migliorando la generalizzazione.

- **Max Pooling** (`MaxPool2d`)
  - Mantiene l'approccio di riduzione dimensionale.

### Blocco convoluzionale 3
- **Livello convoluzionale** (`Conv2d`)
  - **Input channels**: 64
  - **Output channels**: 128
  - **Kernel size**: 3
  - **Padding**: 1

  Questo blocco è fondamentale per acquisire feature di alto livello, che aiutano la rete a identificare caratteristiche distintive tipiche del task, come anomalie nelle texture e nelle transizioni cromatiche.

- **Batch Normalization** (`BatchNorm2d`)
  - Consente una regolarizzazione efficace e un apprendimento più stabile.

- **Max Pooling** (`MaxPool2d`)
  - Completa la fase convoluzionale della rete riducendo ulteriormente la dimensione spaziale delle feature map, essenziale per minimizzare l'overfitting.

### Livelli completamente connessi (Fully Connected Layers)
- **Primo strato Fully Connected** (`Linear`)
  - **Input**: 128 × 28 × 28
  - **Output**: 512

  Trasforma le rappresentazioni spaziali in una rappresentazione vettoriale compatta utile per la classificazione finale. La dimensione ampia di output (512 neuroni) è stata scelta per consentire al modello di apprendere rappresentazioni ricche e informative.
  L'input che abbiamo sarebbe un vettore di dimensioni 128 * 28 * 28, ovvero 128 features map ciascuna di dimensione 28*28 (l'immagine): quello che il livello FC va a compiere è il flattening, ovvero ottenere dal vettore tridimensionale un vettore decisamente più piccolo (512 nel nostro caso).

- **Funzione di attivazione**: ReLU
  - Introduce non linearità, permettendo di apprendere relazioni più complesse nei dati rispetto a funzioni lineari.

- **Dropout (0.5)**
  - Riduce l’overfitting eliminando casualmente alcuni neuroni durante il training, migliorando significativamente la generalizzazione del modello.
  - Nel nostro caso, con un valore di 0.5, andiamo a disattivare la metà dei neuroni: in questo modo, la nostra rete generalizza meglio e riesce a non dipendere da certi neuroni.

- **Secondo strato Fully Connected** (`Linear`)
  - **Input**: 512
  - **Output**: 2 (classificazione binaria)

  Questo livello finale esegue la classificazione tra immagini autentiche e deepfake, fornendo l'output diretto necessario per il processo decisionale.

## Funzione forward
La funzione `forward` definisce la sequenza con cui i dati passano attraverso i livelli della rete:

- Input → Conv1 → BN1 → ReLU → Pool1
- → Conv2 → BN2 → ReLU → Pool2
- → Conv3 → BN3 → ReLU → Pool3
- → Flatten → FC1 → ReLU → Dropout
- → FC2 → Output

## Tecnologia e motivazione delle scelte
- **PyTorch** è stato scelto per la sua flessibilità e semplicità di implementazione delle CNN, oltre che per la sua ampia diffusione e supporto della community.
- **Batch Normalization** migliora significativamente la velocità di convergenza, rende la rete più robusta agli input e permette di utilizzare learning rate più elevati.
- **Max Pooling** riduce la dimensione spaziale preservando le caratteristiche rilevanti e riducendo il rischio di overfitting.
- **Dropout** è una tecnica fondamentale per regolarizzare la rete e garantire una buona generalizzazione, particolarmente importante nel rilevamento dei deepfake dove è essenziale riconoscere caratteristiche sottili e variegate.



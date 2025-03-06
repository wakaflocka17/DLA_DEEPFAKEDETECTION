# 3. METADATA ANALYSIS
## 3.1 Descrizione Generale del Dataset
Il dataset OpenForensics rappresenta una delle più grandi raccolte di immagini reali e manipulate, specificamente progettata per affrontare le sfide della rilevazione e segmentazione di deepfake multi-faccia in ambienti reali ("in-the-wild"). Il suo obiettivo è quello di superare le limitazioni dei dataset esistenti, che spesso contengono immagini a sfondo uniforme, con una singola faccia e scenari poco rappresentativi della complessità reale.
### 3.1.1 Caratteristiche principali
Tra le varie caratteristiche che compongono questo dataset possiamo sicuramente citare:
> 1. Scalabilità e Dimensioni:
Il dataset comprende 115.325 immagini non ristrette, contenenti complessivamente circa 334.000 volti. Le immagini sono raccolte da fonti diverse (ad es. Google Open Images) e garantiscono una elevata varietà sia per quanto riguarda il contesto scenico che le condizioni di illuminazione e risoluzione.

> 2. Annotazioni Ricche e Multi-Task:
Ogni volto presente nelle immagini è etichettato in maniera dettagliata. Le annotazioni includono informazioni sulla categoria di forgery (reale/falso), bounding box, maschera di segmentazione, contorni della manipolazione e landmark facciali generali. Queste annotazioni supportano non solo i compiti tradizionali di classificazione (deepfake detection) ma anche quelli di localizzazione e segmentazione.

> 3. Diversità degli Scenari:
Il dataset si distingue per la presenza di numerosi scenari reali: immagini indoor e outdoor, volti di differenti dimensioni, orientamenti e condizioni di occlusione. Inoltre, il processo di sintesi dei volti manipolati utilizza tecniche avanzate (come GAN, Poisson blending e adattamento del colore) per generare immagini con alta risoluzione e qualità visiva, che si integrano in maniera naturale nei contesti originali.

> 4. Augmentazioni per la Sfida Reale:
Per simulare condizioni reali, OpenForensics include anche un sottoinsieme "Test-Challenge" in cui sono applicate perturbazioni e trasformazioni (modifiche di colore, corruzione, distorsioni e altri effetti esterni come nebbia, neve e pioggia) per aumentare la variabilità e mettere alla prova la robustezza dei metodi di rilevazione e segmentazione.

> 5. Implicazioni per la Ricerca:
Grazie alla sua ampia scala, alla ricchezza delle annotazioni e alla varietà degli scenari, il dataset offre un terreno di prova ideale non solo per la rilevazione dei deepfake ma anche per ulteriori studi sul riconoscimento facciale, l’apprendimento multi-task e l’analisi della robustezza dei modelli in condizioni di forte variabilità.

## 3.2 Struttura dei File di Metadati (poly.json):
## 3.3 Descrizione della struttura interna dei file poly.json: quali campi contengono, che tipo di informazioni sono memorizzate (ad esempio, coordinate, etichette, attributi specifici).
> Significato di ciascun campo e come interpretarli.
> Esempi di record per illustrare la struttura.
## 3.4 Relazioni e Collegamenti:
> Come i file poly.json si integrano con gli altri file del dataset.
> Eventuali relazioni tra i file di metadati e i dati grezzi (ad esempio, se i file poly.json annotano immagini, video o altri tipi di dati).
## 3.5 Processo di Generazione e Validazione:
> Se applicabile, descrivi come sono stati generati i metadati e quali strumenti o procedure sono stati usati.
> Metodi di validazione e controllo qualità sui metadati.
## 3.6 Utilizzo dei Metadati:
> Come i metadati possono essere sfruttati nelle analisi (es. per filtrare, aggregare o visualizzare i dati).
> Eventuali script o pipeline che utilizzano questi file.

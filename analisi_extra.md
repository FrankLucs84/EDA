## ⚖️ Classificazione vs Regressione: differenza fondamentale

Uno degli aspetti centrali in un’analisi predittiva è la **natura della variabile target**, ossia la variabile che vogliamo prevedere.  
La scelta dell’approccio — *classificazione* o *regressione* — dipende proprio da **come è definita questa variabile**.

---

### 🔷 Regressione (es. `diabetes` dataset)

Nel caso del dataset `diabetes`, la variabile `target` è una **variabile numerica continua**.  
Ogni valore rappresenta un indice quantitativo della **progressione della malattia (diabete)** dopo un anno.  
L'obiettivo del modello è quindi **stimare una quantità reale**, ad esempio:

> _"Dato un paziente con queste caratteristiche cliniche, quale sarà il livello atteso della sua condizione diabetica tra un anno?"_

🧠 Questo rientra nei problemi di **regressione**, dove l’output è un numero reale continuo  
(es. prezzo di una casa, tempo di attesa, livello glicemico, ecc.).

---

### 🔶 Classificazione (es. `wine` dataset)

Nel caso del dataset `wine` di `sklearn`, la variabile `target` rappresenta la **classe di appartenenza** di ciascun vino,  
codificata come etichetta numerica (0, 1, 2).  
Qui l'obiettivo non è stimare un valore continuo, ma **assegnare ciascun esempio a una categoria predefinita**, ad esempio:

> _"Questo vino appartiene alla classe 0, 1 o 2 sulla base delle sue caratteristiche chimiche?"_

📌 Questo rientra nei problemi di **classificazione**, dove l’output è una **categoria discreta**  
(es. malato/sano, sì/no, tipo A/B/C).

---

### 📊 Confronto sintetico

| **Caratteristica**           | **Regressione**                    | **Classificazione**                   |
|------------------------------|------------------------------------|---------------------------------------|
| Tipo di variabile target     | Numerica continua                  | Categoriale/discreta                  |
| Output del modello           | Numero reale                       | Etichetta o classe                    |
| Esempio (`diabetes`)         | Livello stimato della malattia     | ✘ (non applicabile)                   |
| Esempio (`wine`)             | ✘ (non applicabile)                | Classe del vino (es. 0, 1, 2)         |
| Tipo di metrica              | RMSE, MAE, R²                      | Accuracy, Precision, Recall, F1       |

---

Nel contesto dell’EDA, riconoscere il tipo di variabile `target` è essenziale per:

- Adottare visualizzazioni appropriate (es. istogrammi vs countplot)
- Scegliere tecniche statistiche corrette (correlazioni vs distribuzioni per classi)
- Prefigurare il tipo di modello da addestrare nella fase successiva

👉 Nel dataset `diabetes`, la natura **continua** della `target` guida **inevitabilmente verso un approccio di regressione**.

Nel caso del dataset `diabetes`, abbiamo un **dataset numerico, normalizzato**, privo di variabili categoriche,  
in cui l’obiettivo è stimare una **variabile continua legata alla gravità futura della malattia**.

---

### 🎯 Qual è il nostro obiettivo?

Questa è una domanda fondamentale che dobbiamo sempre porci all’inizio di qualsiasi analisi.  
Nel caso del dataset `diabetes`, osserviamo che la variabile `target` è **numerica continua** e rappresenta un **indice di gravità della malattia** valutato un anno dopo la raccolta dei dati clinici iniziali.

Secondo la documentazione ufficiale di Scikit-learn, questo dataset è stato costruito proprio per problemi di tipo **regressivo**, non classificatorio.  
Se volessimo svolgere attività di modeling, l’idea sarebbe quindi quella di **usare le caratteristiche cliniche del paziente** (es. pressione, glicemia, BMI) per **predire l’evoluzione numerica del diabete** nel tempo.

In un contesto esplorativo, invece, possiamo analizzare:
- **Come i diversi valori della variabile target** (cioè i diversi gradi di severità del diabete) si distribuiscono nel campione
- **Quali variabili indipendenti** mostrano pattern o correlazioni più evidenti rispetto alla progressione della malattia

In entrambi i casi, avere chiaro fin da subito **cosa vogliamo predire e perché** ci aiuta a strutturare correttamente tutto il flusso dell’analisi.

---

### Analisi univariata

Dopo aver descritto il dataset nella sua interezza. Sarà importante **analizzare singolarmente ciascuna variabile** per comprenderne struttura, distribuzione, e significato. Questo processo viene spesso definito **analisi univariata**.

L’analisi univariata ci permette di:

- Formarci un’opinione preliminare su ogni variabile
- Valutare la qualità della distribuzione dei dati
- Decidere quali variabili necessitano trasformazioni (es. scaling aggiuntivo, logaritmi)
- Individuare variabili da **trattare con cautela** in fase di modellazione (es. con forti asimmetrie)

È solo dopo aver ben compreso ogni singola variabile che possiamo iniziare ad analizzarne le **relazioni reciproche**, oggetto della fase successiva.

---

### Relazioni tra le variabili

Dopo aver compreso ogni variabile singolarmente, il passo successivo è indagare come le variabili si relazionano tra loro, in particolare come influenzano il target, cioè la progressione del diabete.

Questa fase rappresenta il primo vero momento di data intelligence. In contesti clinici, industriali o aziendali, comprendere le relazioni tra variabili significa trovare leve di intervento, pattern nascosti, e variabili predittive strategiche.

A partire da questa fase, possiamo iniziare a costruire **ipotesi interpretative**:

- Il BMI e i trigliceridi sembrano **fortemente associati** all’aggravarsi della malattia
- Alcuni indicatori lipidici (ldl, hdl) sembrano meno informativi
- Potremmo esplorare **interazioni tra variabili**, es. BMI * pressione o BMI * trigliceridi

Questa fase **abilita il passaggio al modeling predittivo**, aiutandoci a ridurre il numero di variabili e a concentrarci su quelle che **veicolano reale informazione** per il nostro obiettivo.

## EXTRA. Modellazione predittiva (baseline con regressione lineare)

## Modellazione Predittiva: Ipotesi e Approccio

Dopo aver completato l’analisi esplorativa (EDA), è naturale chiedersi:  
**"Possiamo costruire un modello che, dato un nuovo paziente, preveda quanto peggiorerà la sua condizione diabetica in futuro?"**

Per iniziare a rispondere a questa domanda, abbiamo costruito un **modello predittivo di tipo regressivo**, utilizzando la **regressione lineare**.

### 📌 Ipotesi di base

- Le **10 variabili cliniche** (BMI, pressione, trigliceridi, colesterolo...) possono essere utilizzate per **prevedere la gravità futura** del diabete (`diabetes_progression`).
- Il modello assume una **relazione lineare** tra le feature e l’output.
- L’obiettivo è stimare il valore numerico della progressione del diabete a partire dai dati iniziali.

---

## Cosa abbiamo fatto

1. Divisione dei dati in **training set (80%)** e **test set (20%)**.
2. Addestramento di un **modello di regressione lineare** sui dati di training.
3. Valutazione della capacità del modello sui dati di test con tre metriche:
   - **RMSE**: errore medio di previsione
   - **R² Score**: quota della variabilità spiegata
   - **Coefficiente** per ogni variabile: misura il peso della feature

---

# Modellazzione predittiva

## 🎯 Obiettivo
Stimare la variabile target `diabetes_progression` a partire da dati clinici, costruendo un modello interpretabile.

## 🔁 Fasi operative

1. **Separazione delle variabili**
   - `X`: feature predittive (età, BMI, pressione, colesterolo...)
   - `y`: variabile target da stimare

2. **Train/Test Split**
   - Divisione del dataset in 80% training / 20% test
   - Permette di valutare il modello su dati mai visti prima

3. **Costruzione del modello**
   - `LinearRegression()` di scikit-learn
   - Il modello apprende i coefficienti per ogni variabile

4. **Predizione**
   - Applicazione del modello ai dati di test per ottenere `y_pred`

## 🧪 Valutazione & Visualizzazione (output finale)

### [1] Metriche
- **RMSE**: errore medio delle previsioni
- **R²**: percentuale di variabilità spiegata (es. 0.48 = 48%)

### [2] Coefficienti
- Tabella: ogni feature ha un coefficiente stimato
- Indica se la variabile ha impatto positivo o negativo e quanto è forte

### [3] Grafico dei Coefficienti
- Barplot ordinato per valore assoluto
- Evidenzia `ltg` e `bmi` come predittori dominanti

### [4] Scatter Reale vs Predetto
- Confronta ogni valore osservato con il corrispettivo stimato
- Idealmente i punti si dispongono sulla diagonale

### [5] Distribuzione degli Errori
- Istogramma dei residui
- Una forma simmetrica e centrata indica modello ben bilanciato

## Report Sintetico del Modello

## Questa fase ha due obiettivi principali:

- Valutare le performance quantitative (es. RMSE, R²)
- Capire il modello – ovvero: quali variabili influenzano davvero la previsione? Il grafico dei coefficienti risponde al secondo punto:
🔍 Traduce in forma visiva l’impatto di ciascuna feature nel modello di regressione.

### 🧾 Metriche ottenute

- **RMSE**: fornisce un’indicazione dell’errore medio. Più è basso, meglio è.
- **R² Score**: es. 0.48 → il modello spiega circa il 48% della variabilità. Un risultato **discreto**, ma migliorabile.

### 🔍 Analisi dei Coefficienti

Le variabili più influenti risultano essere:

- `ltg` (log-trigliceridi): alta correlazione positiva
- `body_mass_index`: impatto marcato
- `mean_blood_pressure`: effetto moderato

Variabili come `sex` e `hdl_cholesterol` hanno impatto trascurabile.

---

## 📊 Visualizzazioni Chiave

- **Scatterplot** tra `y_test` e `y_pred`: punti vicini alla diagonale indicano un buon fit.
- **Distribuzione dei residui**: utile per verificare la natura casuale degli errori.

---

## 🚀 Considerazioni Finali

La regressione lineare fornisce una **baseline interpretabile**, ma non definitiva.  
Per migliorare l’accuratezza e la robustezza predittiva si possono esplorare:

- Modelli complessi: Random Forest, XGBoost, ecc.
- Regularizzazione: Lasso, Ridge
- Interazioni non lineari tra variabili
- Feature engineering

Il modello può inoltre essere **utilizzato in contesto clinico** per:

- Identificare pazienti a rischio elevato
- Supportare decisioni terapeutiche personalizzate
- Comunicare insights in un report per stakeholder

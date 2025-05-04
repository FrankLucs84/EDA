
## 📌 Nota tecnica: Asimmetria e Curtosi

Durante la fase di analisi univariata, per ogni variabile numerica vengono calcolati due indici descrittivi fondamentali:

---

### 📐 Asimmetria (Skewness)

L'**asimmetria** misura la simmetria della distribuzione rispetto alla media.

- `Skew ≈ 0`: distribuzione **simmetrica** (es. normale)
- `Skew > 0`: distribuzione **asimmetrica a destra** (coda lunga a destra)
- `Skew < 0`: distribuzione **asimmetrica a sinistra** (coda lunga a sinistra)

> Una forte asimmetria può suggerire la presenza di outlier o la necessità di trasformazioni (log, root).

---

### 🏔️ Curtosi (Kurtosis)

La **curtosi** misura la "pesantezza" delle code della distribuzione, ovvero la tendenza a generare valori estremi.

- `Kurt ≈ 0`: distribuzione **mesocurtica** (normale)
- `Kurt > 0`: distribuzione **leptocurtica** (code pesanti → outlier)
- `Kurt < 0`: distribuzione **platicurtica** (code leggere → pochi outlier)

> In Pandas, la funzione `.kurt()` calcola la **excess kurtosis**: una distribuzione normale ha curtosi = 0.

---

### 🧪 Perché sono utili?

- Capire **la forma della distribuzione**
- Diagnosticare **problemi di normalità**
- Identificare la **necessità di trasformazioni**
- Supportare il disegno del modello predittivo

Queste misure aiutano a **caratterizzare il comportamento statistico** delle variabili e guidano le decisioni successive nell'analisi.

# =========================
# FASE 1: IMPORTAZIONE
# =========================

# manipolazione dati
import pandas as pd
import numpy as np

# visualizzazione
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# applichiamo uno stile piacevole alla vista e settiamo i parametri di visualizzazione
plt.style.use("ggplot")
rcParams['figure.figsize'] = (12,  6)

from sklearn.datasets import load_diabetes

# Caricamento del dataset
diabetes = load_diabetes()

# Conversione in DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# creiamo la colonna per il target
df['target'] = diabetes.target

# =========================
# FASE 2: COMPRENSIONE
# =========================

# Prime 5 righe del dataset
print("\n[INFO] Prime righe del dataset:")
print(df.head())

# Ultime 5 righe del dataset
print("\n[INFO] Ultime righe del dataset:")
print(df.tail())

# Dimensioni del dataset
print("\n[INFO] Dimensioni (righe, colonne):", df.shape)

# Informazioni generali su tipi di dato e nulli
print("\n[INFO] Info generali sul dataset:")
df.info()

# Statistiche descrittive
print("\n[INFO] Statistiche descrittive delle variabili numeriche:")
print(df.describe())

# =========================
# FASE 3: PREPARAZIONE
# =========================

# Rilevamento di righe duplicate
n_duplicati = df.duplicated().sum()
print(f"\n[INFO] Righe duplicate presenti: {n_duplicati}")

# Nomi delle colonne originali
print("\n[INFO] Nomi originali delle colonne:")
print(df.columns.tolist())

# Rinomina delle colonne secondo documentazione
rename_map = {
    "age": "age",
    "sex": "sex",
    "bmi": "body_mass_index",
    "bp": "mean_blood_pressure",
    "s1": "tc_hdl_ratio",               # tc/hdl cholesterol ratio
    "s2": "ldl_cholesterol",            # low-density lipoprotein
    "s3": "hdl_cholesterol",            # high-density lipoprotein
    "s4": "tch",                        # total cholesterol?
    "s5": "ltg",                        # possibly log of triglycerides level
    "s6": "glu_serum_level",           # blood sugar level
    "target": "diabetes_progression"
}
df.rename(columns=rename_map, inplace=True)

# Verifica unicità dei nomi delle colonne
n_col_unique = df.columns.nunique()
n_col_total = len(df.columns)
print(f"\n[INFO] Colonne totali: {n_col_total}, Colonne uniche: {n_col_unique}")
if n_col_total != n_col_unique:
    print("[WARN] Attenzione: sono presenti colonne duplicate nei nomi!")
else:
    print("[OK] I nomi delle colonne sono univoci.")

# Output finale per verifica
print("\n[INFO] Colonne dopo la rinomina:")
print(df.columns.tolist())

print("\n[INFO] Stato finale del dataset pronto per l'analisi:")
print(df.head())


df['diabetes_progression'].describe()
print(df.describe(include='all'))

# ==========================
# FASE 4: COMPRENSIONE SULLE VARIABILI
# ==========================

# Analisi della variabile target
print("\n[INFO] Statistiche della variabile target:")
print(df['diabetes_progression'].describe())

plt.figure()
df['diabetes_progression'].hist(bins=30)
plt.title("Distribuzione della variabile target")
plt.xlabel("Progressione del diabete (dopo 1 anno)")
plt.ylabel("Frequenza")
plt.savefig(r"C:\Users\frank\Documents\EDA\grafici fase 4\target_distribution.png")
plt.close()

print(f"Curtosi: {df['diabetes_progression'].kurt():.2f}")
print(f"Asimmetria: {df['diabetes_progression'].skew():.2f}")

# Analisi univariata di tutte le feature numeriche
feature_cols = df.columns.drop('diabetes_progression')

for col in feature_cols:
    print(f"\n[INFO] Statistiche per la variabile: {col}")
    print(df[col].describe())

    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"Distribuzione di {col}")
    plt.xlabel(col)
    plt.ylabel("Frequenza")
    plt.savefig(fr"C:\Users\frank\Documents\EDA\grafici fase 4\{col}_distribution.png")
    plt.close()

    print(f"Curtosi: {df[col].kurt():.2f}")
    print(f"Asimmetria: {df[col].skew():.2f}")

# ==========================
# FASE 5: RELAZIONI TRA VARIABILI
# ==========================

# Pairplot
sns.pairplot(df)
plt.savefig(r"C:\Users\frank\Documents\EDA\grafici fase 5\pairplot.png")
plt.close()

# Scatterplot su alcune feature
for x_feature in ["body_mass_index", "ltg", "mean_blood_pressure"]:
    plt.figure()
    sns.scatterplot(x=x_feature, y="diabetes_progression", data=df)
    plt.title(f"{x_feature} vs Progressione diabete")
    plt.savefig(fr"C:\Users\frank\Documents\EDA\grafici fase 5\scatter_{x_feature}.png")
    plt.close()

# Heatmap di correlazione
corrmat = df.corr()
plt.figure()
sns.heatmap(corrmat, 
            cbar=True, 
            annot=True, 
            square=True, 
            fmt='.2f', 
            annot_kws={'size': 10}, 
            yticklabels=df.columns, 
            xticklabels=df.columns, 
            cmap="Spectral_r")
plt.title("Mappa di correlazione delle variabili")
plt.savefig(r"C:\Users\frank\Documents\EDA\grafici fase 5\heatmap_correlazioni.png")
plt.close()

# ================================
# FASI 6 e 7: Brainstorming & Output
# ================================

# Sintesi testuale delle principali correlazioni
print("\n[INSIGHT] Principali relazioni osservate:")
print("- BMI e ltg mostrano correlazione positiva con la progressione del diabete.")
print("- Pressione sanguigna mostra una correlazione moderata.")
print("- Alcune variabili (hdl_cholesterol, sex) mostrano scarsa correlazione con il target.")
print("- Possibili interazioni interessanti: BMI x ltg, age x mean_blood_pressure.")

# Lista di possibili azioni successive
print("\n[AZIONI POSSIBILI]")
azioni = [
    "1. Creare un report visuale per gli stakeholder.",
    "2. Iniziare la modellazione predittiva (regressione, alberi, ecc.).",
    "3. Continuare con analisi più approfondite (interazioni, outlier, clustering)."
]
for azione in azioni:
    print("-", azione)

# Output finale: dataset pulito disponibile come df
print("\n[INFO] Dataset finale pronto per il modeling o il reporting.")
print(df.head())

# ================================
# EXTRA Modellazione predittiva
# ================================
# Questo blocco di codice rappresenta l’ultima fase dell’intero processo di analisi predittiva, 
# dove mettiamo in pratica quanto appreso con l’EDA per costruire, testare e interpretare un modello di regressione lineare.

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Separazione X e y
X = df.drop(columns=["diabetes_progression"])
y = df["diabetes_progression"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modello lineare
model = LinearRegression()
model.fit(X_train, y_train)

# Predizioni
y_pred = model.predict(X_test)

# Valutazione & Visualizzazione (output finale)

# Metriche
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n[METRICHE]")
print(f"RMSE: {rmse:.2f}")
print(f"R² score: {r2:.3f}")

# Coefficienti
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficiente": model.coef_
}).sort_values(by="Coefficiente", key=abs, ascending=False)

print("\n[COEFFICIENTI DEL MODELLO]:")
print(coeff_df)

# Visualizzazione 1: grafico dei coefficienti del modello
plt.figure(figsize=(10, 6))
sns.barplot(data=coeff_df, x="Coefficiente", y="Feature", palette="crest")
plt.title("Importanza delle variabili (coefficienti della regressione)")
plt.xlabel("Coefficiente stimato")
plt.ylabel("Variabile")
plt.tight_layout()
plt.show()

# Visualizzazione 2: scatter reale vs predetto
plt.figure()
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Valori Reali")
plt.ylabel("Valori Predetti")
plt.title("Predizioni vs Reali")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

# Visualizzazione 3: distribuzione residui
residui = y_test - y_pred
plt.figure()
sns.histplot(residui, kde=True, bins=30)
plt.title("Distribuzione degli errori (residui)")
plt.xlabel("Errore")
plt.show()


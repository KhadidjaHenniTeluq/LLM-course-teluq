---
title: "Laboratoire"
weight: 6
---
 
Bonjour à nouveau ! J'espère que vous avez les yeux bien ouverts, car nous passons à la pratique. Ce laboratoire est le moment de vérité : nous allons transformer nos concepts abstraits de BERT et de vecteurs en outils concrets. 

> [!IMPORTANT]
🔑 **Je dois insister :** ne vous contentez pas de copier le code. Regardez comment chaque ligne influence le résultat. Nous allons voir comment un modèle "frozen" peut devenir un allié puissant pour classer des données en un temps record. Prêt·e·s ? Sortez vos notebooks Colab ! 

---

## 🔹 EXERCICE 1 : Classification de sentiments avec Pipeline

**Objectif** : Apprendre à charger et utiliser une pipeline Hugging Face sur le GPU T4 de Colab pour une classification immédiate.

```python
from transformers import pipeline
from datasets import load_dataset

# Chargement d'un échantillon de test de Rotten Tomatoes
test_data = load_dataset("rotten_tomatoes", split="test").select(range(3))

# --- VOTRE TÂCHE : Initialisez la pipeline et lancez l'inférence ---

```
<!-- TODO: add colab link -->

**Attentes** : Vous devez observer que le modèle identifie correctement le sentiment.

> [!WARNING]
⚠️ Notez bien que le score représente la confiance du modèle. Un score de 0.51 signifie que BERT est presque aussi confus que nous devant une critique sarcastique !

<details>
<summary><b>Voir la réponse</b></summary>

<!-- TODO: add solution colab link -->

```python
# --- RÉPONSE ---
# Initialisation de la pipeline avec un modèle RoBERTa spécialisé
pipe = pipeline("sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                device=0) # device=0 active le GPU T4 (Crucial pour la vitesse !)

# Boucle d'inférence
for text in test_data["text"]:
    # Appel de la pipeline sur le texte
    result = pipe(text)
    print(f"Texte: {text[:60]}...")
    print(f"Prédiction: {result[0]['label']} (Score: {result[0]['score']:.4f})\n")
```
</details>

---

## 🔹 EXERCICE 2 : Embeddings + Classifieur Scikit-Learn

**Objectif** : Implémenter la Stratégie 2 (Frozen Layers) en extrayant des caractéristiques avec `sentence-transformers` et en entraînant un modèle linéaire.

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import numpy as np

# Charger 200 exemples pour l'entraînement
dataset = load_dataset("rotten_tomatoes")
train_subset = dataset["train"].select(range(200))
model = SentenceTransformer("all-mpnet-base-v2")

# --- VOTRE TÂCHE : Extraire les embeddings et entraîner la LogisticRegression ---

```
<!-- TODO: add colab link -->

**Attentes** : Comprendre pourquoi l'entraînement est quasi instantané. 

> [!IMPORTANT]
🔑 **Je dois insister :** BERT a déjà fait tout le travail de compréhension du langage pendant son pré-entraînement ; notre classifieur ne fait que tracer une ligne entre les points.

<details>
<summary><b>Voir la réponse</b></summary>
<!-- TODO: add solution colab link -->

```python
# 1. Extraction des caractéristiques (On transforme le texte en vecteurs)
X_train = model.encode(train_subset["text"])
y_train = train_subset["label"]

# 2. Initialisation et entraînement du classifieur (Régression Logistique)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# 3. Vérification de la performance sur le train set
score = clf.score(X_train, y_train)
print(f"Précision sur l'échantillon d'entraînement : {score*100:.2f}%")

```

</details>

---

## 🔹 EXERCICE 3 : Logique de classification Zero-shot

**Objectif** : Implémenter manuellement la classification Zero-shot en utilisant la similarité cosinus entre un document et des étiquettes textuelles.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-mpnet-base-v2")
doc = "I am so annoyed by the constant software crashes!"
candidate_labels = ["technical issue", "billing question", "general praise"]

# --- VOTRE TÂCHE : Trouvez l'étiquette la plus proche via similarité cosinus ---

```
<!-- TODO: add colab link -->

**Attentes** : Expliquez l'impact du changement de label. 

> [!NOTE]
⚠️ Si vous remplacez "technical issue" par "computer problem", le score changera. Le Zero-shot est un art de la précision sémantique !

<details>
<summary><b>Voir la réponse</b></summary>
<!-- TODO: add solution colab link -->

```python
# --- RÉPONSE ---
# 1. Encodage du document et des étiquettes (labels)
doc_embedding = model.encode([doc])
label_embeddings = model.encode(candidate_labels)

# 2. Calcul de la similarité cosinus (produit des angles)
# Cela nous donne un score pour chaque label
similarities = cosine_similarity(doc_embedding, label_embeddings)

# 3. Identification de l'index du score le plus élevé (argmax)
best_index = np.argmax(similarities)

print(f"Document : '{doc}'")
print(f"Classe prédite (Zero-shot) : {candidate_labels[best_index]}")
print(f"Score de similarité : {similarities[0][best_index]:.4f}")

```

</details>
---

**Mots-clés de la semaine** : BERT, Encodeur, Bidirectionnel, MLM, Token [CLS], Fine-tuning, Représentation creuse vs dense, Score F1, Similarité Cosinus, Zero-shot.

**En prévision de la semaine suivante** : Nous allons changer de paradigme. Finie la compréhension pure, place à la création ! Nous explorerons les modèles **Decoder-only** (famille GPT) et l'art de murmurer à l'oreille des IA : le ***Prompt Engineering***.
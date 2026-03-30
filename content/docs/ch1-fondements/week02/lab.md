---
title: "Laboratoire"
weight: 6
---

Bonjour à toutes et à tous ! Prêts pour votre premier voyage au cœur de l'atome sémantique ? Ce laboratoire est un moment de vérité : nous allons manipuler les briques que nous avons étudiées en théorie. Rappelez-vous : dans le monde des LLM, un petit changement dans la tokenisation peut transformer une réponse géniale en une bouillie incompréhensible. Soyez méticuleux, soyez curieux, et surtout, observez bien comment les chiffres commencent à parler !

---

## 🔹 EXERCICE 1 : Comparaison de tokeniseurs

**Objectif** : Visualiser physiquement comment différents modèles découpent la même phrase.

**Description** : Utilisez la bibliothèque `transformers` pour charger BERT et GPT-2 et analysez leur comportement sur un texte complexe.

**Code (Testé pour Colab T4)** :
```python
from transformers import AutoTokenizer

# Chargement des tokeniseurs
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "L'intelligence artificielle est fascinante 🎵 #AI2024"

# Tâche : Tokenisez et affichez le résultat
print(f"BERT : {bert_tokenizer.tokenize(text)}")
print(f"GPT-2: {gpt2_tokenizer.tokenize(text)}")
```
<!-- TODO: add colab link -->

<details>
<summary><b>Voir la réponse</b></summary>
BERT affichera des `##` pour les sous-mots et risque de transformer l'emoji en `[UNK]` s'il n'est pas dans son dictionnaire. GPT-2 gérera l'emoji grâce à sa gestion des octets et ajoutera des symboles comme `Ġ` pour les espaces.
</details>

---

## 🔹 EXERCICE 2 : Création d'embeddings

**Objectif** : Transformer une pensée en un vecteur numérique et vérifier sa "forme".

**Description** : Utilisez `sentence-transformers` pour encoder une critique de film et analysez l'objet résultant.

**Code (Testé pour Colab T4)** :
```python
from sentence_transformers import SentenceTransformer

# Modèle recommandé
model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = "Ce cours sur les LLM est absolument incroyable !"
embedding = model.encode(sentence)

print(f"Dimension du vecteur : {embedding.shape}")
print(f"Les 5 premières valeurs : {embedding[:5]}")
```
<!-- TODO: add colab link -->

<details>
<summary><b>Voir la réponse</b></summary>
La dimension sera de **(384,)**. Les valeurs sont des nombres réels entre -1 et 1 environ.
</details>

---

## 🔹 EXERCICE 3 : Visualisation et Similarité

**Objectif** : Utiliser la réduction de dimension pour "voir" la proximité sémantique.

**Consigne** : Calculez la similarité cosinus entre trois phrases : deux proches et une éloignée. Utilisez ensuite PCA (Principal Component Analysis) pour projeter ces vecteurs en 2D.

**Code (Testé pour Colab T4)** :
```python
from sentence_transformers import util
import numpy as np
from sklearn.decomposition import PCA

sentences = [
    "J'adore les chats.",
    "Les félins sont mes animaux préférés.",
    "La bourse de Paris a clôturé en baisse."
]

embeddings = model.encode(sentences)

# 1. Calcul de similarité
sim = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarité Chat/Félin : {sim.item():.4f}")

# 2. Réduction de dimension (PCA)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
print(f"Coordonnées 2D :\n{reduced_embeddings}")
```
<!-- TODO: add colab link -->

<details>
<summary><b>Voir la réponse</b></summary>
La similarité entre les phrases 1 et 2 sera élevée (> 0.7), tandis qu'avec la phrase 3, elle sera faible (< 0.2). En 2D, les points 1 et 2 apparaîtront regroupés, loin du point 3.
</details>

---

**Mots-clés de la semaine** : Tokenisation, Sous-mots (Subwords), BPE, WordPiece, Embeddings denses, Espace vectoriel, Similarité Cosinus, [CLS]/[SEP], PCA.

**En prévision de la semaine suivante** : La semaine prochaine, nous monterons encore d'un cran : nous entrerons dans la salle des machines du Transformer pour voir exactement comment les têtes d'attention manipulent ces vecteurs.

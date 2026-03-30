---
title: "Laboratoire"
weight: 6
---

Bonjour à toutes et à tous ! Nous y sommes : le moment de vérité où les équations de la semaine se transforment en réalité numérique. Dans ce laboratoire, nous allons "ouvrir le capot" d'un Transformer pour voir ses pistons (l'attention) et ses engrenages (les blocs) en mouvement. 

> [!IMPORTANT]
‼️ **Je dois insister :** l'architecture que vous allez manipuler aujourd'hui est le socle de tout l'édifice des LLM. 

Ne vous contentez pas d'exécuter les cellules : observez comment la structure du modèle dicte sa capacité à comprendre. Prêt·e·s à explorer les entrailles de la machine ? C'est parti !  

---
## Exercice 1 : Visualisation de l'attention

**Objectif** : Utiliser un modèle BERT pour extraire les poids d'attention et comprendre comment un token "regarde" ses voisins.

```python
# --- QUESTION ---
from transformers import AutoModel, AutoTokenizer
import torch

# Chargement du modèle avec l'option de retour d'attention
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

sentence = "The cat sat on the mat"
inputs = tokenizer(sentence, return_tensors="pt")

# --- VOTRE TÂCHE : Récupérez les attentions et affichez la forme de la première couche ---
```
<!-- TODO: add colab link -->

{{< colab url="" label="Voir la solution sur Colab" >}}

<details>
<summary><b>Voir la réponse</b></summary>

<!-- TODO: add solution colab link -->

```python
outputs = model(**inputs)
# 'attentions' est un tuple de 12 tenseurs (un par couche)
attentions = outputs.attentions 

# Récupération de la première couche
first_layer_attention = attentions[0]

print(f"Forme de l'attention (Couche 1) : {first_layer_attention.shape}")
# Attendu : [1, 12, 8, 8] -> [Batch, Heads, Seq_len, Seq_len]
print("Succès : Le modèle a bien généré une matrice d'interaction pour les 12 têtes !")
```

</details>

---

## Exercice 2 : Analyse de structure interne

**Objectif** : Apprendre à lire l'architecture d'un modèle pour identifier le nombre de couches et la dimension cachée.

```python
# --- QUESTION ---
from transformers import AutoModelForCausalLM

# Utilisons un modèle léger pour l'analyse
model = AutoModelForCausalLM.from_pretrained("gpt2")

# --- VOTRE TÂCHE : Identifiez le nombre de couches et la dimension d'entrée (n_embd) ---
```
<!-- TODO: add colab link -->

{{< colab url="" label="Voir la solution sur Colab" >}}
<!-- TODO: add solution colab link -->

<details>
<summary><b>Voir la réponse</b></summary>

```python
print(model) # Affiche la structure complète

# Extraction via la configuration
n_layers = model.config.n_layer
embedding_dim = model.config.n_embd

print(f"\n--- RAPPORT D'ARCHITECTURE ---")
print(f"Nombre de blocs Transformer : {n_layers}")
print(f"Dimension des vecteurs (Model Dim) : {embedding_dim}")
```

</details>

---

## Exercice 3 : Mesure de l'impact du KV Cache

**Objectif** : Démontrer empiriquement l'accélération apportée par le caching des Keys et Values lors de la génération.

```python
# --- QUESTION ---
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
input_ids = tokenizer("Once upon a time in a galaxy far, far away", return_tensors="pt").input_ids

# --- VOTRE TÂCHE : Comparez le temps de génération de 20 tokens avec et sans cache ---
```
<!-- TODO: add colab link -->

{{< colab url="" label="Voir la solution sur Colab" >}}
<!-- TODO: add solution colab link -->

<details>
<summary><b>Voir la réponse</b></summary>

```python
# 1. Sans KV Cache
start = time.time()
output_no_cache = model.generate(input_ids, max_new_tokens=20, use_cache=False)
end_no_cache = time.time() - start

# 2. Avec KV Cache
start = time.time()
output_cache = model.generate(input_ids, max_new_tokens=20, use_cache=True)
end_cache = time.time() - start

print(f"Temps SANS cache : {end_no_cache:.4f}s")
print(f"Temps AVEC cache : {end_cache:.4f}s")
print(f"Facteur d'accélération : {end_no_cache/end_cache:.2f}x")
```

> [!WARNING]
⚠️ **Note :** Sur de très longues séquences, l'écart devient massif !


</details>

---

**Mots-clés de la semaine** : Self-Attention, Multi-head, Query/Key/Value, RoPE, KV Cache, FlashAttention, RMSNorm, Feedforward, LM Head.

**En prévision de la semaine suivante** : Nous allons utiliser ces connaissances pour explorer les modèles spécialisés dans la compréhension : la famille BERT (Encoder-only) et leurs applications en classification.

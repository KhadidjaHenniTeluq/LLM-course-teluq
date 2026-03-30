---
title: "Laboratoire"
weight: 6
---

Bonjour à toutes et à tous ! Nous y sommes : le moment de vérité où les équations de la semaine se transforment en réalité numérique. Dans ce laboratoire, nous allons "ouvrir le capot" d'un Transformer pour voir ses pistons (l'attention) et ses engrenages (les blocs) en mouvement. 

> [!IMPORTANT]
‼️ **Je dois insister :** l'architecture que vous allez manipuler aujourd'hui est le socle de tout l'édifice des LLM. 

Ne vous contentez pas d'exécuter les cellules : observez comment la structure du modèle dicte sa capacité à comprendre. Prêt·e·s à explorer les entrailles de la machine ? C'est parti !  

---
## 🔹 EXERCICE 1 : Visualisation de la structure et de l'attention

**Objectif** : Charger un modèle BERT-base et extraire ses poids d'attention pour comprendre la forme des données internes.

```python
from transformers import AutoModel, AutoTokenizer
import torch

# 1. INITIALISATION
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# On demande explicitement de sortir les attentions
model = AutoModel.from_pretrained(model_name, output_attentions=True)

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")
```

<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- RÉPONSE ---
# Exécution du modèle
with torch.no_grad():
    outputs = model(**inputs)

# Récupération des poids d'attention (tuple de 12 couches)
attentions = outputs.attentions 

# Analyse de la première couche
first_layer_attn = attentions[0]

print(f"Nombre de couches d'attention : {len(attentions)}")
print(f"Forme du tenseur d'attention (Couche 1) : {first_layer_attn.shape}")
# Attendu : [1, 12, 8, 8] -> [Batch, Heads, Tokens, Tokens]
```
**EXPLICATIONS DÉTAILLÉES**
*   **Résultats** : Vous voyez 12 couches et 12 têtes. La matrice 8x8 correspond aux interactions entre les 8 tokens du texte.
*   **Justification** : Chaque tête d'attention calcule sa propre matrice d'affinité. Si un mot regarde son voisin, la valeur à l'intersection dans cette matrice sera élevée.

</details>

---

## 🔹 EXERCICE 2 : Analyse de la configuration d'un bloc moderne

**Objectif** : Extraire les hyperparamètres d'un modèle Llama-like pour identifier les mécanismes d'optimisation (GQA, RMSNorm).

```python
from transformers import AutoConfig

# 1. CHARGEMENT DE LA CONFIG
# Utilisons un modèle compact et moderne
model_id = "microsoft/Phi-3-mini-4k-instruct"
```

<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- RÉPONSE ---
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

print(f"--- RAPPORT D'ARCHITECTURE PHI-3 ---")
print(f"Nombre de couches (Blocs Transformer) : {config.num_hidden_layers}")
print(f"Dimension cachée (d_model) : {config.hidden_size}")
print(f"Nombre de têtes de Query : {config.num_attention_heads}")

# Vérification du Grouped-Query Attention (GQA)
if hasattr(config, "num_key_value_heads"):
    print(f"Nombre de têtes de Key/Value : {config.num_key_value_heads}")
    ratio = config.num_attention_heads // config.num_key_value_heads
    print(f"Utilise le GQA avec un ratio de {ratio}:1 pour économiser la VRAM.")

```

**EXPLICATIONS DÉTAILLÉES**
*   **Justification** : Si le nombre de têtes K/V est inférieur aux têtes Query, le modèle utilise GQA.
*   Cela signifie qu'il est optimisé pour les longues conversations en réduisant la taille du KV Cache.

</details>

---

## 🔹 EXERCICE 3 : Profilage du KV Cache

**Objectif** : Mesurer l'impact de l'optimisation KV Cache sur le temps de génération d'un paragraphe.

```python
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. PRÉPARATION
model_name = "gpt2" # Modèle léger pour éviter les délais sur Colab
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

input_text = "The development of large language models has led to a major shift in how we"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
```

<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- RÉPONSE ---
# TEST 1 : GÉNÉRATION SANS CACHE
start_no_cache = time.time()
# On désactive le cache via use_cache=False
out_no_cache = model.generate(**inputs, max_new_tokens=40, use_cache=False)
end_no_cache = time.time() - start_no_cache

# TEST 2 : GÉNÉRATION AVEC CACHE
start_cache = time.time()
# Le cache est activé par défaut (use_cache=True)
out_cache = model.generate(**inputs, max_new_tokens=40, use_cache=True)
end_cache = time.time() - start_cache

print(f"Temps SANS cache : {end_no_cache:.4f} secondes")
print(f"Temps AVEC cache : {end_cache:.4f} secondes")
print(f"🚀 Gain de performance : {(end_no_cache / end_cache):.2f}x plus rapide !")
```

**EXPLICATIONS DÉTAILLÉES**
*   **Résultats** : Vous devriez observer un gain significatif (souvent 2x ou plus).
*   **Justification** : Sans cache, le modèle doit recalculer l'attention pour TOUTE la phrase à chaque nouvelle lettre. Avec le cache, il ne calcule que pour le dernier mot et "lit" le reste en mémoire. 

> [!IMPORTANT]
🔔 **Note éthique** : Moins de calcul signifie aussi une consommation électrique réduite !

</details>

---

**Mots-clés de la semaine** : Self-Attention, Query/Key/Value, Multi-head, RoPE (Positional), RMSNorm, Residual Connections, GQA, FlashAttention, KV Cache, Forward Pass.

**En prévision de la semaine prochaine** : Nous allons utiliser ces connaissances pour explorer les modèles spécialisés dans la compréhension pure : les modèles **Encoder-only** (la famille BERT) et leurs applications en classification.
---
title: "Laboratoire"
weight: 6
---

Bonjour à toutes et à tous ! Nous passons maintenant aux travaux pratiques de cette semaine "visionnaire". Dans ce laboratoire, vous allez apprendre à manipuler l'espace vectoriel où les mots et les images se rencontrent enfin. 

> [!IMPORTANT]
🔑 **Je dois insister :** la multimodalité demande une rigueur particulière sur la gestion de la mémoire. Nous allons utiliser des modèles de pointe comme CLIP et BLIP-2 sur notre GPU T4. Ne vous contentez pas de générer des légendes : essayez de comprendre comment le modèle "traduit" les pixels en concepts. Prêt·e·s à ouvrir les yeux de vos algorithmes ? C'est parti !

---
### 🔹 EXERCICE 1 : CLIP pour la recherche d'images (Niveau 1)

**Objectif** : Utiliser CLIP pour identifier quelle description textuelle correspond le mieux à une image donnée.

```python
# --- CODE (QUESTION) ---
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# 1. Chargement
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image de 2 chats
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# TÂCHE : Calculez la similarité entre l'image et ces 3 textes.
texts = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

```
<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- RÉPONSE (CORRIGÉ) ---
# 2. Prétraitement simultané
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 3. Inférence (Calcul des scores)
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image 
    probs = logits_per_image.softmax(dim=1) # Normalisation

# 4. Résultat
for i, text in enumerate(texts):
    print(f"Probabilité '{text}': {probs[0][i].item():.4f}")
```
**EXPLICATIONS DÉTAILLÉES** :
*   **ATTENDU** : "a photo of a cat" doit avoir un score proche de 1.0.
*   **JUSTIFICATION** : CLIP a appris à projeter ces deux modalités dans le même espace. La similarité cosinus (logits) est maximale pour la paire cohérente.

</details>

---
### 🔹 EXERCICE 2 : BLIP-2 Captioning (Niveau 2)

**Objectif** : Générer une description automatique pour une image en utilisant BLIP-2 et la quantification pour économiser la VRAM.

```python
# --- CODE (QUESTION) ---
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# TÂCHE : Chargez BLIP-2 en float16 et générez une légende pour l'image de l'exercice 1.
```

<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (CORRIGÉ) ---
# 1. Chargement optimisé pour Colab T4
model_id = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# 2. Préparation
inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)

# 3. Génération de la légende
generated_ids = model.generate(**inputs, max_new_tokens=20)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(f"Légende générée : {caption}")
```
**EXPLICATIONS DÉTAILLÉES** : 
*   **ATTENDU** : Une phrase type "two cats sleeping on a couch".
*   **JUSTIFICATION** : Le Q-Former a extrait les concepts "cat", "sleep" et "couch" et les a injectés comme contexte dans le décodeur OPT du LLM.

</details>

---
### 🔹 EXERCICE 3 : VQA (Visual Question Answering) complexe (Niveau 3)

**Objectif** : Poser une question nécessitant un raisonnement sur l'image à BLIP-2.

```python
# --- CODE (QUESTION) ---
# TÂCHE : Demandez au modèle combien de chats sont présents sur l'image.
```

<details>
<summary>Voir la réponse</summary>

```python
# --- RÉPONSE (CORRIGÉ) ---
# 1. Construction du prompt de question
question = "Question: How many cats are there in this picture? Answer:"

# 2. Prétraitement avec texte ET image
inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)

# 3. Génération raisonnée
out = model.generate(**inputs, max_new_tokens=10)
answer = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

print(f"Question : {question}")
print(f"IA : {answer}")
```
**EXPLICATIONS DÉTAILLÉES** : 
*   **ATTENDU** : "two" ou "2".
*   **JUSTIFICATION** : Le modèle doit non seulement identifier les objets mais aussi compter, ce qui prouve l'intégration réussie des capacités logiques du LLM avec la perception du ViT.

</details>

---
**Mots-clés de la semaine** : Multimodal, CLIP, ViT, Patches, Q-Former, BLIP-2, Alignement, VQA, Image Captioning, Espace Vectoriel Commun.

**En prévision de la semaine suivante** : Nous allons apprendre à adapter ces géants à vos besoins spécifiques. Comment ré-entraîner un modèle de 7 milliards de paramètres sur un simple PC ? Bienvenue dans le monde du **Fine-tuning supervisé** et de la méthode **LoRA**.
---
title: "10.2 Encodage visuel : Le Vision Transformer (ViT)"
weight: 3
---

## Quand l'IA apprend à regarder : L'image est une séquence
Bonjour à toutes et à tous ! Nous avons vu dans la section précédente (10.1) comment le texte et l'image peuvent cohabiter dans une même galaxie mathématique. Mais une question brûlante doit vous brûler les lèvres : comment l'IA fait-elle pour "lire" une image ? Un fichier JPG n'est qu'une grille de millions de chiffres (les pixels). 

> [!IMPORTANT]
🔑 **Je dois insister :** pendant des décennies, nous avons utilisé des outils appelés CNN (Réseaux de Neurones Convolutifs) qui regardaient l'image par de petites fenêtres locales. Mais en 2020, tout a basculé. Des chercheurs ont posé une question folle : "Et si nous traitions une image exactement comme une phrase ?". 

Bienvenue dans le monde du **Vision Transformer (ViT)**. Respirez, nous allons apprendre à découper la réalité en jetons visuels.

---
## La fin du règne des Convolutions (CNN)
Pendant longtemps, le standard était le CNN. Il fonctionnait comme un détective avec une loupe, balayant l'image pixel par pixel pour trouver des bords, puis des formes, puis des objets. C'était efficace, mais cela souffrait d'une limite : le modèle avait du mal à comprendre les relations à très longue distance dans l'image (par exemple, le lien entre un nuage en haut à gauche et son reflet dans une flaque en bas à droite). 

Le **Vision Transformer**, introduit par l'article "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020), a balayé cette approche en appliquant l'architecture que vous connaissez déjà par cœur : le Transformer (Semaine 3). 

Comme l'illustre la **Figure 10-8 : Vision Transformer vs Transformer** , le concept est d'une élégance rare : le Transformer ne sait pas qu'il traite une image. Pour lui, tout est une séquence. Si vous lui donnez une suite de vecteurs, qu'ils représentent des mots ou des morceaux de paysage, il appliquera la même mathématique de l'attention. 

{{< bookfig src="199.png" week="10" >}}

---
## L'art du "Patching" : Transformer les pixels en mots
> [!IMPORTANT]
🔑 **C'est le concept le plus important de cette section :** Un LLM traite des mots. Un ViT traite des **Patches** (morceaux). Comme nous ne pouvons pas donner chaque pixel individuellement au modèle (ce serait beaucoup trop lourd pour le calcul d'attention), nous devons découper l'image.

Regardez attentivement la **Figure 10-9 : Le processus de "tokenisation" d'image**. Imaginez que vous déchirez une photo en petits carrés de 16x16 pixels. 
*   Si votre image fait 224x224 pixels, vous obtenez une grille de 14x14 carrés, soit 196 "mots visuels".
*   Ces carrés sont ensuite "aplatis" (flattened). On transforme un petit bloc 2D en une longue ligne de chiffres.

{{< bookfig src="201.png" week="10" >}}

*   C'est ce que montre la **Figure 10-10 : Texte passé aux encodeurs** par analogie : tout comme nous codons "What a horrible movie!" en tokens, nous codons notre paysage en une séquence de patches.

{{< bookfig src="200.png" week="10" >}}

---
## La Projection Linéaire : Donner une voix aux patches
Une fois que nous avons nos petits carrés aplatis, ils n'ont pas encore la "bonne forme" pour entrer dans le Transformer. Un Transformer attend des vecteurs d'une taille précise (par exemple, 768 dimensions pour ViT-base). 

On utilise alors une **Projection Linéaire**. 

> [!TIP]
🔑 **Mon intuition :** C'est comme si vous passiez chaque morceau de photo à travers un filtre qui le transforme en une "description mathématique" compacte. Cette étape transforme les données brutes des pixels en un **Patch Embedding**. C'est le jumeau visuel du Word Embedding que nous avons vu en Semaine 2. 

---
## Recomposer le puzzle : Encodage positionnel et Class Token

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Si vous donnez simplement les patches au modèle, il saura ce qu'il y a dedans (ex: "un bout d'oreille", "un bout de ciel"), mais il ne saura pas *où* ils se trouvent. Pour lui, une photo de chat et un puzzle mélangé de la même photo seraient identiques.

Pour résoudre cela, le ViT utilise deux astuces cruciales détaillées dans la **Figure 10-11 : L'algorithme principal de ViT** :

{{< bookfig src="202.png" week="10" >}}

1.  **Positional Embeddings** : On ajoute à chaque patch un vecteur qui indique sa position (ex: "Je suis le patch n°1, en haut à gauche"). Sans cela, le mécanisme d'attention est spatialement aveugle.
2.  **Le [CLS] Token (Class Token)** : Comme pour BERT (Semaine 4), on ajoute un token spécial au début de la séquence. Ce token n'est pas un morceau d'image. Son rôle est de "voyager" à travers toutes les couches d'attention pour récolter les informations de tous les autres patches. À la fin, c'est le vecteur de ce token `[CLS]` qui servira à dire : "C'est un chat !". 

---
## ViT vs CNN : Le choc des cultures
Pourquoi préférer ViT à un CNN classique ? 

| Caractéristique | CNN (Convolutif) | ViT (Transformer) |
| :--- | :--- | :--- |
| **Vision** | Locale (filtre par filtre) | Globale (chaque patch voit tous les autres) |
| **Biais Inductif** | Fort (suppose que les pixels voisins sont liés) | Faible (doit tout apprendre de zéro) |
| **Besoin de données** | Modéré | Massif (excellent sur d'immenses jeux de données) |
| **Scalabilité** | Difficile à très grande échelle | Excellente (plus on lui donne de données, meilleur il est) |

> [!IMPORTANT]
🔑 **Je dois insister :** Le ViT est "plus bête" au début qu'un CNN parce qu'il n'a aucune notion innée de l'espace. Mais une fois entraîné sur des millions d'images, il surpasse les CNN car sa vision globale lui permet de comprendre des contextes beaucoup plus complexes.

---
## Laboratoire de code : Utiliser un Vision Transformer
Voyons comment charger un ViT pré-entraîné par Google et l'utiliser pour analyser une image. Nous allons utiliser `ViTImageProcessor` pour le découpage en patches et `ViTModel` pour l'extraction de sens.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers pillow requests torch

from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch

# 1. CHARGEMENT DU MODÈLE ET DU PROCESSEUR
# patch16 signifie des carrés de 16x16. 224 est la taille de l'image.
model_id = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_id)
model = ViTModel.from_pretrained(model_id).to("cuda")

# 2. CHARGEMENT D'UNE IMAGE
url = "http://images.cocodataset.org/val2017/000000039769.jpg" # Deux chats
image = Image.open(requests.get(url, stream=True).raw)

# 3. PRÉTRAITEMENT (PATCHING & NORMALISATION)
# Le processeur redimensionne et découpe l'image automatiquement
inputs = processor(images=image, return_tensors="pt").to("cuda")

# 4. INFÉRENCE (FORWARD PASS)
with torch.no_grad():
    outputs = model(**inputs)

# 5. ANALYSE DES SORTIES
last_hidden_states = outputs.last_hidden_state
print(f"Forme de la sortie : {last_hidden_states.shape}")
# Résultat attendu : [1, 197, 768]
# Pourquoi 197 ? 196 patches (14x14) + 1 [CLS] token !

# Extraction du vecteur de compréhension globale ([CLS])
global_representation = last_hidden_states[:, 0, :]
print(f"Dimension du 'résumé visuel' : {global_representation.shape}")
```

⚠️ Notez bien la forme de la sortie `[1, 197, 768]`. 
> [!IMPORTANT]
🔑 **C'est non-négociable :** vous devez comprendre que pour l'IA, cette image est devenue une "phrase" de 197 mots, où chaque mot a une définition de 768 nombres. C'est ainsi que la vision fusionne avec le langage.

---
## Applications industrielles du ViT
Le ViT n'est pas seulement un objet d'étude, c'est le moteur de nombreuses technologies actuelles :
*   **Imagerie Médicale** : Détecter des anomalies dans des IRM en analysant les relations subtiles entre différents tissus (attention globale).
*   **Voitures Autonomes** : Comprendre la scène complète (route, piétons, panneaux) d'un seul bloc plutôt que de chercher des objets isolés.
*   **Recherche par Image** : Comme nous l'avons vu avec CLIP, le ViT est l'encodeur qui permet de trouver des photos "similaires" à un concept abstrait.

---
## Éthique et Transparence : La cécité des patches

> [!CAUTION]
⚠️ Mes chers étudiants, le regard de la machine est fragmenté. 

Le ViT a des failles que vous devez connaître en tant qu'ingénieurs responsables :
1.  **Sensibilité à la résolution** : Si un objet important est plus petit qu'un patch (16x16 pixels), le modèle risque de l'ignorer totalement ou de le confondre avec du bruit de fond. 
2.  **Attaques adverses** : En changeant quelques pixels de manière invisible pour l'œil humain, on peut tromper l'attention du Transformer et lui faire "voir" un avion à la place d'un chien. 
3.  **Biais de données** : Si le ViT n'a vu que des paysages urbains durant son entraînement, il sera "aveugle" aux subtilités d'une forêt tropicale ou d'un désert. Sa perception est limitée par son éducation numérique.

> [!TIP]
🔑 **Mon message** : Le Vision Transformer nous apprend une leçon d'humilité : la vision n'est pas une vérité magique, c'est une reconstruction statistique. 

> En découpant le monde en patches, nous permettons à la machine de l'analyser, mais nous acceptons aussi qu'elle puisse rater l'essentiel si celui-ci se cache dans les détails. Soyez toujours les yeux critiques derrière les algorithmes.

---
Vous savez maintenant comment la machine transforme la lumière en nombres. Vous avez compris le mécanisme du ViT. Mais savoir "voir" n'est que la moitié du chemin. Pour qu'une IA soit vraiment multimodale, elle doit pouvoir *parler* de ce qu'elle voit. Dans la prochaine section ➡️, nous allons étudier **BLIP-2**, l'architecture qui utilise un "traducteur" génial pour permettre à un LLM de discuter avec un Vision Transformer.

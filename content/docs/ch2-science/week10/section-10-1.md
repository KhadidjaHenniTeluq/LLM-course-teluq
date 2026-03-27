---
title: "10.1 Alignement vision-langage "
weight: 2
---

{{< katex />}}

## La fin de la prison textuelle : Pourquoi la multimodalité ?
Imaginez que vous essayiez d'expliquer ce qu'est un "coucher de soleil" à quelqu'un qui n'a jamais vu la lumière. Vous pouvez utiliser des milliers de mots, mais vous n'atteindrez jamais la richesse d'une seule image. Pendant des décennies, le Traitement du Langage Naturel (NLP) et la Vision par Ordinateur (Computer Vision) étaient deux mondes totalement isolés. Les linguistes travaillaient sur des séquences de caractères, les spécialistes de l'image sur des grilles de pixels. 

C'est ce que nous montre la **Figure 10-1 : Modèles multimodaux** . 

{{< bookfig src="198.png" week="10" >}}

Cette illustration fondamentale nous présente l'IA moderne comme un carrefour. À gauche, nous voyons les entrées (Input modality) : du texte, du code, des images, et même de l'audio. À droite, les sorties possibles. 

> [!NOTE]
🔑 **Je dois insister sur ce point :** un modèle multimodal est capable de créer des ponts sémantiques entre ces mondes. Il comprend que le mot "chat" écrit en ASCII et la photo d'un petit félin roux sont deux représentations du même concept universel.

---
## Le défi de l'alignement : La Pierre de Rosette sémantique
Le problème mathématique est colossal. Un mot est une unité discrète (un token). Une image est une structure continue de millions de pixels organisés en trois canaux (Rouge, Vert, Bleu). Comment faire pour que la machine comprenne que le vecteur du mot "Montagne" doit être "proche" géométriquement d'un amas de pixels gris et blancs représentant l'Everest ?

Nous avons besoin d'un **alignement**. Pour y parvenir, les chercheurs ont créé une sorte de "Pierre de Rosette" mathématique : l'**Espace Vectoriel Commun**. Comme l'illustrent les **Figures 10-2 et 10-3 : Embeddings multimodaux** , l'idée est de projeter les deux mondes dans la même galaxie mathématique. 
*   **La Figure 10-2** montre le processus de création : on passe l'image dans un encodeur visuel et le texte dans un encodeur textuel, puis on force les deux résultats à converger. 

{{< bookfig src="203.png" week="10" >}}

*   **La Figure 10-3** nous montre le résultat final : dans cet espace, le point "Photo de chat" et le point "Texte : Un chat mignon" finissent par se toucher, tandis que le point "Photo de voiture" est projeté très loin d'eux. 

{{< bookfig src="204.png" week="10" >}}

> [!NOTE]
🔑 **Notez bien cette intuition :** l'alignement transforme la perception en une question de distance.

---
## La révolution CLIP (Contrastive Language-Image Pretraining)
En 2021, OpenAI a publié un modèle qui a tout changé : **CLIP**. Son nom contient la clé de son succès : l'apprentissage **contrastif**. Plutôt que d'essayer de "décrire" chaque image (ce qui est très difficile à annoter pour des millions de photos), on va apprendre au modèle à faire des comparaisons.

Regardons ensemble la **Figure 10-4 : Le type de données pour CLIP** . Pour entraîner CLIP, on ne lui donne pas des labels simples comme "Chien" ou "Chat". On lui donne des paires image/légende provenant de tout le web (400 millions de paires !). Par exemple : une photo de paysage avec la légende "Un superbe coucher de soleil sur la mer".

{{< bookfig src="205.png" week="10" >}}

L'architecture, détaillée dans les **Figures 10-5 à 10-7** , est un chef-d'œuvre de simplicité :
1.  **Le double encodeur (Figure 10-5)** : CLIP possède deux cerveaux. Un **Text Encoder** (un Transformer classique) et un **Image Encoder** (souvent un Vision Transformer, que nous verrons en 10.2).

{{< bookfig src="206.png" week="10" >}}

2.  **La matrice de similarité (Figure 10-6)** : Pendant l'entraînement, on présente au modèle un lot (batch) de $N$ images et $N$ légendes. Cela crée une grille de $N \times N$ combinaisons possibles.

{{< bookfig src="207.png" week="10" >}}

3.  **L'objectif contrastif (Figure 10-7)** : On demande au modèle de maximiser la similarité (le produit scalaire) pour les $N$ paires qui vont ensemble (la diagonale de la matrice) et de minimiser la similarité pour toutes les autres combinaisons fausses ($N^2 - N$). 

{{< bookfig src="208.png" week="10" >}}

> [!TIP]
🔑 **Mon analogie :** C'est comme un immense jeu de Memory. Le modèle doit apprendre à associer chaque photo à son étiquette correcte parmi des milliers d'intrus. À force de jouer, il finit par comprendre ce qu'est un "coucher de soleil", car c'est le seul concept qui lie statistiquement toutes les photos de ciel orangé à leurs légendes. 

---
## Le pouvoir du "Zero-shot" visuel
C'est ici que CLIP devient proprement magique. Comme il a appris à aligner des concepts textuels globaux avec des images, il est capable de classer des images qu'il n'a jamais vues dans des catégories qu'il n'a jamais apprises spécifiquement.

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Dans l'ancien monde (avant CLIP), si vous vouliez reconnaître des "Chiens de prairie", vous deviez collecter 1000 photos de chiens de prairie et ré-entraîner votre modèle.

Avec CLIP, vous n'avez qu'à lui dire : "Parmi ces trois étiquettes : 'Un chien de prairie', 'Une forêt', 'Une voiture', laquelle ressemble le plus à cette photo ?". CLIP va calculer la similarité entre l'embedding de l'image et les embeddings des trois phrases. L'étiquette avec le score le plus haut l'emporte. 

> [!NOTE]
🔑 **Je dois insister :** c'est la naissance de la classification visuelle par le langage.

---
## Applications pratiques : Un nouveau monde de possibilités
L'alignement vision-langage n'est pas qu'une prouesse théorique. Il alimente des outils que vous utilisez peut-être déjà :
*   **Recherche d'images par le sens** : Tapez "tristesse" dans votre application de photos, et elle trouvera les photos de pluie ou de visages sombres grâce à CLIP, même sans mots-clés "métadonnées".
*   **Modération de contenu** : Identifier des images dangereuses en décrivant simplement ce qu'il faut interdire en langage naturel.
*   **Aide aux malvoyants** : Décrire une scène en temps réel en identifiant les objets et leurs relations.
*   **Génération d'images (DALL-E / Stable Diffusion)** : CLIP est le cerveau qui guide ces modèles. Quand vous tapez "Un astronaute à cheval", c'est CLIP qui vérifie que l'image générée ressemble bien à ce que les mots décrivent.

---
## Laboratoire de code : Inférence CLIP sur Colab
Voyons comment manipuler cet espace commun. Nous allons utiliser la bibliothèque `transformers` pour charger CLIP et calculer la similarité entre une image et plusieurs descriptions.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers pillow requests

from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests
import torch

# 1. CHARGEMENT DU MODÈLE ET DU PROCESSEUR
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to("cuda")
processor = CLIPProcessor.from_pretrained(model_id)

# 2. PRÉPARATION DE L'IMAGE (Un chiot dans la neige)
url = "./images/puppy.png"
image = Image.open(requests.get(url, stream=True).raw)

# 3. NOS ÉTIQUETTES (Texte à aligner)
candidate_labels = ["a puppy playing in the snow", "a cat on a sofa", "a red car", "a sunny beach"]

# 4. PRÉTRAITEMENT ET ALIGNEMENT
# Le processeur transforme à la fois l'image en tenseurs et le texte en tokens
inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True).to("cuda")

# 5. INFÉRENCE
with torch.no_grad():
    outputs = model(**inputs)

# Récupération des scores de similarité (Logits)
# CLIP calcule le produit scalaire entre les embeddings
logits_per_image = outputs.logits_per_image 
probs = logits_per_image.softmax(dim=1) # Normalisation en probabilités

# 6. AFFICHAGE DES RÉSULTATS
for i, label in enumerate(candidate_labels):
    print(f"Probabilité pour '{label}': {probs[0][i].item():.4f}")

```

> [!IMPORTANT]
⚠️ Observez les probabilités. Vous verrez que "a puppy playing in the snow" obtient un score écrasant (souvent > 99%). Pourquoi ? Parce que CLIP a identifié non seulement l'objet ("puppy") mais aussi le contexte ("snow"). C'est la force de l'alignement global.

---
## Éthique et Transparence : Le miroir des préjugés visuels

> [!CAUTION]
⚠️ Mes chers étudiants, soyez extrêmement vigilants. 

CLIP a été entraîné sur Internet, un miroir souvent déformant de la réalité. 
1.  **Biais de représentation** : Si CLIP a vu plus d'images d'hommes en costume associés au mot "Directeur", il aura un score de similarité plus faible pour une femme dans le même rôle. C'est le biais de corrélation visuelle. 
2.  **Toxicité cachée** : Certains mots peuvent "activer" des associations d'images inappropriées ou stéréotypées. 
3.  **L'illusion de compréhension** : CLIP ne "comprend" pas ce qu'est un chien. Il comprend que les motifs de pixels d'un chien coïncident statistiquement avec le token "chien". 

> [!TIP]
🔑 **Mon conseil** : Ne déployez jamais un système de classification visuelle basé sur CLIP pour des décisions humaines (sécurité, recrutement, surveillance) sans avoir audité ses biais sur des populations diverses. L'IA voit le monde à travers nos propres préjugés numériques.

---
## Synthèse : La boussole de la multimodalité
Pour clore cette section, retenez cette structure :

| Composant | Rôle | Analogie |
| :--- | :--- | :--- |
| **Image Encoder** | Traduit les pixels en vecteurs | Le regard |
| **Text Encoder** | Traduit les tokens en vecteurs | L'écoute |
| **Contrastive Loss** | Force les vecteurs liés à se rapprocher | L'apprentissage |
| **Espace Commun** | Lieu où le texte et l'image cohabitent | La compréhension |

---
Vous maîtrisez maintenant le concept de l'alignement. Vous savez comment faire pour que deux mondes que tout oppose — le texte et l'image — se rejoignent dans une même langue mathématique. Mais comment l'encodeur visuel fait-il pour "lire" les pixels ? C'est ce que nous allons découvrir dans la prochaine section ➡️ en étudiant le **Vision Transformer (ViT)**, l'algorithme qui a appris à traiter les images comme des paragraphes.
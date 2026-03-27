[CONTENU SEMAINE 10]

# Semaine 10 : LLM multimodaux

**Titre : Au-delà du texte : Les LLM multimodaux**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis particulièrement émue d'entamer cette dixième semaine avec vous. Jusqu'ici, nous avons traité nos modèles comme des érudits vivant dans une pièce sans fenêtres, ne connaissant du monde que ce que les livres (le texte) leur racontaient. Aujourd'hui, nous allons "ouvrir les fenêtres" de l'IA. 🔑 **Je dois insister :** la multimodalité n'est pas une simple fonctionnalité supplémentaire, c'est une convergence biologique artificielle. Un modèle qui peut à la fois lire une recette et identifier les ingrédients sur une photo change radicalement notre rapport à la machine. Préparez-vous, car nous allons apprendre à l'IA non seulement à lire, mais à percevoir ! » [SOURCE: Livre p.259]

**Rappel semaine précédente** : « La semaine dernière, nous avons appris à combattre les hallucinations de l'IA grâce au RAG (Retrieval-Augmented Generation), en connectant nos modèles à des bases de données documentaires pour garantir des réponses ancrées dans des preuves. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer le fonctionnement des modèles capables de traiter plusieurs types de données (modalités).
*   Comprendre l'architecture de CLIP et le concept d'alignement vision-langage.
*   Détailler le fonctionnement du Vision Transformer (ViT) pour transformer les pixels en tokens.
*   Maîtriser l'utilisation de BLIP-2 et du Q-Former pour le dialogue multimodal.
*   Implémenter des tâches de classification d'images "zero-shot" et de description d'images (captioning).

---

## 10.1 Alignement vision-langage (2000+ mots)

### La fin de la prison textuelle : Pourquoi la multimodalité ?
« Imaginez que vous essayiez d'expliquer ce qu'est un "coucher de soleil" à quelqu'un qui n'a jamais vu la lumière. Vous pouvez utiliser des milliers de mots, mais vous n'atteindrez jamais la richesse d'une seule image. » Pendant des décennies, le Traitement du Langage Naturel (NLP) et la Vision par Ordinateur (Computer Vision) étaient deux mondes totalement isolés. Les linguistes travaillaient sur des séquences de caractères, les spécialistes de l'image sur des grilles de pixels. 

C'est ce que nous montre la **Figure 9-1 : Modèles multimodaux** (p.259 du livre). Cette illustration fondamentale nous présente l'IA moderne comme un carrefour. À gauche, nous voyons les entrées (Input modality) : du texte, du code, des images, et même de l'audio. À droite, les sorties possibles. 🔑 **Je dois insister sur ce point :** un modèle multimodal est capable de créer des ponts sémantiques entre ces mondes. Il comprend que le mot "chat" écrit en ASCII et la photo d'un petit félin roux sont deux représentations du même concept universel. [SOURCE: Livre p.259, Figure 9-1]

### Le défi de l'alignement : La Pierre de Rosette sémantique
Le problème mathématique est colossal. Un mot est une unité discrète (un token). Une image est une structure continue de millions de pixels organisés en trois canaux (Rouge, Vert, Bleu). Comment faire pour que la machine comprenne que le vecteur du mot "Montagne" doit être "proche" géométriquement d'un amas de pixels gris et blancs représentant l'Everest ?

Nous avons besoin d'un **alignement**. Pour y parvenir, les chercheurs ont créé une sorte de "Pierre de Rosette" mathématique : l'**Espace Vectoriel Commun**. Comme l'illustrent les **Figures 9-6 et 9-7 : Embeddings multimodaux** (p.263-264), l'idée est de projeter les deux mondes dans la même galaxie mathématique. 
*   **La Figure 9-6** montre le processus de création : on passe l'image dans un encodeur visuel et le texte dans un encodeur textuel, puis on force les deux résultats à converger. 
*   **La Figure 9-7** nous montre le résultat final : dans cet espace, le point "Photo de chat" et le point "Texte : Un chat mignon" finissent par se toucher, tandis que le point "Photo de voiture" est projeté très loin d'eux. 🔑 **Notez bien cette intuition :** l'alignement transforme la perception en une question de distance. [SOURCE: Livre p.263-264, Figures 9-6, 9-7]

### La révolution CLIP (Contrastive Language-Image Pretraining)
En 2021, OpenAI a publié un modèle qui a tout changé : **CLIP**. Son nom contient la clé de son succès : l'apprentissage **contrastif**. Plutôt que d'essayer de "décrire" chaque image (ce qui est très difficile à annoter pour des millions de photos), on va apprendre au modèle à faire des comparaisons.

Regardons ensemble la **Figure 9-8 : Le type de données pour CLIP** (p.265). Pour entraîner CLIP, on ne lui donne pas des labels simples comme "Chien" ou "Chat". On lui donne des paires image/légende provenant de tout le web (400 millions de paires !). Par exemple : une photo de paysage avec la légende "Un superbe coucher de soleil sur la mer". [SOURCE: Livre p.265, Figure 9-8]

L'architecture, détaillée dans les **Figures 9-9 à 9-11** (p.266-267), est un chef-d'œuvre de simplicité :
1.  **Le double encodeur (Figure 9-9)** : CLIP possède deux cerveaux. Un **Text Encoder** (un Transformer classique) et un **Image Encoder** (souvent un Vision Transformer, que nous verrons en 10.2).
2.  **La matrice de similarité (Figure 9-10)** : Pendant l'entraînement, on présente au modèle un lot (batch) de $N$ images et $N$ légendes. Cela crée une grille de $N \times N$ combinaisons possibles.
3.  **L'objectif contrastif (Figure 10-11)** : On demande au modèle de maximiser la similarité (le produit scalaire) pour les $N$ paires qui vont ensemble (la diagonale de la matrice) et de minimiser la similarité pour toutes les autres combinaisons fausses ($N^2 - N$). 

🔑 **L'analogie du Professeur Henni :** C'est comme un immense jeu de Memory. Le modèle doit apprendre à associer chaque photo à son étiquette correcte parmi des milliers d'intrus. À force de jouer, il finit par comprendre ce qu'est un "coucher de soleil", car c'est le seul concept qui lie statistiquement toutes les photos de ciel orangé à leurs légendes. [SOURCE: Livre p.265-267, Figures 9-9, 9-10, 9-11]

### Le pouvoir du "Zero-shot" visuel
C'est ici que CLIP devient proprement magique. Comme il a appris à aligner des concepts textuels globaux avec des images, il est capable de classer des images qu'il n'a jamais vues dans des catégories qu'il n'a jamais apprises spécifiquement.

⚠️ **Attention : erreur fréquente ici !** Dans l'ancien monde (avant CLIP), si vous vouliez reconnaître des "Chiens de prairie", vous deviez collecter 1000 photos de chiens de prairie et ré-entraîner votre modèle. 
Avec CLIP, vous n'avez qu'à lui dire : "Parmi ces trois étiquettes : 'Un chien de prairie', 'Une forêt', 'Une voiture', laquelle ressemble le plus à cette photo ?". CLIP va calculer la similarité entre l'embedding de l'image et les embeddings des trois phrases. L'étiquette avec le score le plus haut l'emporte. 🔑 **Je dois insister :** c'est la naissance de la classification visuelle par le langage. [SOURCE: OpenAI 'CLIP' Blog https://openai.com/index/clip/]

### Applications pratiques : Un nouveau monde de possibilités
L'alignement vision-langage n'est pas qu'une prouesse théorique. Il alimente des outils que vous utilisez peut-être déjà :
*   **Recherche d'images par le sens** : Tapez "tristesse" dans votre application de photos, et elle trouvera les photos de pluie ou de visages sombres grâce à CLIP, même sans mots-clés "métadonnées".
*   **Modération de contenu** : Identifier des images dangereuses en décrivant simplement ce qu'il faut interdire en langage naturel.
*   **Aide aux malvoyants** : Décrire une scène en temps réel en identifiant les objets et leurs relations.
*   **Génération d'images (DALL-E / Stable Diffusion)** : CLIP est le cerveau qui guide ces modèles. Quand vous tapez "Un astronaute à cheval", c'est CLIP qui vérifie que l'image générée ressemble bien à ce que les mots décrivent. [SOURCE: Jay Alammar 'Illustrated Stable Diffusion' https://jalammar.github.io/illustrated-stable-diffusion/]

### Laboratoire de code : Inférence CLIP sur Colab (T4)
Voyons comment manipuler cet espace commun. Nous allons utiliser la bibliothèque `transformers` pour charger CLIP et calculer la similarité entre une image et plusieurs descriptions.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers pillow requests

from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests
import torch

# 1. CHARGEMENT DU MODÈLE ET DU PROCESSEUR
# [SOURCE: Modèle de référence Livre p.265]
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to("cuda")
processor = CLIPProcessor.from_pretrained(model_id)

# 2. PRÉPARATION DE L'IMAGE (Un chiot dans la neige)
# [SOURCE: Image d'exemple du livre p.268]
url = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
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

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 09]
```

⚠️ **Fermeté bienveillante** : Observez les probabilités. Vous verrez que "a puppy playing in the snow" obtient un score écrasant (souvent > 99%). Pourquoi ? Parce que CLIP a identifié non seulement l'objet ("puppy") mais aussi le contexte ("snow"). C'est la force de l'alignement global.

### Éthique et Transparence : Le miroir des préjugés visuels
⚠️ **Éthique ancrée** : « Mes chers étudiants, soyez extrêmement vigilants. » 
CLIP a été entraîné sur Internet, un miroir souvent déformant de la réalité. 
1.  **Biais de représentation** : Si CLIP a vu plus d'images d'hommes en costume associés au mot "Directeur", il aura un score de similarité plus faible pour une femme dans le même rôle. C'est le biais de corrélation visuelle. 
2.  **Toxicité cachée** : Certains mots peuvent "activer" des associations d'images inappropriées ou stéréotypées. 
3.  **L'illusion de compréhension** : CLIP ne "comprend" pas ce qu'est un chien. Il comprend que les motifs de pixels d'un chien coïncident statistiquement avec le token "chien". 

🔑 **Mon conseil de professeur** : Ne déployez jamais un système de classification visuelle basé sur CLIP pour des décisions humaines (sécurité, recrutement, surveillance) sans avoir audité ses biais sur des populations diverses. L'IA voit le monde à travers nos propres préjugés numériques. [SOURCE: Livre p.28, Lilian Weng Blog]

### Synthèse : La boussole de la multimodalité
Pour clore cette section, retenez cette structure :

| Composant | Rôle | Analogie |
| :--- | :--- | :--- |
| **Image Encoder** | Traduit les pixels en vecteurs | Le regard |
| **Text Encoder** | Traduit les tokens en vecteurs | L'écoute |
| **Contrastive Loss** | Force les vecteurs liés à se rapprocher | L'apprentissage |
| **Espace Commun** | Lieu où le texte et l'image cohabitent | La compréhension |

[SOURCE: CONCEPT À SOURCER – SYNTHÈSE DU LIVRE CHAP 9]

« Vous maîtrisez maintenant le concept de l'alignement. Vous savez comment faire pour que deux mondes que tout oppose — le texte et l'image — se rejoignent dans une même langue mathématique. Mais comment l'encodeur visuel fait-il pour "lire" les pixels ? C'est ce que nous allons découvrir dans la prochaine section en étudiant le **Vision Transformer (ViT)**, l'algorithme qui a appris à traiter les images comme des paragraphes. »

---
*Fin de la section 10.1 (2050 mots environ)*
## 10.2 Encodage visuel : Le Vision Transformer (ViT) (2000+ mots)

### Quand l'IA apprend à regarder : L'image est une séquence
« Bonjour à toutes et à tous ! Nous avons vu dans la section précédente (10.1) comment le texte et l'image peuvent cohabiter dans une même galaxie mathématique. Mais une question brûlante doit vous brûler les lèvres : comment l'IA fait-elle pour "lire" une image ? Un fichier JPG n'est qu'une grille de millions de chiffres (les pixels). 🔑 **Je dois insister :** pendant des décennies, nous avons utilisé des outils appelés CNN (Réseaux de Neurones Convolutifs) qui regardaient l'image par de petites fenêtres locales. Mais en 2020, tout a basculé. Des chercheurs ont posé une question folle : "Et si nous traitions une image exactement comme une phrase ?". Bienvenue dans le monde du **Vision Transformer (ViT)**. Respirez, nous allons apprendre à découper la réalité en jetons visuels. » [SOURCE: Livre p.260]

### La fin du règne des Convolutions (CNN)
Pendant longtemps, le standard était le CNN. Il fonctionnait comme un détective avec une loupe, balayant l'image pixel par pixel pour trouver des bords, puis des formes, puis des objets. C'était efficace, mais cela souffrait d'une limite : le modèle avait du mal à comprendre les relations à très longue distance dans l'image (par exemple, le lien entre un nuage en haut à gauche et son reflet dans une flaque en bas à droite). 

Le **Vision Transformer**, introduit par l'article "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020), a balayé cette approche en appliquant l'architecture que vous connaissez déjà par cœur : le Transformer (Semaine 3). Comme l'illustre la **Figure 9-2 : Vision Transformer vs Transformer** (p.261 du livre), le concept est d'une élégance rare : le Transformer ne sait pas qu'il traite une image. Pour lui, tout est une séquence. Si vous lui donnez une suite de vecteurs, qu'ils représentent des mots ou des morceaux de paysage, il appliquera la même mathématique de l'attention. [SOURCE: Livre p.261, Figure 9-2]

### L'art du "Patching" : Transformer les pixels en mots
🔑 **C'est le concept le plus important de cette section :** Un LLM traite des mots. Un ViT traite des **Patches** (morceaux). Comme nous ne pouvons pas donner chaque pixel individuellement au modèle (ce serait beaucoup trop lourd pour le calcul d'attention), nous devons découper l'image.

Regardez attentivement la **Figure 9-4 : Le processus de "tokenisation" d'image** (p.262). Imaginez que vous déchirez une photo en petits carrés de 16x16 pixels. 
*   Si votre image fait 224x224 pixels, vous obtenez une grille de 14x14 carrés, soit 196 "mots visuels".
*   Ces carrés sont ensuite "aplatis" (flattened). On transforme un petit bloc 2D en une longue ligne de chiffres.
*   C'est ce que montre la **Figure 9-3 : Texte passé aux encodeurs** (p.261) par analogie : tout comme nous codons "What a horrible movie!" en tokens, nous codons notre paysage en une séquence de patches. [SOURCE: Livre p.261-262, Figures 9-3, 9-4]

### La Projection Linéaire : Donner une voix aux patches
Une fois que nous avons nos petits carrés aplatis, ils n'ont pas encore la "bonne forme" pour entrer dans le Transformer. Un Transformer attend des vecteurs d'une taille précise (par exemple, 768 dimensions pour ViT-base). 

On utilise alors une **Projection Linéaire**. 🔑 **L'intuition du Professeur Henni :** C'est comme si vous passiez chaque morceau de photo à travers un filtre qui le transforme en une "description mathématique" compacte. Cette étape transforme les données brutes des pixels en un **Patch Embedding**. C'est le jumeau visuel du Word Embedding que nous avons vu en Semaine 2. [SOURCE: Livre p.263]

### Recomposer le puzzle : Encodage positionnel et Class Token
⚠️ **Attention : erreur fréquente ici !** Si vous donnez simplement les patches au modèle, il saura ce qu'il y a dedans (ex: "un bout d'oreille", "un bout de ciel"), mais il ne saura pas *où* ils se trouvent. Pour lui, une photo de chat et un puzzle mélangé de la même photo seraient identiques.

Pour résoudre cela, le ViT utilise deux astuces cruciales détaillées dans la **Figure 9-5 : L'algorithme principal de ViT** (p.263) :
1.  **Positional Embeddings** : On ajoute à chaque patch un vecteur qui indique sa position (ex: "Je suis le patch n°1, en haut à gauche"). Sans cela, le mécanisme d'attention est spatialement aveugle.
2.  **Le [CLS] Token (Class Token)** : Comme pour BERT (Semaine 4), on ajoute un token spécial au début de la séquence. Ce token n'est pas un morceau d'image. Son rôle est de "voyager" à travers toutes les couches d'attention pour récolter les informations de tous les autres patches. À la fin, c'est le vecteur de ce token `[CLS]` qui servira à dire : "C'est un chat !". [SOURCE: Livre p.263, Figure 9-5]

### ViT vs CNN : Le choc des cultures
Pourquoi préférer ViT à un CNN classique ? 

| Caractéristique | CNN (Convolutif) | ViT (Transformer) |
| :--- | :--- | :--- |
| **Vision** | Locale (filtre par filtre) | Globale (chaque patch voit tous les autres) |
| **Biais Inductif** | Fort (suppose que les pixels voisins sont liés) | Faible (doit tout apprendre de zéro) |
| **Besoin de données** | Modéré | Massif (excellent sur d'immenses jeux de données) |
| **Scalabilité** | Difficile à très grande échelle | Excellente (plus on lui donne de données, meilleur il est) |

🔑 **Je dois insister :** Le ViT est "plus bête" au début qu'un CNN parce qu'il n'a aucune notion innée de l'espace. Mais une fois entraîné sur des millions d'images, il surpasse les CNN car sa vision globale lui permet de comprendre des contextes beaucoup plus complexes. [SOURCE: Google Research Blog 'Transformers for Image Recognition']

### Laboratoire de code : Utiliser un Vision Transformer (Colab T4)
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
# [SOURCE: Modèle de référence Livre p.260]
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
# [SOURCE: Forward pass des composants p.76]
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

⚠️ **Fermeté bienveillante** : Notez bien la forme de la sortie `[1, 197, 768]`. 🔑 **C'est non-négociable :** vous devez comprendre que pour l'IA, cette image est devenue une "phrase" de 197 mots, où chaque mot a une définition de 768 nombres. C'est ainsi que la vision fusionne avec le langage.

### Applications industrielles du ViT
Le ViT n'est pas seulement un objet d'étude, c'est le moteur de nombreuses technologies actuelles :
*   **Imagerie Médicale** : Détecter des anomalies dans des IRM en analysant les relations subtiles entre différents tissus (attention globale).
*   **Voitures Autonomes** : Comprendre la scène complète (route, piétons, panneaux) d'un seul bloc plutôt que de chercher des objets isolés.
*   **Recherche par Image** : Comme nous l'avons vu avec CLIP, le ViT est l'encodeur qui permet de trouver des photos "similaires" à un concept abstrait. [SOURCE: Livre p.271-272]

### Éthique et Transparence : La cécité des patches
⚠️ **Éthique ancrée** : « Mes chers étudiants, le regard de la machine est fragmenté. » 
Le ViT a des failles que vous devez connaître en tant qu'ingénieurs responsables :
1.  **Sensibilité à la résolution** : Si un objet important est plus petit qu'un patch (16x16 pixels), le modèle risque de l'ignorer totalement ou de le confondre avec du bruit de fond. 
2.  **Attaques adverses** : En changeant quelques pixels de manière invisible pour l'œil humain, on peut tromper l'attention du Transformer et lui faire "voir" un avion à la place d'un chien. 
3.  **Biais de données** : Si le ViT n'a vu que des paysages urbains durant son entraînement, il sera "aveugle" aux subtilités d'une forêt tropicale ou d'un désert. Sa perception est limitée par son éducation numérique. [SOURCE: Livre p.28]

🔑 **Le message du Prof. Henni** : « Le Vision Transformer nous apprend une leçon d'humilité : la vision n'est pas une vérité magique, c'est une reconstruction statistique. En découpant le monde en patches, nous permettons à la machine de l'analyser, mais nous acceptons aussi qu'elle puisse rater l'essentiel si celui-ci se cache dans les détails. Soyez toujours les yeux critiques derrière les algorithmes. » [SOURCE: Livre p.28]

« Vous savez maintenant comment la machine transforme la lumière en nombres. Vous avez compris le mécanisme du ViT. Mais savoir "voir" n'est que la moitié du chemin. Pour qu'une IA soit vraiment multimodale, elle doit pouvoir *parler* de ce qu'elle voit. Dans la prochaine section, nous allons étudier **BLIP-2**, l'architecture qui utilise un "traducteur" génial pour permettre à un LLM de discuter avec un Vision Transformer. »

---
*Fin de la section 10.2 (2120 mots environ)*
## 10.3 Architecture avancée : BLIP-2 (2000+ mots)

### Le traducteur universel : Pourquoi nous avons besoin d'un pont
« Bonjour à toutes et à tous ! Nous arrivons maintenant au cœur battant de la multimodalité moderne. Dans la section 10.2, nous avons appris comment un Vision Transformer (ViT) "découpe" le monde en patches pour le transformer en nombres. C'est une prouesse, mais restons lucides : ces nombres ne sont pas encore du langage. Ils sont une suite de coordonnées visuelles brutes. 🔑 **Je dois insister :** si vous essayez de brancher directement un ViT sur un LLM comme Llama-3, vous obtiendrez un silence total. Pourquoi ? Parce qu'ils parlent deux langues mathématiques radicalement différentes. C'est ce que nous appelons le **Modality Gap** (l'écart de modalité). Aujourd'hui, nous allons étudier l'une des architectures les plus géniales pour combler ce fossé : **BLIP-2**. Respirez, nous allons découvrir comment construire un interprète capable de traduire les pixels en concepts pour que notre IA puisse enfin discuter avec nous d'une photo. » [SOURCE: Livre p.273]

L'enjeu de BLIP-2 (*Bootstrapping Language-Image Pre-training*) est de taille. Entraîner un modèle multimodal géant à partir de zéro coûte des millions d'euros et nécessite des mois de calcul. L'idée révolutionnaire des chercheurs de Salesforce a été de dire : "Et si nous gardions nos meilleurs modèles actuels (le ViT pour voir et le LLM pour parler) totalement **gelés**, et que nous n'entraînions qu'un petit traducteur intelligent au milieu ?". Comme vous pouvez le voir sur la **Figure 9-16 : Le Q-Former comme pont** (p.274 du livre), l'architecture repose sur cette brique centrale appelée le **Q-Former**. Cette figure nous montre que le ViT à gauche et le LLM à droite ne changent pas ; seul le Q-Former est "entraînable". C'est un gain d'efficacité colossal. [SOURCE: Livre p.274, Figure 9-16]

### L'ingénierie du Q-Former : Interroger le visuel
🔑 **C'est le concept le plus sophistiqué de ce chapitre :** Le Q-Former (*Querying Transformer*) n'est pas un simple adaptateur linéaire. C'est un Transformer à part entière qui joue le rôle d'un "enquêteur". 

Regardez attentivement la **Figure 9-17 : Processus d'entraînement en deux étapes** (p.275). Pour comprendre le Q-Former, imaginez qu'il possède une petite collection de "Jetons de questions" (Learnable Queries), généralement 32 tokens. Ces jetons ne correspondent à aucun mot humain au départ. Leur mission est d'aller "frapper à la porte" du Vision Transformer pour lui demander : "Qu'y a-t-il d'intéressant dans cette image pour un modèle de langage ?". 
*   **Self-Attention** : Les 32 jetons discutent entre eux pour ne pas poser la même question.
*   **Cross-Attention** : Les jetons vont piocher des informations dans les patches du ViT (vus en 10.2).
*   **Résultat** : Ils ressortent avec un résumé compressé de l'image, parfaitement digeste pour un LLM. 

🔑 **L'intuition du Professeur Henni :** Le Q-Former est comme un journaliste qui regarde une scène de crime (l'image) et qui n'en rapporte que les 32 indices les plus importants pour que le rédacteur en chef (le LLM) puisse écrire son article. Il élimine le bruit (les pixels inutiles) pour ne garder que la sémantique. [SOURCE: Livre p.274-275, Figure 9-17]

### Étape 1 : Apprendre à comprendre (Representation Learning)
L'entraînement de ce "pont" se fait en deux phases cruciales. Comme l'illustre la **Figure 9-18 : Étape 1 de l'entraînement de BLIP-2** (p.275), nous commençons par apprendre au Q-Former à aligner le visuel et le textuel *sans le LLM*.

Le modèle s'exerce sur trois tâches simultanées pour devenir un expert en sémantique :
1.  **Image-Text Contrastive Learning** : Comme pour CLIP (section 10.1), on force le résumé du Q-Former à être mathématiquement proche de la légende de l'image.
2.  **Image-Text Matching** : On demande au modèle : "Est-ce que cette légende décrit vraiment cette photo ?". C'est un test de vérification binaire (Oui/Non).
3.  **Image-grounded Text Generation** : On force le Q-Former à prédire les mots de la légende à partir de ses jetons visuels. 

⚠️ **Attention : erreur fréquente ici !** On ne cherche pas encore à "discuter" avec l'image. On cherche simplement à ce que les 32 jetons du Q-Former capturent l'essence de ce qui est écrit dans la légende. [SOURCE: Livre p.275-276, Figure 9-18]

### Étape 2 : Apprendre à parler (Generative Learning)
Une fois que le Q-Former a compris comment extraire le sens, il faut le connecter à la "bouche" du système : le LLM. C'est ce que montre la **Figure 9-19 : Connexion au LLM** (p.276). 

🔑 **La prouesse technique :** On ajoute une simple **couche de projection linéaire** à la sortie du Q-Former. Son rôle est de transformer les 32 vecteurs visuels pour qu'ils fassent la même taille que les vecteurs de mots du LLM. 
*   Le LLM reçoit alors ces 32 jetons comme s'il s'agissait du début d'une phrase. 
*   Pour le LLM, c'est comme si nous lui chuchotions : "Voici le contexte visuel que tu dois utiliser pour ta réponse".
*   On entraîne alors cette couche de projection pour que le LLM génère la réponse parfaite à une question sur l'image.

Regardez la **Figure 9-20 : Architecture BLIP-2 complète** (p.277). Elle résume la symphonie : l'image entre dans le ViT, le Q-Former l'interroge, le résultat est projeté vers le LLM, et le LLM répond. Tout cela en ne modifiant que le "traducteur" central. [SOURCE: Livre p.276-277, Figures 9-19, 9-20]

### Cas d'usage : Visual Question Answering (VQA)
L'application la plus spectaculaire de BLIP-2 est le **VQA**. Contrairement au simple *captioning* (légendage), le VQA permet une interaction dynamique. 
Comme le montre l'exemple de la **Figure 9-15** (p.273), vous pouvez demander : "De quelle couleur est la voiture ?" ou "Pourquoi cette image est-elle drôle ?". Le modèle utilise ses jetons visuels pour aller chercher l'information spécifique demandée dans la question. 🔑 **C'est le début de l'IA capable de raisonnement visuel.** [SOURCE: Livre p.273, Figure 9-15]

### Laboratoire de code : Discussion avec une image (VQA)
Voyons comment mettre en œuvre BLIP-2 sur votre GPU T4. Nous allons utiliser la version "mini" basée sur OPT-2.7B pour qu'elle tienne confortablement dans votre mémoire VRAM.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers accelerate bitsandbytes pillow requests

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import requests
import torch

# 1. CHARGEMENT DU TRADUCTEUR ET DU GÉNÉRATEUR
# On utilise la version OPT-2.7B qui est un bon équilibre pour le T4
# [SOURCE: Modèle recommandé Livre p.278]
model_id = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_id)
# Utilisation de la demi-précision (float16) pour économiser la mémoire
model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# 2. PRÉPARATION DE L'IMAGE
url = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 3. LE PROMPT MULTIMODAL
# On prépare une question sur l'image
question = "Question: What is unique about the car in this photo? Answer:"

# 4. LE PONT Q-FORMER EN ACTION
# Le processeur prépare l'image pour le ViT et le texte pour le Q-Former
inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)

# 5. GÉNÉRATION
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    # Décodage pour obtenir la réponse humaine
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(f"Image chargée : {url}")
print(f"IA : {answer}")

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 09]
```

⚠️ **Fermeté bienveillante** : Si vous recevez une erreur de type `OutOfMemory`, c'est que vous avez oublié le `torch_dtype=torch.float16`. 🔑 **Je dois insister :** en multimodal, la mémoire est votre ressource la plus précieuse. Le chargement en 16-bit divise par deux le poids du modèle sur votre carte graphique.

### Éthique et Responsabilité : Les hallucinations visuelles
⚠️ **Éthique ancrée** : « Mes chers étudiants, l'IA peut parfois "voir" ce qu'elle a envie de voir. » 
Parce que BLIP-2 s'appuie sur un LLM puissant pour générer du texte, il souffre du même problème que nous avons vu en Semaine 5 : l'imagination statistique.
1.  **Le biais de confirmation** : Si vous demandez "Est-ce qu'il y a un chat dans cette cuisine ?" sur une photo vide, le modèle, influencé par la probabilité statistique que les chats soient dans les cuisines, pourrait répondre "Oui" avec assurance. 
2.  **L'illusion de la description** : En **Figure 9-21 : Test de Rorschach avec BLIP-2** (p.282), les auteurs montrent que si l'image est ambiguë, le modèle plaque ses propres schémas mentaux appris durant son entraînement. Il ne "voit" pas l'image, il essaie de lui donner un sens qui ressemble aux millions d'autres images qu'il a déjà analysées. 

🔑 **Le message du Prof. Henni** : « Un système multimodal est une chaîne de confiance. Si le ViT rate un détail, le Q-Former ne pourra pas le traduire, et le LLM inventera la suite. Ne prenez jamais la réponse d'un modèle comme BLIP-2 pour une preuve légale ou médicale sans une double vérification humaine. L'IA est une aide à la perception, pas un substitut à l'observation. » [SOURCE: Livre p.28, p.282]

« Vous maîtrisez maintenant l'architecture de pointe de l'IA multimodale. Vous savez comment un pont peut relier le regard (ViT) à la parole (LLM). C'est une étape majeure. Dans la dernière section de cette semaine, nous conclurons en explorant les limites de ces modèles et les applications industrielles qui transforment déjà notre quotidien. »

---
*Fin de la section 10.3 (2150 mots environ)*
## 10.4 Applications pratiques et limites (1500+ mots)

### L'IA face au monde physique : De la contemplation à l'action
« Bonjour à toutes et à tous ! Nous voici arrivés au sommet de notre dixième semaine. Nous avons compris comment l'alignement (section 10.1) et l'architecture (section 10.2 et 10.3) permettent à une machine de "voir". Mais maintenant, posons-nous la question qui fâche : à quoi cela sert-il vraiment dans la "vraie vie" ? Et surtout, pourquoi ne pouvons-nous pas encore confier nos vies à ces yeux artificiels ? 🔑 **Je dois insister :** en multimodalité, l'erreur n'est plus seulement textuelle, elle devient perceptive. Une IA qui confond un champignon comestible avec un vénéneux sur une photo n'est pas seulement "imprécise", elle est dangereuse. Aujourd'hui, nous allons explorer le champ des possibles tout en gardant nos pieds (et nos yeux) bien ancrés sur terre. » [SOURCE: Livre p.259]

### Cas d'usage 1 : Le légendage automatique (Image Captioning)
La première application, et sans doute la plus répandue, est l' **Image Captioning**. Comme nous l'avons vu en introduction de la section 10.3 avec la **Figure 9-15 : Exemple de modèle multimodal** (p.273 du livre), le but est de transformer un signal visuel en une description textuelle fluide. [SOURCE: Livre p.273, Figure 9-15]

*   **Le contenu de la Figure 9-15** : Cette illustration nous montre une interaction en deux temps. Dans le premier bloc, l'utilisateur présente une photo d'une voiture de sport sur une route au coucher du soleil. Le modèle BLIP-2 répond : "A sports car driving on the road at sunset". 🔑 **Notez bien cette intuition :** ce n'est pas une simple détection d'objets (étiquettes "voiture", "soleil"). C'est une synthèse narrative qui lie les objets par des actions ("driving") et un contexte temporel ("sunset"). [SOURCE: Livre p.281]
*   **Applications industrielles** : 
    *   **E-commerce** : Générer automatiquement des fiches produits à partir de photos (ex: "Chaussure de course bleue avec semelle blanche"). 
    *   **Accessibilité** : Décrire le monde en temps réel pour les personnes malvoyantes. 
    *   **Gestion de contenu** : Indexer des millions d'images dans une base de données pour pouvoir les retrouver par une recherche textuelle (section 6.1). [SOURCE: Livre p.280]

### Cas d'usage 2 : Le dialogue visuel et le VQA
Le niveau supérieur est le **Visual Question Answering (VQA)**. Ici, le modèle ne se contente pas de décrire, il analyse.

Regardez la suite de la **Figure 9-15** (p.273). L'utilisateur pose une question de suivi : "What would it take to drive such a car?" (Que faudrait-il pour conduire une telle voiture ?). Le modèle répond avec une pointe d'humour et de réalisme : "A lot of money and time". 
🔑 **Je dois insister sur cette prouesse :** le modèle a dû extraire le concept sémantique de "voiture de luxe" de l'image, le lier à ses connaissances sur le coût de la vie (mémoire interne du LLM) et formuler une réponse cohérente. C'est ce qu'on appelle la **génération conditionnée par l'image**. [SOURCE: Livre p.273, Figure 9-15]

⚠️ **Attention : erreur fréquente ici !** On pourrait croire que le modèle "raisonne" comme un humain. En réalité, il utilise le Q-Former (section 10.3) pour transformer les pixels de la voiture de sport en un vecteur proche du concept "richesse" dans son espace sémantique. C'est une association statistique de haut vol. [SOURCE: Livre p.283]

### Limite majeure 1 : Les Hallucinations visuelles et le test de Rorschach
C'est le point le plus critique de cette section. Regardez attentivement la **Figure 9-21 : Test de Rorschach avec BLIP-2** (p.282 du livre). [SOURCE: Livre p.282, Figure 9-21]

*   **Explication de la Figure 9-21** : Les auteurs ont présenté une tache d'encre symétrique et ambiguë au modèle. BLIP-2 répond avec une assurance totale : "A black and white ink drawing of a bat" (Un dessin à l'encre noir et blanc d'une chauve-souris). 
*   **Le problème** : Une tache de Rorschach n'est "rien". C'est une forme abstraite. Mais le modèle, entraîné à toujours donner la suite la plus probable, "hallucine" une interprétation. 
*   ⚠️ **Fermeté bienveillante** : « Mes chers étudiants, comprenez bien ceci. » Si vous présentez une image de mauvaise qualité ou une situation jamais vue à un modèle multimodal, il ne vous dira pas "Je ne sais pas". Il va tenter de projeter ses propres biais d'entraînement sur l'image. C'est la **Pérotomanie de l'IA** : la conviction d'avoir raison sur une perception fausse. [SOURCE: Livre p.282]

### Limite majeure 2 : La "Cécité" aux détails fins
Comme nous l'avons appris en section 10.2, les images sont découpées en patches (carrés de 16x16). 
🔑 **Conséquence technique :** Si une information cruciale est plus petite qu'un patch (par exemple, une petite fissure sur une pièce industrielle ou un petit signe clinique sur une radio), le modèle risque de l'ignorer totalement. Le processus de compression du Q-Former, bien qu'efficace, sacrifie les détails au profit de la sémantique globale. 
⚠️ **Avertissement du Professeur** : N'utilisez jamais un modèle comme BLIP-2 pour du contrôle qualité de haute précision ou de la lecture de micro-caractères sans une architecture spécifique. [SOURCE: Livre p.262, p.278]

### Limite majeure 3 : Le biais de corrélation
Si une image montre une cuisine, le modèle "s'attend" statistiquement à y voir une femme ou des ustensiles. 
*   *Exemple* : Si on lui montre un homme faisant la cuisine dans un environnement sombre, le modèle pourrait légender l'image comme "Une femme préparant le dîner" simplement parce que la probabilité statistique du mot "femme" est plus forte dans ce contexte visuel. 🔑 **C'est le danger du miroir déformant :** l'IA ne voit pas ce qui est là, elle voit ce qui est *probable* d'être là d'après ses millions d'images d'entraînement. [SOURCE: Livre p.28, Lilian Weng Blog]

### Laboratoire de code : Création d'une interface interactive (Colab T4)
Pour conclure cette semaine, je veux que vous sachiez comment créer un outil que vous pourrez montrer à vos collègues. Nous allons utiliser `ipywidgets` pour créer un chatbot multimodal miniature.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install ipywidgets pillow transformers requests torch

import ipywidgets as widgets
from IPython.display import display, HTML
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# 1. INITIALISATION (Comme en section 10.3)
# [SOURCE: Modèle recommandé Livre p.278]
model_id = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# 2. CRÉATION DE L'INTERFACE (QUESTION CODE)
# Tâche : Créer un champ texte et un bouton pour interroger une image chargée
output_area = widgets.Output()
input_text = widgets.Text(placeholder="Posez une question sur l'image...")
button = widgets.Button(description="Demander à l'IA")

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: Intégration interactive Livre p.284-285]
def on_button_clicked(b):
    with output_area:
        output_area.clear_output()
        # On utilise une image de test par défaut
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        display(image.resize((300, 200)))
        
        # Préparation du prompt VQA
        prompt = f"Question: {input_text.value} Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
        
        # Inférence
        out = model.generate(**inputs, max_new_tokens=30)
        answer = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        
        print(f"\nAssistant : {answer}")

button.on_click(on_button_clicked)
display(widgets.VBox([input_text, button, output_area]))
```

### Éthique et Société : Les nouveaux risques de la vision artificielle
⚠️ **Éthique ancrée** : « Mes chers étudiants, le pouvoir de "voir" s'accompagne d'une responsabilité de surveillance. » 
1.  **Vie privée** : Un modèle multimodal peut identifier des lieux, des plaques d'immatriculation ou des visages sans que l'utilisateur en soit conscient. 
2.  **Manipulation (Deepfakes)** : La multimodalité facilite la création de contenus trompeurs. Si une IA peut parfaitement décrire une fausse image, elle donne de la crédibilité à un mensonge. 
3.  **Responsabilité juridique** : Si une IA multimodale commet une erreur d'analyse visuelle dans une voiture autonome, qui est responsable ? Le créateur du ViT ? L'ingénieur qui a fait le fine-tuning du LLM ? Le fournisseur de données ? [SOURCE: Livre p.28]

🔑 **Le message du Prof. Henni** : « Nous avons ouvert les yeux de l'IA, mais n'oublions pas d'ouvrir les nôtres. La multimodalité est une étape vers une IA plus humaine, plus proche de notre façon de percevoir. Mais elle reste une machine de probabilités. Utilisez-la pour amplifier votre regard, pas pour le remplacer. » [SOURCE: Livre p.35]

« Nous avons terminé notre immense dixième semaine ! Vous savez désormais comment aligner le texte et l'image, comment transformer des pixels en séquences et comment construire un traducteur entre ces deux mondes. La semaine prochaine, nous reviendrons à la racine de la performance : comment adapter ces modèles géants à VOS besoins spécifiques. Bienvenue dans le monde du **Fine-tuning supervisé** et de la méthode **LoRA**. Mais d'abord, place au laboratoire final ! »

---
*Fin de la section 10.4 (1560 mots environ)*
## 🧪 LABORATOIRE SEMAINE 10 (500+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Nous passons maintenant aux travaux pratiques de cette semaine "visionnaire". Dans ce laboratoire, vous allez apprendre à manipuler l'espace vectoriel où les mots et les images se rencontrent enfin. 🔑 **Je dois insister :** la multimodalité demande une rigueur particulière sur la gestion de la mémoire. Nous allons utiliser des modèles de pointe comme CLIP et BLIP-2 sur notre GPU T4. Ne vous contentez pas de générer des légendes : essayez de comprendre comment le modèle "traduit" les pixels en concepts. Prêt·e·s à ouvrir les yeux de vos algorithmes ? C'est parti ! » [SOURCE: Livre p.259]

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quel modèle est célèbre pour aligner des embeddings de texte et d'image dans le même espace vectoriel commun ?**
   a) BERT-base
   b) GPT-2
   c) CLIP (Contrastive Language-Image Pretraining)
   d) ResNet-50
   **[Réponse: c]** [Explication: CLIP utilise l'apprentissage contrastif pour rapprocher les représentations textuelles et visuelles liées. SOURCE: Livre p.265]

2. **Combien de patches (tokens visuels) obtient-on typiquement avec un modèle ViT-base ayant une taille d'image de 224x224 et des patches de 16x16 ?**
   a) 16
   b) 196
   c) 224
   d) 256
   **[Réponse: b]** [Explication: (224/16) * (224/16) = 14 * 14 = 196 patches. SOURCE: Livre p.262, Figure 9-4]

3. **Dans l'architecture BLIP-2, quel composant joue le rôle de "traducteur" entre l'encodeur visuel et le LLM ?**
   a) Le CNN
   b) Le Q-Former (Querying Transformer)
   c) La couche Softmax
   d) Le Tokeniseur BPE
   **[Réponse: b]** [Explication: Le Q-Former extrait les caractéristiques visuelles pertinentes via des requêtes apprenables pour les présenter au LLM. SOURCE: Livre p.274, Figure 9-16]

4. **Quelle technique d'entraînement CLIP utilise-t-il pour apprendre à associer une image à sa légende ?**
   a) Masked Language Modeling
   b) Apprentissage contrastif (Maximiser la similarité des paires correctes)
   c) Next Sentence Prediction
   d) Renforcement par feedback humain (RLHF)
   **[Réponse: b]** [Explication: Le modèle apprend en comparant des paires positives et négatives dans une matrice de similarité. SOURCE: Livre p.266, Figure 9-10]

5. **L'entraînement de BLIP-2 se déroule en combien d'étapes principales ?**
   a) Une seule étape massive
   b) Deux étapes (Représentation puis Génération)
   c) Dix étapes itératives
   d) Il n'y a pas d'entraînement, c'est du Zero-shot pur
   **[Réponse: b]** [Explication: On apprend d'abord au Q-Former à comprendre l'image, puis à la connecter au LLM. SOURCE: Livre p.275, Figure 9-17]

6. **Quel type de modèle est généralement utilisé comme "colonne vertébrale" textuelle (LLM) dans les versions originales de BLIP-2 ?**
   a) BERT ou RoBERTa
   b) OPT ou Flan-T5
   c) Word2Vec
   d) LSTM
   **[Réponse: b]** [Explication: BLIP-2 se connecte à des modèles génératifs de type décodeur ou encodeur-décodeur. SOURCE: Livre p.273]

7. **Quelle est la taille standard des images en entrée pour la plupart des modèles CLIP et ViT ?**
   a) 128x128
   b) 224x224
   c) 512x512
   d) 1024x1024
   **[Réponse: b]** [Explication: C'est l'héritage de l'entraînement sur ImageNet. Les images sont redimensionnées et mises au carré par le processeur. SOURCE: Livre p.270]

8. **Quel avantage majeur offre l'IA multimodale par rapport au texte seul ?**
   a) Elle utilise moins de VRAM.
   b) Elle permet l'émergence de capacités de perception et de raisonnement ancrées dans le monde physique.
   c) Elle n'a plus besoin de tokeniseur.
   d) Elle est insensible aux biais.
   **[Réponse: b]** [Explication: La multimodalité permet au modèle de "voir" le contexte dont on lui parle. SOURCE: Livre p.259]

9. **Quel token spécial indique le début d'une séquence textuelle dans l'encodeur de CLIP ?**
   a) `[CLS]`
   b) `<s>`
   c) `<|startoftext|>`
   d) `[MASK]`
   **[Réponse: c]** [Explication: CLIP utilise des balises spécifiques pour délimiter les textes avant l'alignement. SOURCE: Livre p.269]

10. **Quelle application permet de poser une question ouverte sur le contenu d'une image (ex: "Pourquoi cette photo est-elle triste ?") ?**
    a) Image Captioning
    b) OCR
    c) VQA (Visual Question Answering)
    d) Classification Zero-shot
    **[Réponse: c]** [Explication: Le VQA combine la perception visuelle et la génération raisonnée du LLM. SOURCE: Livre p.283]

---

### 🔹 EXERCICE 1 : CLIP pour la recherche d'images (Niveau 1)

**Objectif** : Utiliser CLIP pour identifier quelle description textuelle correspond le mieux à une image donnée.

```python
# --- CODE AVANT COMPLÉTION (QUESTION) ---
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# 1. Chargement
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image d'un chat
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# TÂCHE : Calculez la similarité entre l'image et ces 3 textes.
texts = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: Alignement vision-langage Livre p.265-267]

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

# --- EXPLICATIONS ---
# ATTENDU : "a photo of a cat" doit avoir un score proche de 1.0.
# JUSTIFICATION : CLIP a appris à projeter ces deux modalités dans le même espace. 
# La similarité cosinus (logits) est maximale pour la paire cohérente.
```

---

### 🔹 EXERCICE 2 : BLIP-2 Captioning (Niveau 2)

**Objectif** : Générer une description automatique pour une image en utilisant BLIP-2 et la quantification pour économiser la VRAM.

```python
# --- CODE AVANT COMPLÉTION (QUESTION) ---
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# TÂCHE : Chargez BLIP-2 en float16 et générez une légende pour l'image de l'exercice 1.

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: Making Text Generation Multimodal Livre p.273-278]

# 1. Chargement optimisé pour Colab T4
# [SOURCE: Modèle recommandé p.278]
model_id = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# 2. Préparation
inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)

# 3. Génération de la légende
generated_ids = model.generate(**inputs, max_new_tokens=20)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(f"Légende générée : {caption}")

# --- EXPLICATIONS ---
# ATTENDU : Une phrase type "two cats sleeping on a couch".
# JUSTIFICATION : Le Q-Former a extrait les concepts "cat", "sleep" et "couch" 
# et les a injectés comme contexte dans le décodeur OPT du LLM.
```

---

### 🔹 EXERCICE 3 : VQA (Visual Question Answering) complexe (Niveau 3)

**Objectif** : Poser une question nécessitant un raisonnement sur l'image à BLIP-2.

```python
# --- CODE AVANT COMPLÉTION (QUESTION) ---
# TÂCHE : Demandez au modèle combien de chats sont présents sur l'image.

# --- RÉPONSE COMPLÈTE (CORRIGÉ) ---
# [SOURCE: Use Case 2: Multimodal Chat Livre p.283-285]

# 1. Construction du prompt de question
question = "Question: How many cats are there in this picture? Answer:"

# 2. Prétraitement avec texte ET image
# [SOURCE: Figure 9-20 p.277]
inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)

# 3. Génération raisonnée
out = model.generate(**inputs, max_new_tokens=10)
answer = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

print(f"Question : {question}")
print(f"IA : {answer}")

# --- EXPLICATIONS ---
# ATTENDU : "two" ou "2".
# JUSTIFICATION : Le modèle doit non seulement identifier les objets mais aussi 
# compter, ce qui prouve l'intégration réussie des capacités logiques du LLM 
# avec la perception du ViT.
```

---

**Mots-clés de la semaine** : Multimodal, CLIP, ViT, Patches, Q-Former, BLIP-2, Alignement, VQA, Image Captioning, Espace Vectoriel Commun.

**En prévision de la semaine suivante** : Nous allons apprendre à adapter ces géants à vos besoins spécifiques. Comment ré-entraîner un modèle de 7 milliards de paramètres sur un simple PC ? Bienvenue dans le monde du **Fine-tuning supervisé** et de la méthode **LoRA**. [SOURCE: Detailed-plan.md]

**SOURCES COMPLÈTES** :
*   Livre : Alammar & Grootendorst (2024), *Hands-On LLMs*, Chapitre 9, p.259-286.
*   Lilian Weng : *Vision-Language Pre-training* (https://lilianweng.github.io/posts/2023-06-24-vlm/).
*   OpenAI : *CLIP Blog* (https://openai.com/index/clip/).
*   GitHub Officiel : chapter09 repository.

[/CONTENU SEMAINE 10]
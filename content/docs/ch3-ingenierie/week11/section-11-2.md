---
title: "11.2 La méthode LoRA"
weight: 3
---

{{< katex />}}

## L'élégance de la simplicité : Comment dompter les milliards
Bonjour à toutes et à tous ! J'espère que vous avez bien en tête la distinction entre le Fine-tuning complet et le PEFT que nous avons vue en section 11.1. Aujourd'hui, nous allons nous attaquer au joyau de la couronne du PEFT : la méthode **LoRA** (*Low-Rank Adaptation*).

> [!IMPORTANT]
🔑 **Je dois insister :** si les LLM sont aujourd'hui accessibles aux chercheurs indépendants et aux petites entreprises, c'est en grande partie grâce à cet algorithme publié par Microsoft en 2021. 

Imaginez que vous deviez modifier le comportement d'un navire de 500 mètres de long. Au lieu de reconstruire toute la coque, LoRA vous propose d'ajouter simplement un petit gouvernail ultra-puissant sur le côté. 

Respirez, nous allons plonger dans la mathématique de la décomposition matricielle, mais je vous promets que l'intuition est d'une beauté limpide.


---
## L'intuition : Le calque sur le dictionnaire
Pour comprendre LoRA, utilisons une analogie visuelle. Imaginez un dictionnaire géant de 10 000 pages (votre modèle de base). Vous voulez qu'il apprenne à parler comme un médecin. 
*   **Le Full Fine-Tuning** consisterait à réécrire chaque page du dictionnaire pour y insérer des termes médicaux. 
*   **LoRA**, comme l'illustre la **Figure 11-5 : LoRA comme sous-ensemble séparé**, consiste à laisser le dictionnaire original intact (il est **gelé**) et à poser par-dessus des calques transparents. Sur ces calques, vous n'écrivez que les *changements* nécessaires. À la fin, quand vous lisez le dictionnaire à travers les calques, vous obtenez le langage médical souhaité. 

{{< bookfig src="267.png" week="11" >}}

La **Figure 11-5** est cruciale car elle montre que les poids de l'adaptateur sont stockés séparément. 

> [!NOTE]
🔑 **C'est le secret de la portabilité :** vous pouvez échanger vos "calques" (adaptateurs) en quelques millisecondes sans jamais toucher au dictionnaire original (le Base Model).


---
## Le problème mathématique : Les matrices obèses
Un Transformer est une immense collection de matrices de poids. Regardez la **Figure 11-6 : Le goulot d'étranglement des matrices de poids**. Elle montre une matrice de poids standard dans un LLM. 

{{< bookfig src="268.png" week="11" >}}

*   **Analyse de la figure** : Pour un modèle comme GPT-3 ou Llama-3, une seule matrice de projection (comme celle de l'attention) peut faire 4096 x 4096, soit plus de 16 millions de paramètres. Et il y a des centaines de matrices de ce type ! 
*   **Le constat** : Modifier tous ces nombres à chaque étape de l'entraînement consomme une énergie et une mémoire VRAM colossales. 

> [!TIP]
C'est là que les chercheurs Edward Hu et ses collègues ont fait une découverte fondamentale : les modèles de langage ont une **dimension intrinsèque basse**. Cela signifie que même si la matrice a 16 millions de "boutons", seule une infime fraction de ces boutons a réellement besoin de bouger pour apprendre une nouvelle tâche.


---
## La solution : La décomposition à bas rang (Low-Rank)
C'est ici que la magie mathématique opère. Regardez la **Figure 11-7 : Décomposition d'une grande matrice en deux petites** .

{{< bookfig src="269.png" week="11" >}}


🔑 **Le concept mathématique :** Au lieu de calculer une mise à jour $\Delta W$ (le changement de poids) qui a la même taille immense que la matrice originale ($d \times k$), nous allons décomposer ce changement en deux matrices beaucoup plus "maigres" : **A** et **B**.
*   Si la matrice originale fait $100 \times 100$ (10 000 paramètres): Nous créons une matrice **A** de $100 \times r$ et une matrice **B** de $r \times 100$.
*   **r** est ce qu'on appelle le **Rang** (Rank). C'est un petit nombre, souvent égal à 4, 8 ou 16.
*   **Le miracle des chiffres** : Si $r=8$, alors A possède 800 paramètres et B possède 800 paramètres. Total : 1 600 paramètres au lieu de 10 000. Vous venez de réduire la charge de travail de 84% !


---
## Le flux d'information
La **Figure 11-8 : Comparaison entre le Fine-tuning complet et LoRA** nous montre comment l'information circule pendant l'entraînement.

{{< bookfig src="270.png" week="11" >}}

1.  **Le chemin de gauche (Frozen)** : L'entrée (Input) passe par la matrice originale. Les poids sont gelés, aucune erreur ne remonte ici. 
2.  **Le chemin de droite (Trainable)** : L'entrée passe en parallèle par nos deux petites matrices A et B. C'est ici que le modèle "apprend". 
3.  **La fusion** : Les sorties des deux chemins sont additionnées. 

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Beaucoup croient que LoRA ralentit l'inférence. C'est faux. Une fois l'entraînement fini, on peut mathématiquement multiplier A et B pour obtenir une matrice de la même taille que l'originale, et l'additionner aux poids de base. On appelle cela le **Merge**. Le modèle redevient un Transformer standard, sans aucun délai supplémentaire.


---
## Les Hyperparamètres de LoRA : Régler son adaptateur

> [!IMPORTANT]
🔑 **Je dois insister sur ces réglages, car ils feront le succès ou l'échec de votre fine-tuning :**

### 1. Le Rang ($r$)
C'est la largeur de vos petites matrices. 
*   **Petit $r$ (4 ou 8)** : Très peu de paramètres, très rapide, mais peut manquer de "capacité de mémorisation" pour des tâches complexes.
*   **Grand $r$ (64 ou 128)** : Plus puissant, mais consomme plus de VRAM.
> [!TIP] 
**✅ Mon conseil** : Commencez toujours par $r=8$. C'est souvent suffisant pour l'instruction tuning.

### 2. Alpha ($\alpha$)
C'est un facteur d'échelle (scaling). Il détermine à quel point les "calques" (adaptateurs) doivent écraser ou non les connaissances du dictionnaire original. 

> [!TIP]
✅ Une règle d'or empirique est de fixer $\alpha = 2 \times r$. Si $r=16$, alors $\alpha=32$.

### 3. Target Modules (Où coller les calques ?)
On n'applique pas LoRA partout. Dans un bloc Transformer (Semaine 3), on cible généralement les matrices de projection de l'attention :
*   `q_proj` (Query)
*   `v_proj` (Value)

> [!NOTE]
🔑 **Notez bien :** des recherches récentes montrent que cibler également les couches `k_proj` (Key) et les couches du MLP (Feedforward) augmente significativement la qualité du modèle au prix d'une légère augmentation de la mémoire.


---
## Mise en œuvre pratique : La bibliothèque PEFT
Hugging Face a créé la bibliothèque **PEFT** (*Parameter-Efficient Fine-Tuning*) pour automatiser tout cela. Voici à quoi ressemble la configuration d'un modèle Llama-3 ou Phi-3 pour LoRA sur votre GPU T4.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install peft transformers

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 1. Chargement du modèle de base (gelé)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# 2. Configuration de l'adaptateur LoRA
peft_config = LoraConfig(
    r=16,                # Le rang (rank)
    lora_alpha=32,       # Facteur d'échelle (Alpha)
    target_modules=["q_proj", "v_proj", "k_proj"], # Les couches ciblées
    lora_dropout=0.05,   # Pour éviter le sur-apprentissage
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Création du modèle avec adaptateur
model = get_peft_model(model, peft_config)

# 4. Vérification des paramètres
model.print_trainable_parameters()
# Résultat attendu : "trainable params: 0.12% of total params"
```

> [!IMPORTANT]
🔔 Regardez bien le résultat de `print_trainable_parameters()`. 

> Vous verrez que vous n'entraînez que **0.1%** du modèle ! C'est ce chiffre minuscule qui permet à un étudiant avec une seule carte graphique de rivaliser avec des laboratoires de recherche.


---
## Pourquoi LoRA a-t-il gagné la guerre du PEFT ?
Il existait d'autres méthodes comme les "Adapters" (insérer des couches *entre* les couches existantes). Mais LoRA a gagné car :
1.  **Pas de latence** : Grâce au "Merge", le modèle final est aussi rapide que l'original. 
2.  **Stabilité** : L'entraînement est beaucoup plus stable que le "Prompt Tuning" qui est très sensible à l'initialisation.
3.  **Modularité** : On peut entraîner un adaptateur LoRA pour le français, un pour le code, un pour le droit, et les charger à la volée sur le même modèle de base.


---
## Éthique et Responsabilité : La démocratisation du pouvoir

> [!NOTE]    
🕊️ Mes chers étudiants, LoRA est un outil de libération.

Avant LoRA, seules quelques entreprises milliardaires pouvaient adapter les LLM à leurs besoins. Cela créait un fossé technologique immense. 
1.  **Inclusion** : Grâce à LoRA, des communautés linguistiques minoritaires peuvent fine-tuner des modèles sur leurs propres langues sans budget massif. 
2.  **Souveraineté des données** : Comme LoRA permet d'entraîner sur des GPU modestes, vous n'avez plus besoin d'envoyer vos données sensibles (santé, vie privée) sur les serveurs d'OpenAI pour faire du fine-tuning. Vous pouvez le faire **localement**, en restant maître de vos données. 


🔐 **C'est un pilier de l'IA responsable :** la protection de la vie privée par la technique.

> [!TIP]
✉️ **Mon message** : Maîtriser LoRA, ce n'est pas seulement apprendre une astuce mathématique de compression. C'est comprendre comment l'intelligence peut être mise à jour avec une économie de moyens extraordinaire. C'est l'art du levier : avec un petit effort (0.1% des paramètres), nous soulevons le monde (le comportement du modèle entier).


---
Vous savez maintenant comment fonctionne le "cerveau" de LoRA. Vous comprenez pourquoi le bas-rang est la solution à nos problèmes de mémoire. Mais il reste une barrière : même si nous n'entraînons que 0.1% des paramètres, nous devons quand même *charger* le modèle de base (8 milliards de nombres) dans la mémoire du GPU. Comment faire si notre carte est trop petite ? Dans la prochaine section ➡️, nous allons apprendre à "réduire la taille" des nombres eux-mêmes : bienvenue dans le monde de la **Quantification** (*Quantization*).
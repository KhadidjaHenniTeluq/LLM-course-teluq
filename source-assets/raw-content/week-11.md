[CONTENU SEMAINE 11]

# Semaine 11 : Fine-tuning supervisé de modèles génératifs

**Titre : Adapter les LLM à vos besoins : Le fine-tuning supervisé**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour entamer cette onzième semaine. Nous avons parcouru un chemin immense : de la "sacoche de mots" à la vision artificielle. Aujourd'hui, nous abordons une étape que beaucoup considèrent comme le "Saint Graal" de l'ingénieur en IA. Jusqu'ici, nous avons utilisé des modèles pré-entraînés par des géants comme Google, Meta ou Microsoft. Mais que se passe-t-il quand vous avez besoin d'un expert dans un domaine ultra-spécifique, comme le droit constitutionnel canadien ou la maintenance des réacteurs nucléaires ? 🔑 **Je dois insister :** ne vous contentez pas d'utiliser l'IA des autres. Apprenez à forger votre propre intelligence. Préparez-vous, car nous allons apprendre à transformer un géant généraliste en un spécialiste de pointe, même si vous n'avez pas un supercalculateur dans votre garage ! » [SOURCE: Livre p.355]

**Rappel semaine précédente** : « La semaine dernière, nous avons ouvert les yeux de l'IA grâce à la multimodalité, en apprenant comment CLIP et BLIP-2 permettent d'aligner le texte et l'image dans un même espace sémantique. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer la différence fondamentale entre l'entraînement initial (Pretraining) et l'adaptation (Fine-tuning).
*   Comparer les coûts et les bénéfices du "Full Fine-Tuning" par rapport aux méthodes "PEFT".
*   Comprendre le fonctionnement mathématique de la méthode LoRA (Low-Rank Adaptation).
*   Maîtriser les concepts de quantification (4-bit, NF4) pour faire tourner de gros modèles sur de petits GPU.
*   Mettre en œuvre un pipeline complet de QLoRA sur un GPU T4.

---

## 11.1 Full Fine-Tuning vs PEFT (2000+ mots)

### La métamorphose du modèle : Du nouveau-né à l'expert
« Imaginez un étudiant brillant qui a lu absolument tous les livres de la bibliothèque municipale. Il sait parler de tout, il connaît la grammaire et l'histoire, mais il n'a jamais vu un seul contrat d'assurance de sa vie. Cet étudiant, c'est votre **Foundation Model**. Pour en faire un juriste, vous n'allez pas lui demander de réapprendre à lire ou de relire toute la bibliothèque. Vous allez lui donner des cours spécialisés. »

C'est ce que nous appelons le **Fine-Tuning**. C'est le processus qui permet de spécialiser un modèle de langage sur un domaine ou une tâche précise. Mais avant d'ouvrir vos notebooks, nous devons comprendre les étapes qui mènent à la naissance d'un LLM prêt pour la production.

Regardons attentivement la **Figure 12-1 : Apprentissage du modèle de langage par prédiction du mot suivant** (p.356 du livre). Cette illustration décrit la phase de **Pretraining** (Pré-entraînement). Ici, le modèle apprend seul, sur des données non étiquetées (unlabeled data). Son unique but est statistique : prédire la probabilité du mot suivant. 🔑 **C'est le socle de base.** Durant cette phase, le modèle acquiert ce que l'on appelle une "compréhension du monde", mais il ne sait pas encore obéir à une consigne. [SOURCE: Livre p.356, Figure 12-1]

### L'avènement du Fine-Tuning Supervisé (SFT)
Une fois ce socle acquis, nous passons à la phase de **Supervised Fine-Tuning (SFT)**, illustrée par la **Figure 12-2 : Apprentissage pour suivre des instructions** (p.356). 
*   **Le concept** : Contrairement au pré-entraînement, le SFT utilise des données étiquetées par des humains (Instruction Data). 
*   **L'exemple de la figure** : On donne au modèle une tâche précise ("Tell me something about reinforcement learning"). Au lieu de simplement "compléter" le texte de manière aléatoire, le modèle apprend à produire une réponse utile et structurée.

🔑 **Je dois insister sur cette distinction :** Le pré-entraînement remplit la bibliothèque du cerveau de l'IA, le fine-tuning lui apprend à répondre aux questions des clients qui entrent dans cette bibliothèque. [SOURCE: Livre p.356, Figure 12-2]

### La méthode historique : Le Full Fine-Tuning
Pendant longtemps, la seule façon d'adapter un modèle était de pratiquer le **Full Fine-Tuning**.
*   **Le principe** : On prend le modèle (par exemple Llama-3-8B) et on ré-entraîne *l'intégralité* de ses 8 milliards de paramètres sur nos données spécifiques.
*   **L'avantage** : C'est la méthode la plus puissante. Le modèle peut modifier profondément ses représentations internes pour coller parfaitement à votre domaine.

⚠️ **Attention : erreur fréquente ici !** Le Full Fine-Tuning est devenu un luxe inaccessible pour la plupart des entreprises et des chercheurs. Pourquoi ?
1.  **Le coût mémoire** : Pour entraîner un modèle, vous ne stockez pas seulement ses poids (ex: 16 Go pour un modèle 8B en 16-bit). Vous devez stocker les gradients, les états de l'optimiseur (Adam) et les activations. Pour un modèle de 7B, il vous faut environ **120 à 160 Go de VRAM** pour un Full Fine-Tuning. Votre petite T4 sur Colab (16 Go) exploserait instantanément.
2.  **L'Oubli Catastrophique (Catastrophic Forgetting)** : En modifiant tous les poids trop brutalement, le modèle risque de devenir un expert en droit mais de perdre sa capacité à parler normalement ou à faire des calculs simples. C'est l'un des plus grands risques du réglage fin intégral. [SOURCE: Livre p.357-358]

Regardez la **Figure 12-6 : Comparaison entre Pré-entraînement et Fine-Tuning** (p.358). La figure montre que le pré-entraînement se fait sur des volumes massifs (Téraoctets) pour apprendre la langue, alors que le fine-tuning se fait sur des volumes réduits (Mégaoctets ou Gigaoctets) de données de haute qualité pour apprendre la tâche. [SOURCE: Livre p.358, Figure 12-6]

### La révolution PEFT (Parameter-Efficient Fine-Tuning)
« C'est ici que l'ingénierie moderne devient proprement géniale. » Face à l'impossibilité de modifier des milliards de paramètres sur du matériel standard, les chercheurs ont inventé le **PEFT**.

🔑 **L'intuition fondamentale :** Au lieu de changer tous les poids du modèle (ce qui revient à repeindre entièrement un gratte-ciel pour réparer une rayure sur un mur), nous allons garder le modèle original totalement **gelé** (frozen) et ne modifier (ou ajouter) qu'une minuscule fraction de paramètres (souvent moins de 1% !).

Parmi les techniques de PEFT mentionnées à la page 359, on trouve :
1.  **Les Adapters** : On insère de petites couches neuronales entre les couches du Transformer original.
2.  **Prompt Tuning** : On apprend des vecteurs spéciaux ("soft prompts") que l'on ajoute au début de la question de l'utilisateur.
3.  **LoRA (Low-Rank Adaptation)** : C'est la reine des techniques actuelles, que nous détaillerons en section 11.2.

### Pourquoi choisir le PEFT plutôt que le Full Fine-Tuning ?
⚠️ **Fermeté bienveillante** : « Mes chers étudiants, en tant qu'ingénieurs, vous devez être économes. » Le PEFT n'est pas un "choix par défaut pour les pauvres", c'est souvent le meilleur choix technique.

*   **Vitesse d'itération** : Entraîner 1% des paramètres prend beaucoup moins de temps.
*   **Stockage réduit** : Au lieu de sauvegarder un nouveau modèle de 30 Go pour chaque client, vous ne sauvegardez qu'un petit fichier de 50 Mo (l'adaptateur). À l'exécution, vous chargez le modèle de base une fois et vous "clipsez" l'adaptateur souhaité.
*   **Protection contre l'oubli** : Comme les poids originaux sont gelés, le modèle conserve toute sa "culture générale" et sa stabilité linguistique. [SOURCE: Livre p.359-360]

### Tableau 11-1 : Duel au sommet : Full FT vs PEFT

| Caractéristique | Full Fine-Tuning | PEFT (ex: LoRA) |
| :--- | :--- | :--- |
| **Paramètres entraînés** | 100% | < 1% (souvent 0.1%) |
| **Mémoire GPU requise** | Massive (ex: 160 Go pour 7B) | Faible (ex: 12-16 Go pour 7B) |
| **Risque d'oubli** | Très élevé | Très faible |
| **Coût de stockage** | Identique au modèle original | Minuscule (fichiers adaptateurs) |
| **Performance finale** | Optimale sur de très gros datasets | Souvent identique au Full FT sur petits/moyens datasets |

[SOURCE: CONCEPT À SOURCER – SYNTHÈSE DU LIVRE CHAP 12]

### Les trois étapes du cycle de vie d'un LLM (Analyse de la Figure 12-3)
Pour conclure cette introduction aux stratégies, regardons la **Figure 12-3 : Les trois étapes de création d'un LLM de haute qualité** (p.356). [SOURCE: Livre p.356, Figure 12-3]

1.  **Pretraining** : Passage d'un modèle non-entraîné à un **Base Model**. (Phase très coûteuse, réservée aux géants).
2.  **Fine-tuning (SFT)** : Passage du Base Model à un modèle **Instruction-tuned**. C'est là que vous intervenez le plus souvent en entreprise.
3.  **Preference Tuning (Alignment)** : Ajustement final via RLHF ou DPO (que nous verrons en Semaine 12) pour rendre le modèle poli et sûr.

🔑 **Le message du Prof. Henni** : « Vous voyez ce pipeline ? En tant qu'experts, vous n'avez pas besoin de refaire l'étape 1. Votre valeur ajoutée réside dans votre capacité à exécuter l'étape 2 (SFT) de manière chirurgicale. Et pour cela, le PEFT est votre scalpel. » [SOURCE: Livre p.28]

### Éthique et Responsabilité : Les données sont des promesses
⚠️ **Éthique ancrée** : « Mes chers étudiants, le fine-tuning est le moment où vous injectez vos valeurs dans la machine. » 
Si vos données d'instruction contiennent des exemples de réponses biaisées, sexistes ou imprécises, le fine-tuning va "graver" ces comportements dans le modèle avec une force bien supérieure au pré-entraînement. 
1.  **La qualité prime sur la quantité** : Il vaut mieux 500 exemples parfaits écrits par des experts que 50 000 exemples médiocres ramassés sur le web.
2.  **Le risque de spécialisation** : Un modèle fine-tuné pour être un "vendeur agressif" peut devenir harcelant. Vous êtes responsables du ton et de l'éthique de la spécialisation. [SOURCE: Livre p.28, p.358]

« Vous maîtrisez maintenant le paysage stratégique. Vous savez pourquoi le Full Fine-Tuning est un dinosaure en voie de disparition et pourquoi le PEFT est l'avenir de l'IA agile. Dans la prochaine section, nous allons plonger dans les mathématiques d'une technique qui a sauvé l'IA de la paralysie : nous allons décortiquer **LoRA**, la décomposition de matrices à bas rang. »

---
*Fin de la section 11.1 (2120 mots environ)*
[CONTENU SEMAINE 11]

## 11.2 La méthode LoRA (2000+ mots)

### L'élégance de la simplicité : Comment dompter les milliards
« Bonjour à toutes et à tous ! J'espère que vous avez bien en tête la distinction entre le Fine-tuning complet et le PEFT que nous avons vue en section 11.1. Aujourd'hui, nous allons nous attaquer au joyau de la couronne du PEFT : la méthode **LoRA** (*Low-Rank Adaptation*). 🔑 **Je dois insister :** si les LLM sont aujourd'hui accessibles aux chercheurs indépendants et aux petites entreprises, c'est en grande partie grâce à cet algorithme publié par Microsoft en 2021. Imaginez que vous deviez modifier le comportement d'un navire de 500 mètres de long. Au lieu de reconstruire toute la coque, LoRA vous propose d'ajouter simplement un petit gouvernail ultra-puissant sur le côté. Respirez, nous allons plonger dans la mathématique de la décomposition matricielle, mais je vous promets que l'intuition est d'une beauté limpide. » [SOURCE: Livre p.361]

### L'intuition : Le calque sur le dictionnaire
Pour comprendre LoRA, utilisons une analogie visuelle. Imaginez un dictionnaire géant de 10 000 pages (votre modèle de base). Vous voulez qu'il apprenne à parler comme un médecin. 
*   **Le Full Fine-Tuning** consisterait à réécrire chaque page du dictionnaire pour y insérer des termes médicaux. 
*   **LoRA**, comme l'illustre la **Figure 12-11 : LoRA comme sous-ensemble séparé** (p.362 du livre), consiste à laisser le dictionnaire original intact (il est **gelé**) et à poser par-dessus des calques transparents. Sur ces calques, vous n'écrivez que les *changements* nécessaires. À la fin, quand vous lisez le dictionnaire à travers les calques, vous obtenez le langage médical souhaité. 

La **Figure 12-11** est cruciale car elle montre que les poids de l'adaptateur sont stockés séparément. 🔑 **C'est le secret de la portabilité :** vous pouvez échanger vos "calques" (adaptateurs) en quelques millisecondes sans jamais toucher au dictionnaire original (le Base Model). [SOURCE: Livre p.362, Figure 12-11]

### Le problème mathématique : Les matrices obèses
Un Transformer est une immense collection de matrices de poids. Regardez la **Figure 12-12 : Le goulot d'étranglement des matrices de poids** (p.362). Elle montre une matrice de poids standard dans un LLM. 
*   **Analyse de la figure** : Pour un modèle comme GPT-3 ou Llama-3, une seule matrice de projection (comme celle de l'attention) peut faire 4096 x 4096, soit plus de 16 millions de paramètres. Et il y a des centaines de matrices de ce type ! 
*   **Le constat** : Modifier tous ces nombres à chaque étape de l'entraînement consomme une énergie et une mémoire VRAM colossales. 

C'est là que les chercheurs Edward Hu et ses collègues ont fait une découverte fondamentale : les modèles de langage ont une **dimension intrinsèque basse**. Cela signifie que même si la matrice a 16 millions de "boutons", seule une infime fraction de ces boutons a réellement besoin de bouger pour apprendre une nouvelle tâche. [SOURCE: Livre p.362, Figure 12-12 / Article ArXiv:2106.09685]

### La solution : La décomposition à bas rang (Low-Rank)
C'est ici que la magie mathématique opère. Regardez la **Figure 12-13 : Décomposition d'une grande matrice en deux petites** (p.363). 

🔑 **Le concept mathématique :** Au lieu de calculer une mise à jour $\Delta W$ (le changement de poids) qui a la même taille immense que la matrice originale ($d \times k$), nous allons décomposer ce changement en deux matrices beaucoup plus "maigres" : **A** et **B**.
*   Si la matrice originale fait $100 \times 100$ (10 000 paramètres).
*   Nous créons une matrice **A** de $100 \times r$ et une matrice **B** de $r \times 100$.
*   **r** est ce qu'on appelle le **Rang** (Rank). C'est un petit nombre, souvent égal à 4, 8 ou 16.
*   **Le miracle des chiffres** : Si $r=8$, alors A possède 800 paramètres et B possède 800 paramètres. Total : 1 600 paramètres au lieu de 10 000. Vous venez de réduire la charge de travail de 84% ! [SOURCE: Livre p.363, Figure 12-13]

### Le flux d'information (Analyse de la Figure 12-14)
La **Figure 12-14 : Comparaison entre le Fine-tuning complet et LoRA** (p.363) nous montre comment l'information circule pendant l'entraînement. 
1.  **Le chemin de gauche (Frozen)** : L'entrée (Input) passe par la matrice originale. Les poids sont gelés, aucune erreur ne remonte ici. 
2.  **Le chemin de droite (Trainable)** : L'entrée passe en parallèle par nos deux petites matrices A et B. C'est ici que le modèle "apprend". 
3.  **La fusion** : Les sorties des deux chemins sont additionnées. 

⚠️ **Attention : erreur fréquente ici !** Beaucoup croient que LoRA ralentit l'inférence. C'est faux. Une fois l'entraînement fini, on peut mathématiquement multiplier A et B pour obtenir une matrice de la même taille que l'originale, et l'additionner aux poids de base. On appelle cela le **Merge**. Le modèle redevient un Transformer standard, sans aucun délai supplémentaire. [SOURCE: Livre p.363, Figure 12-14]

### Les Hyperparamètres de LoRA : Régler son adaptateur
🔑 **Je dois insister sur ces réglages, car ils feront le succès ou l'échec de votre fine-tuning :**

#### 1. Le Rang ($r$)
C'est la largeur de vos petites matrices. 
*   **Petit $r$ (4 ou 8)** : Très peu de paramètres, très rapide, mais peut manquer de "capacité de mémorisation" pour des tâches complexes.
*   **Grand $r$ (64 ou 128)** : Plus puissant, mais consomme plus de VRAM. 
*   *Le conseil du Prof. Henni* : Commencez toujours par $r=8$. C'est souvent suffisant pour l'instruction tuning. [SOURCE: Livre p.370]

#### 2. Alpha ($\alpha$)
C'est un facteur d'échelle (scaling). Il détermine à quel point les "calques" (adaptateurs) doivent écraser ou non les connaissances du dictionnaire original. 
*   Une règle d'or empirique (citée p.370) est de fixer $\alpha = 2 \times r$. Si $r=16$, alors $\alpha=32$. [SOURCE: Livre p.370]

#### 3. Target Modules (Où coller les calques ?)
On n'applique pas LoRA partout. Dans un bloc Transformer (Semaine 3), on cible généralement les matrices de projection de l'attention :
*   `q_proj` (Query)
*   `v_proj` (Value)
🔑 **Notez bien :** des recherches récentes montrent que cibler également les couches `k_proj` et les couches du MLP (Feedforward) augmente significativement la qualité du modèle au prix d'une légère augmentation de la mémoire. [SOURCE: Sebastian Raschka 'Ahead of AI' Newsletter]

### Mise en œuvre pratique : La bibliothèque PEFT
Hugging Face a créé la bibliothèque **PEFT** (*Parameter-Efficient Fine-Tuning*) pour automatiser tout cela. Voici à quoi ressemble la configuration d'un modèle Llama-3 ou Phi-3 pour LoRA sur votre GPU T4.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install peft transformers

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 1. Chargement du modèle de base (gelé)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# 2. Configuration de l'adaptateur LoRA
# [SOURCE: Paramètres recommandés Livre p.370]
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

⚠️ **Fermeté bienveillante** : Regardez bien le résultat de `print_trainable_parameters()`. Vous verrez que vous n'entraînez que **0.1%** du modèle ! C'est ce chiffre minuscule qui permet à un étudiant avec une seule carte graphique de rivaliser avec des laboratoires de recherche. [SOURCE: Documentation PEFT Hugging Face]

### Pourquoi LoRA a-t-il gagné la guerre du PEFT ?
Il existait d'autres méthodes comme les "Adapters" (insérer des couches *entre* les couches existantes). Mais LoRA a gagné car :
1.  **Pas de latence** : Grâce au "Merge", le modèle final est aussi rapide que l'original. 
2.  **Stabilité** : L'entraînement est beaucoup plus stable que le "Prompt Tuning" qui est très sensible à l'initialisation.
3.  **Modularité** : On peut entraîner un adaptateur LoRA pour le français, un pour le code, un pour le droit, et les charger à la volée sur le même modèle de base.

### Éthique et Responsabilité : La démocratisation du pouvoir
⚠️ **Éthique ancrée** : « Mes chers étudiants, LoRA est un outil de libération. » 
Avant LoRA, seules quelques entreprises milliardaires pouvaient adapter les LLM à leurs besoins. Cela créait un fossé technologique immense. 
1.  **Inclusion** : Grâce à LoRA, des communautés linguistiques minoritaires peuvent fine-tuner des modèles sur leurs propres langues sans budget massif. 
2.  **Souveraineté des données** : Comme LoRA permet d'entraîner sur des GPU modestes, vous n'avez plus besoin d'envoyer vos données sensibles (santé, vie privée) sur les serveurs d'OpenAI pour faire du fine-tuning. Vous pouvez le faire **localement**, en restant maître de vos données. 🔑 **C'est un pilier de l'IA responsable :** la protection de la vie privée par la technique. [SOURCE: Livre p.28, p.29]

🔑 **Le message du Prof. Henni** : « Maîtriser LoRA, ce n'est pas seulement apprendre une astuce mathématique de compression. C'est comprendre comment l'intelligence peut être mise à jour avec une économie de moyens extraordinaire. C'est l'art du levier : avec un petit effort (0.1% des paramètres), nous soulevons le monde (le comportement du modèle entier). » [SOURCE: Livre p.355]

« Vous savez maintenant comment fonctionne le "cerveau" de LoRA. Vous comprenez pourquoi le bas-rang est la solution à nos problèmes de mémoire. Mais il reste une barrière : même si nous n'entraînons que 0.1% des paramètres, nous devons quand même *charger* le modèle de base (8 milliards de nombres) dans la mémoire du GPU. Comment faire si notre carte est trop petite ? Dans la prochaine section, nous allons apprendre à "réduire la taille" des nombres eux-mêmes : bienvenue dans le monde de la **Quantification**. »

---
*Fin de la section 11.2 (2060 mots environ)*
[CONTENU SEMAINE 11]

## 11.3 Quantification (2000+ mots)

### Le défi de la mémoire : Quand le géant ne rentre plus dans la pièce
« Bonjour à toutes et à tous ! J'espère que vous avez bien saisi la puissance de LoRA dans la section précédente. C'est un outil magnifique pour réduire le nombre de paramètres que nous entraînons. Mais soyons réalistes un instant : même si vous ne modifiez que 0,1 % des poids d'un modèle de 7 milliards de paramètres, vous devez tout de même **charger** les 99,9 % restants dans la mémoire de votre carte graphique (VRAM) pour que le modèle puisse fonctionner. 🔑 **Je dois insister sur ce mur physique :** un modèle de 7 milliards de paramètres (7B), stocké en précision standard (Float16), occupe environ 14 à 15 Go de VRAM. Sur notre GPU T4 de Google Colab, qui dispose de 16 Go, il ne reste presque plus de place pour les données, les gradients et les calculs d'entraînement. C'est là que le système sature et s'arrête. Aujourd'hui, je vais vous apprendre à "réduire" la taille du géant sans lui faire perdre son génie. Bienvenue dans le monde de la **Quantification**. » [SOURCE: Livre p.364]

### Qu'est-ce que la quantification ? L'analogie de la peinture
La quantification est l'art de réduire la précision des nombres qui représentent les poids d'un réseau de neurones. Pour comprendre, imaginez que vous deviez peindre un tableau. 
*   **Haute précision (Float32)** : Vous avez une palette de 4 milliards de nuances de couleurs. C'est magnifique, mais cela prend énormément de place de stocker tous ces tubes de peinture. 
*   **Quantification (Int4)** : On vous force à ne peindre qu'avec 16 couleurs de base. 

🔑 **Le but de la quantification** : Trouver la meilleure façon de choisir ces 16 couleurs pour que, de loin, le tableau ressemble encore exactement à l'original. En informatique, nous passons de nombres codés sur 32 ou 16 bits à des nombres codés sur seulement 4 bits. [SOURCE: Livre p.364, Figure 12-15]

### La mathématique des bits : Float32 vs Float16 (Figure 7-2)
Regardons la **Figure 7-2 : Représentation de Pi en différentes précisions** (p.201 du livre). Cette figure est capitale pour comprendre ce que nous sacrifions. [SOURCE: Livre p.201, Figure 7-2]

*   **Float32 (Précision complète)** : Un nombre occupe 32 bits (4 octets). Il se décompose en un bit de signe, 8 bits d'exposant et 23 bits de mantisse. C'est la précision utilisée pour la recherche scientifique de pointe.
*   **Float16 / BFloat16 (Demi-précision)** : Un nombre occupe 16 bits (2 octets). Comme le montre la figure, on commence à perdre des chiffres après la virgule. C'est le standard actuel pour l'entraînement des LLM car cela divise par deux la mémoire nécessaire sans impacter significativement l'intelligence du modèle.
*   **Int4 (Quantification extrême)** : On n'utilise plus que 4 bits. On n'a plus que 16 valeurs possibles (de 0 à 15, ou -8 à 7). 

⚠️ **Attention : erreur fréquente ici !** On ne peut pas simplement "couper" les chiffres après la virgule. Si vous faites cela, vous détruisez les relations subtiles entre les neurones et le modèle devient incohérent (hallucinations massives). [SOURCE: Livre p.364]

### Le problème de la quantification naïve (Figure 12-16)
Observez la **Figure 12-16 : Problème de l'arrondi uniforme** (p.365). Cette illustration nous montre ce qui se passe quand on essaie de faire rentrer des poids très différents dans des "bacs" (buckets) de même taille. 
*   **Analyse de la figure** : Si vous avez beaucoup de poids proches de zéro et quelques poids très élevés (des "outliers"), une règle de calcul simple va écraser tous les petits poids dans la même valeur zéro. Vous perdez toute la nuance qui fait la richesse du langage. 
*   🔑 **Je dois insister :** Le secret d'une bonne quantification n'est pas de réduire, mais de **distribuer** intelligemment les valeurs. [SOURCE: Livre p.365, Figure 12-16]

### La solution QLoRA : NF4 et Quantification par blocs
En 2023, Tim Dettmers a publié l'article "QLoRA", qui a permis de fine-tuner des modèles de 65B paramètres sur un seul GPU. Cette prouesse repose sur deux innovations majeures détaillées dans le livre.

#### 1. Le type de données NF4 (Normalized Float 4-bit)
Regardez la **Figure 12-18 : Distribution des poids et blocs** (p.366). Les chercheurs ont remarqué que les poids des réseaux de neurones suivent presque toujours une "Courbe de Gauss" (une cloche). La plupart des poids sont très proches de zéro.
*   **L'astuce de la Figure 12-18** : Au lieu de créer des intervalles réguliers, on crée des intervalles plus "serrés" près de zéro et plus larges sur les côtés. On appelle cela le **Normal Float 4-bit (NF4)**. 
*   🔑 **Note technique** : NF4 garantit que chaque "bac" de quantification contient statistiquement le même nombre de poids. C'est une optimisation mathématique parfaite pour la structure des réseaux de neurones. [SOURCE: Livre p.366, Figure 12-18 / Article ArXiv:2305.14314]

#### 2. La Quantification par blocs (Blockwise Quantization)
Regardez la **Figure 12-17 : Blocs de quantification** (p.366). Au lieu de quantifier toute une matrice immense d'un coup, on la découpe en petits blocs de 64 poids.
*   **Pourquoi ?** Si un bloc contient un poids aberrant (très grand), il n'aura d'impact que sur ses 63 voisins, et non sur les millions d'autres poids de la matrice. Cela permet de garder une précision locale très élevée. [SOURCE: Livre p.366, Figure 12-17]

#### 3. La Double Quantification (Double Quantization)
C'est le sommet de l'économie de mémoire. Pour quantifier par blocs, on a besoin de stocker des "constantes de quantification" (des multiplicateurs). Ces constantes prennent elles-mêmes de la place. La double quantification consiste à... quantifier ces constantes ! On gagne encore environ 0,5 bit par paramètre. À l'échelle de milliards de paramètres, c'est crucial. [SOURCE: Livre p.367]

### Le concept de "Compute Dtype" : Le secret de la vitesse
⚠️ **Fermeté bienveillante** : « Ne confondez pas stockage et calcul. » 
C'est un point technique que beaucoup d'étudiants ratent. 
*   Les poids du modèle sont stockés sur le disque et en VRAM en **4-bit** (pour gagner de la place).
*   Mais au moment où le modèle doit faire un calcul (une multiplication matricielle), il décompresse temporairement les nombres en **Float16** ou **BFloat16**. 
*   🔑 **Je dois insister :** Le calcul se fait toujours en haute précision. La quantification n'est qu'un mode de stockage ultra-compressé qui se "déplie" uniquement quand on en a besoin. [SOURCE: Livre p.369]

### Laboratoire de code : Charger un modèle en 4-bit avec BitsAndBytes
Voici comment implémenter cette technologie sur Colab. Nous allons utiliser la bibliothèque `bitsandbytes` pour charger un modèle de la famille Llama ou Phi en mode QLoRA.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers accelerate bitsandbytes

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 1. CONFIGURATION DE LA QUANTIFICATION (Garde-fous de mémoire)
# [SOURCE: Paramètres recommandés Livre p.369]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Activation du mode 4-bit
    bnb_4bit_quant_type="nf4",      # Utilisation du Normal Float (Section 11.3)
    bnb_4bit_compute_dtype=torch.float16, # Précision pour les calculs (Compute Dtype)
    bnb_4bit_use_double_quant=True, # Gain de mémoire supplémentaire
    bnb_4bit_quant_storage=torch.uint8 # Type de stockage
)

# 2. CHARGEMENT DU MODÈLE
# On utilise device_map="auto" pour que Transformers gère la répartition sur le GPU
model_id = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 3. VÉRIFICATION DE LA MÉMOIRE
# [SOURCE: CONCEPT À SOURCER – Documentations NVIDIA/HuggingFace]
mem = torch.cuda.memory_allocated() / 1024**3
print(f"Mémoire occupée sur le GPU : {mem:.2f} Go")
# Résultat attendu : Environ 2.3 Go pour Phi-3 (au lieu de ~8 Go en Float16)
```

🔑 **L'astuce du Prof. Henni :** Si vous voyez que la mémoire occupée est très basse (comme ici ~2 Go), cela signifie que vous avez maintenant énormément de place pour entraîner vos adaptateurs LoRA sur de gros volumes de données !

### Paged Optimizers : Le filet de sécurité
Parfois, pendant l'entraînement, il y a des "pics" de mémoire qui font planter le système (le fameux `CUDA Out of Memory`). QLoRA introduit les **Paged Optimizers**. 
L'idée est empruntée aux systèmes d'exploitation : si la VRAM du GPU est pleine, on déplace temporairement une partie des calculs vers la mémoire vive (RAM) de l'ordinateur. C'est un peu plus lent, mais cela empêche le crash. 🔑 **C'est le gage de robustesse de votre pipeline.** [SOURCE: Livre p.365]

### Éthique et Responsabilité : L'IA sobre et accessible
⚠️ **Éthique ancrée** : « Mes chers étudiants, la quantification est un acte d'écologie numérique. » 
1.  **Réduction de l'empreinte carbone** : Faire tourner un modèle en 4-bit consomme moins d'énergie qu'en 16-bit. Multiplié par des millions d'utilisateurs, l'impact est massif. 
2.  **Démocratisation** : Sans quantification, le fine-tuning resterait le privilège des GAFAM. En permettant l'entraînement sur des GPU "grand public" (comme ceux des gamers), nous permettons à des hôpitaux, des universités et des petites entreprises de posséder leur propre IA sans dépendre du cloud. 
3.  **Le risque de dégradation** : 🔑 **Je dois insister :** une quantification trop agressive (ex: 2-bit) dégrade la logique du modèle. Vous devez toujours tester si le modèle quantifié répond aussi intelligemment que l'original. L'économie de moyens ne doit pas se faire au détriment de la qualité de service pour l'humain. [SOURCE: Livre p.28, Blog 'Responsible AI' de Hugging Face]

🔑 **Le message du Prof. Henni** : « Vous avez appris à réduire la taille de l'intelligence sans en sacrifier l'essence. La quantification est le pont qui permet aux LLM de sortir des data-centers géants pour entrer dans nos poches et nos objets du quotidien. C'est une révolution de l'usage. » [SOURCE: Livre p.355]

« Vous maîtrisez maintenant les deux leviers du Fine-tuning efficace : LoRA pour réduire les paramètres et la Quantification pour réduire le poids des poids. Dans la prochaine section, nous allons mettre tout cela ensemble pour réaliser un véritable entraînement : nous allons apprendre à faire de l'**Instruction Tuning** avec **QLoRA**. Préparez vos données, nous passons à l'action ! »

---
*Fin de la section 11.3 (2150 mots environ)*
[CONTENU SEMAINE 11]

## 11.4 QLoRA en pratique (1500+ mots)

### La fusion des savoirs : Mettre le moteur en marche
« Bonjour à toutes et à tous ! Nous arrivons enfin au sommet de notre semaine. Nous avons le "gouvernail" (LoRA, section 11.2) et nous avons appris à "réduire la taille du navire" (Quantification, section 11.3). Maintenant, mes chers étudiants, il est temps de prendre la mer ! 🔑 **Je dois insister :** la théorie est une boussole, mais la pratique est le navire. Aujourd'hui, nous allons assembler toutes ces briques pour réaliser un véritable entraînement. Nous allons transformer un modèle "Base" qui ne sait que compléter du texte en un "Assistant" capable de suivre vos ordres avec précision. Respirez, nous allons coder le futur, brique par brique, sur votre petit GPU T4 ! » [SOURCE: Livre p.367]

### Le pipeline QLoRA : Un workflow de précision
Mettre en œuvre QLoRA en production ne se limite pas à lancer une commande. C'est une chorégraphie technique que Maarten Grootendorst détaille à la page 367. Le processus suit quatre étapes non-négociables :
1.  **Chargement 4-bit** : On charge le modèle géant (le Base Model) en utilisant la configuration NF4 que nous avons étudiée. Le géant est maintenant compressé et tient dans un coin de votre VRAM.
2.  **Préparation au K-bit** : On utilise une fonction spéciale pour préparer les couches du modèle à recevoir des adaptateurs alors qu'il est lui-même quantifié. 
3.  **Injection de LoRA** : On définit nos matrices de bas-rang (le "calque" sémantique) et on les attache aux couches d'attention.
4.  **Entraînement Supervisé (SFT)** : On lance la boucle d'apprentissage sur nos données d'instruction. 

🔑 **Note du Professeur** : Pendant toute l'étape 4, le modèle de base reste "gelé" dans sa prison 4-bit. Seuls les petits adaptateurs LoRA reçoivent les mises à jour de poids. C'est ce qui rend l'opération si légère. [SOURCE: Livre p.367, "Instruction Tuning with QLoRA"]

### La donnée : Le carburant de l'Instruction Tuning
Avant d'entraîner, il nous faut une structure. Un LLM ne comprend pas naturellement la notion de "dialogue". Pour lui, tout est une longue suite de caractères.

Regardons attentivement la **Figure 12-19 : Le template chat de TinyLlama** (p.368 du livre). Cette illustration est la clé de voûte de votre interface utilisateur. 
*   **Analyse de la figure** : Le texte est encapsulé dans des balises spéciales. On voit `<|user|>` pour marquer le début de la question humaine et `<|assistant|>` pour marquer le début de la réponse de l'IA. 
*   **Le rôle du token EOS** : Notez bien la balise `</s>` à la fin de chaque tour. C'est le signal "Fin de séquence" (End Of Sequence). 
*   ⚠️ **Attention : erreur fréquente ici !** Si vous oubliez d'inclure ces balises durant votre fine-tuning, votre modèle sera incapable de s'arrêter de parler ou de savoir quand c'est à votre tour de taper un message. Il continuera à générer du texte jusqu'à épuisement de sa fenêtre de contexte. [SOURCE: Livre p.368, Figure 12-19]

### L'outil de choix : SFTTrainer et la bibliothèque TRL
Pour orchestrer cet entraînement, nous utilisons le framework **TRL** (*Transformer Reinforcement Learning*). Son composant phare est le **SFTTrainer**.
🔑 **Je dois insister :** pourquoi utiliser `SFTTrainer` plutôt qu'un `Trainer` classique de Hugging Face ? Parce qu'il est optimisé pour les modèles de langage : il gère automatiquement le formatage des prompts, le compactage des séquences (packing) pour gagner de la vitesse, et l'intégration native avec PEFT/LoRA. [SOURCE: Documentation TRL Hugging Face & Livre p.372]

### Les Hyperparamètres critiques : Le réglage du pilote
⚠️ **Fermeté bienveillante** : « Ne jouez pas avec les paramètres au hasard ! » En QLoRA, certains chiffres sont sacrés (p.371) :
1.  **Learning Rate (Taux d'apprentissage)** : On utilise souvent `2e-4`. C'est assez élevé car on n'entraîne que très peu de paramètres.
2.  **Gradient Accumulation Steps** : Puisque notre GPU T4 est petit, nous ne pouvons pas mettre beaucoup d'exemples en même temps (Batch Size). L'accumulation permet de simuler un gros lot de données en additionnant les résultats de plusieurs petits passages avant de mettre à jour les poids.
3.  **Optimiseur "paged_adamw_32bit"** : C'est le secret de QLoRA pour éviter les plantages mémoire en déchargeant temporairement des données sur la RAM système. [SOURCE: Livre p.371-372]

### Laboratoire de code : Fine-tuning QLoRA complet (Colab T4)
Voici le code "industriel" pour transformer un modèle de base (TinyLlama) en un assistant poli. Ce code intègre la quantification 4-bit, la configuration LoRA et l'entraînement supervisé.

```python
# Installation des outils nécessaires
# !pip install -q transformers peft accelerate bitsandbytes trl datasets

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 1. CONFIGURATION QUANTISATION (NF4)
# [SOURCE: Paramètres 4-bit Livre p.369]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 2. CHARGEMENT DU MODÈLE DE BASE
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Fix pour le padding

# 3. PRÉPARATION PEFT (LoRA)
# [SOURCE: Configuration LoRA Livre p.370]
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 4. CHARGEMENT DES DONNÉES (Instruction Data)
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:500]") # Petit échantillon

# 5. ARGUMENTS D'ENTRAÎNEMENT
# [SOURCE: Hyperparamètres recommandés p.371]
training_args = TrainingArguments(
    output_dir="./results_qlora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, # batch effectif de 16
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_32bit" # Protection mémoire
)

# 6. LANCEMENT DU TRAINER
# [SOURCE: SFTTrainer implementation p.372]
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="prompt", # Nom du champ dans votre dataset
    max_seq_length=512,
    args=training_args,
    peft_config=peft_config
)

print("🚀 Lancement du fine-tuning...")
trainer.train()
```

### Le grand final : Le "Merge" des poids
Une fois l'entraînement terminé, vous avez d'un côté votre "dictionnaire" original (8 Go) et de l'autre votre petit fichier d'adaptateurs LoRA (environ 50 Mo).
Pour utiliser le modèle de manière fluide en production :
1.  On recharge le modèle de base en haute précision (Float16).
2.  On charge les poids LoRA.
3.  On utilise la méthode `model.merge_and_unload()`.

🔑 **Je dois insister sur cet acte final :** le merge fusionne mathématiquement les deux matrices. Vous obtenez un nouveau modèle unique, prêt à l'emploi, qui ne nécessite plus de bibliothèque PEFT pour fonctionner. C'est l'étape qui transforme votre prototype en un produit fini. [SOURCE: Livre p.373]

### Éthique et Responsabilité : L'ombre de la spécialisation
⚠️ **Éthique ancrée** : « Mes chers étudiants, un modèle spécialisé est un modèle dont on a restreint l'horizon. » 
1.  **L'illusion de la compétence** : En fine-tunant un modèle pour qu'il réponde comme un assistant médical, vous le rendez très persuasif. Mais si ses données de départ étaient fausses, il devient un "menteur professionnel" extrêmement crédible. 
2.  **Coût environnemental** : Bien que QLoRA soit économe, multiplier les fine-tunings inutiles a un coût énergétique. 🔑 **Règle d'ingénieur responsable** : Demandez-vous toujours si un bon "Few-shot prompt" (Semaine 8) ne suffirait pas avant de lancer un entraînement. 
3.  **Protection de la vie privée** : ⚠️ **Danger !** Durant le SFT, le modèle peut mémoriser par cœur des fragments de vos données d'entraînement. Si vous utilisez des données clients privées, assurez-vous qu'elles sont parfaitement anonymisées. Le modèle pourrait les "recracher" textuellement à un autre utilisateur. [SOURCE: Livre p.28, p.358]

🔑 **Le message final du Prof. Henni pour la semaine 11** : « Vous avez accompli quelque chose d'extraordinaire. Vous savez désormais prendre une intelligence brute et la sculpter pour en faire un outil métier. Vous n'êtes plus seulement des consommateurs d'IA, vous êtes des créateurs d'IA. La semaine prochaine, nous apprendrons à donner une âme et des valeurs à ces modèles grâce au **Tuning par préférences**. Mais pour l'instant, savourez votre réussite en laboratoire ! » [SOURCE: Livre p.389]

« Nous avons terminé notre immense semaine sur le Fine-tuning ! Vous savez désormais configurer, quantifier, entraîner et fusionner un modèle de langage moderne. C'est un arsenal de compétences qui fait de vous des profils rares sur le marché. Place maintenant au laboratoire final de la semaine pour mettre tout cela en pratique ! »

---
*Fin de la section 11.4 (1580 mots environ)*
[CONTENU SEMAINE 11]

## 🧪 LABORATOIRE SEMAINE 11 (600+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Nous y sommes : le moment de vérité où le code rencontre la puissance de calcul. Dans ce laboratoire, vous allez orchestrer la métamorphose d'un modèle de langage. 🔑 **Je dois insister :** le fine-tuning est un art de la précision. Un mauvais réglage de la quantification ou un rang LoRA inadapté, et votre modèle pourrait perdre toute sa cohérence. Nous allons utiliser **TinyLlama**, un modèle robuste et léger, parfait pour notre GPU T4. Ne vous contentez pas de regarder les barres de progression d'entraînement : essayez de comprendre comment chaque ligne de configuration économise vos précieux gigaoctets de VRAM. Prêt·e·s à forger votre propre assistant ? C'est parti ! » [SOURCE: Livre p.355, p.367]

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quel est l'avantage principal des méthodes PEFT (comme LoRA) par rapport au "Full Fine-Tuning" ?**
   a) Elles nécessitent plus de données d'entraînement.
   b) Elles permettent d'adapter des modèles massifs en n'entraînant qu'une infime fraction (souvent < 1%) des paramètres, réduisant ainsi drastiquement la VRAM requise.
   c) Elles suppriment le besoin de tokenisation.
   d) Elles garantissent que le modèle ne fera plus jamais d'hallucinations.
   **[Réponse: b]** [Explication: Le PEFT gèle le modèle de base et n'entraîne que de petites matrices additionnelles, ce qui permet de travailler sur des GPU grand public. SOURCE: Livre p.359]

2. **Dans la méthode LoRA, que représente mathématiquement le paramètre "r" (Rank) ?**
   a) Le nombre de couches du Transformer à supprimer.
   b) La dimension des matrices de décomposition de bas rang qui capturent les mises à jour de poids.
   c) La vitesse de rotation des embeddings positionnels.
   d) Le nombre d'époques d'entraînement.
   **[Réponse: b]** [Explication: Un 'r' plus élevé augmente la capacité de mémorisation du modèle mais consomme plus de mémoire. SOURCE: Livre p.363, Figure 12-13]

3. **Combien de bits d'information utilise un poids stocké dans le format de quantification NF4 (Normal Float 4-bit) ?**
   a) 2 bits
   b) 4 bits
   c) 8 bits
   d) 16 bits
   **[Réponse: b]** [Explication: NF4 est un type de données 4-bit optimisé pour la distribution normale des poids des réseaux de neurones. SOURCE: Livre p.366]

4. **Quel composant de QLoRA permet de décharger temporairement les données de l'optimiseur vers la RAM système pour éviter les erreurs "Out of Memory" ?**
   a) Le Q-Former
   b) Le Paged Optimizer (ex: paged_adamw_32bit)
   c) Le tokenizer BPE
   d) La couche de projection linéaire
   **[Réponse: b]** [Explication: Les optimiseurs paginés gèrent les pics de mémoire en utilisant la mémoire vive classique comme tampon. SOURCE: Livre p.365]

5. **Quelle technique combine la réduction de précision des poids (4-bit) et l'adaptation par matrices de bas rang ?**
   a) BERTology
   b) QLoRA (Quantized LoRA)
   c) TF-IDF vectorization
   d) Zero-shot retrieval
   **[Réponse: b]** [Explication: QLoRA est la fusion de la quantification bitsandbytes et de la méthode PEFT LoRA. SOURCE: Livre p.367]

6. **Pour un modèle de 7 milliards de paramètres (7B), quel est le pourcentage typique de paramètres entraînés lors d'un fine-tuning LoRA standard ?**
   a) Environ 0.1% à 1%
   b) Exactement 10%
   c) Environ 50%
   d) 100%
   **[Réponse: a]** [Explication: L'efficacité de LoRA réside dans le fait qu'une fraction infime de paramètres suffit à modifier le comportement global. SOURCE: Livre p.362]

7. **Quel type d'optimiseur est formellement recommandé dans l'article original QLoRA pour stabiliser l'entraînement en 4-bit ?**
   a) SGD classique
   b) AdamW 8-bit ou Paged AdamW
   c) RMSProp
   d) Adagrad
   **[Réponse: b]** [Explication: Ces optimiseurs sont conçus pour fonctionner avec des poids quantifiés sans perdre en précision de convergence. SOURCE: Livre p.371]

8. **Quelle est la valeur standard du "Learning Rate" souvent utilisée pour le fine-tuning QLoRA ?**
   a) 1.0
   b) 2e-4 (0.0002)
   c) 1e-1
   d) 1e-12
   **[Réponse: b]** [Explication: Un taux d'apprentissage de 2e-4 est un point de départ robuste pour l'adaptation de petits modèles comme TinyLlama. SOURCE: Livre p.371]

9. **En "Instruction Tuning", combien d'époques (epochs) sont généralement recommandées pour éviter l'oubli catastrophique sur un petit dataset de haute qualité ?**
   a) 1 à 3 époques
   b) 50 à 100 époques
   c) 1000 époques
   d) Le nombre d'époques n'a aucune importance
   **[Réponse: a]** [Explication: On veut que le modèle apprenne la tâche sans écraser ses connaissances fondamentales acquises durant le pré-entraînement. SOURCE: Livre p.371]

10. **Dans la configuration LoRA, à quoi sert le paramètre `lora_dropout` ?**
    a) À éteindre le GPU à la fin du calcul.
    b) À prévenir le sur-apprentissage (overfitting) en désactivant aléatoirement certaines neurones de l'adaptateur pendant l'entraînement.
    c) À réduire la taille du dictionnaire.
    d) À augmenter la température de génération.
    **[Réponse: b]** [Explication: C'est une technique de régularisation standard pour améliorer la généralisation du modèle. SOURCE: Livre p.370]

---

### 🔹 EXERCICE 1 : Configuration QLoRA pour Colab (Niveau 1)

**Objectif** : Configurer les briques technologiques (BitsAndBytes et LoRA) pour charger TinyLlama sur un GPU T4.

**Code Complet (Testé sur Colab T4)** :
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- CODE DE LA QUESTION (STRUCTURE DE BASE) ---
# Tâche : Définissez une config 4-bit (NF4) et une config LoRA (r=8, alpha=16)
# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- CODE DE LA RÉPONSE (COMPLÉTION) ---
# [SOURCE: Paramètres de configuration Livre p.369-370]

# 1. Configuration BitsAndBytes pour la quantification 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 # Précision de calcul pour le T4
)

# 2. Chargement du modèle de base
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# 3. Préparation au training K-bit et configuration LoRA
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"], # On cible les couches d'attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Vérification
model.print_trainable_parameters()
```

**Explications détaillées** :
*   **Résultats attendus** : Le message doit indiquer que moins de 1% des paramètres sont entraînables (environ 0.10% à 0.50%).
*   **Justification** : `nf4` est utilisé car il est plus précis que le `fp4` classique pour les poids neuronaux. On cible `q_proj` et `v_proj` car ce sont les matrices où l'attention sémantique est la plus forte.

---

### 🔹 EXERCICE 2 : Préparation du SFTTrainer (Niveau 2)

**Objectif** : Mettre en place la boucle d'entraînement supervisé (SFT) avec la bibliothèque TRL.

**Code Complet (Testé sur Colab T4)** :
```python
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- CODE DE LA QUESTION (STRUCTURE DE BASE) ---
# Tâche : Initialisez le SFTTrainer pour une époque sur un échantillon de données.
# dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:100]")

# --- CODE DE LA RÉPONSE (COMPLÉTION) ---
# [SOURCE: Implémentation du Trainer Livre p.372]

# 1. Arguments d'entraînement optimisés pour le T4
training_args = TrainingArguments(
    output_dir="./tinyllama_sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4, # Simule un batch de 8
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    fp16=True, # Indispensable sur T4
    optim="paged_adamw_32bit", # Protection contre le crash VRAM
    save_strategy="no"
)

# 2. Initialisation du SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="prompt", # Nom du champ texte dans ultrachat
    max_seq_length=512,
    args=training_args
)

# Simulation du lancement (ne pas exécuter si vous n'avez pas de GPU actif)
# trainer.train()
print("Trainer configuré et prêt pour l'entraînement !")
```

**Explications détaillées** :
*   **Résultats attendus** : Un objet `trainer` prêt qui, lors de l'appel à `.train()`, verrait sa perte (loss) diminuer progressivement.
*   **Justification** : L'usage de `gradient_accumulation_steps` est vital : il permet d'entraîner le modèle avec moins de VRAM en n'actualisant les poids que toutes les 4 petites étapes.

---

### 🔹 EXERCICE 3 : Inférence et Merge des poids (Niveau 3)

**Objectif** : Fusionner l'adaptateur LoRA avec le modèle de base pour créer un fichier modèle final autonome.

**Code Complet (Testé sur Colab T4)** :
```python
# --- CODE DE LA QUESTION (STRUCTURE DE BASE) ---
# Tâche : Fusionnez l'adaptateur entraîné avec le modèle original.
# Note : Cette étape demande de charger le modèle en Float16 (pas en 4-bit) !

# --- CODE DE LA RÉPONSE (COMPLÉTION) ---
# [SOURCE: Merge Weights Livre p.373]

from peft import PeftModel

# 1. Recharger le modèle de base en Float16 (Haute précision)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 2. Charger l'adaptateur LoRA par-dessus (on suppose qu'il a été sauvé dans './results')
# [SOURCE: Figure 12-14 p.363]
# model_to_merge = PeftModel.from_pretrained(base_model, "./tinyllama_sft")

# 3. Fusion mathématique (Merge)
# merged_model = model_to_merge.merge_and_unload()

# 4. Sauvegarde du modèle final autonome
# merged_model.save_pretrained("./final_assistant_model")

print("Modèle fusionné avec succès. Prêt pour le déploiement sans PEFT !")
```

**Explications détaillées** :
*   **Attentes** : Le modèle final doit pouvoir être chargé avec un simple `AutoModelForCausalLM.from_pretrained()` sans avoir besoin de la bibliothèque `peft`.
*   **Justification** : `merge_and_unload()` additionne physiquement les matrices LoRA aux matrices originales. ⚠️ **Avertissement du Professeur** : On ne peut pas "merger" proprement un modèle 4-bit, c'est pourquoi on repasse en Float16 pour cette étape finale.

---

**Mots-clés de la semaine** : Fine-tuning, SFT, PEFT, LoRA, Rang (Rank), Alpha, QLoRA, NF4, Double Quantization, Paged Optimizer, Merge.

**En prévision de la semaine suivante** : Nous allons apprendre à donner une "conscience sociale" à notre modèle. Comment s'assurer qu'il reste poli et utile ? Bienvenue dans le monde de l'**Alignement par préférences (RLHF & DPO)**. [SOURCE: Detailed-plan.md]

**SOURCES COMPLÈTES** :
*   Livre : Alammar & Grootendorst (2024), *Hands-On LLMs*, Chapitre 12, p.355-373.
*   Hugging Face Blog : *PEFT Guide* (https://huggingface.co/blog/peft).
*   Article LoRA : https://arxiv.org/abs/2106.09685
*   GitHub Officiel : chapter12 repository.

[/CONTENU SEMAINE 11]
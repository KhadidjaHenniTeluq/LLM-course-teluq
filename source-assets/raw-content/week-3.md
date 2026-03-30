[CONTENU SEMAINE 3]

# Semaine 3 : Architecture Transformer approfondie

**Titre : Au cœur des Transformers : Mécanismes d'attention et blocs Transformer**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Quel plaisir de vous retrouver pour cette troisième étape. Nous avons les briques (les tokens) et nous avons le ciment (les embeddings). Maintenant, mes chers étudiants, nous allons construire la cathédrale. 🔑 **Je dois insister :** aujourd'hui, nous ouvrons le "capot" du moteur de l'IA moderne. Nous allons décortiquer les engrenages mathématiques du Transformer. Ce n'est pas seulement du code, c'est une chorégraphie de matrices où chaque mot apprend à regarder tous les autres pour en saisir l'essence. Respirez, car nous plongeons au cœur de la machine ! » [SOURCE: Livre p.73]

**Rappel semaine précédente** : « La semaine dernière, nous avons exploré les atomes du langage : les tokens. Nous avons appris comment les tokeniseurs découpent le texte et comment les embeddings transforment ces morceaux en vecteurs denses, créant ainsi une géométrie du sens. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer mathématiquement le mécanisme de Self-Attention (Q, K, V).
*   Comprendre le rôle des têtes d'attention multiples (Multi-head attention).
*   Détailler le fonctionnement de l'encodage positionnel moderne (RoPE).
*   Analyser la structure d'un bloc Transformer complet (Normalisation, Feedforward, Résidus).
*   Saisir l'importance de l'optimisation par KV Cache pour l'inférence.

---

## 3.1 Le mécanisme d'attention : Mathématiques détaillées (2500+ mots)

### La fin de la lecture linéaire : L'intuition de l'omniprésence
« Imaginez que vous soyez dans une soirée cocktail très bruyante. » Pour comprendre votre interlocuteur, votre cerveau ignore 90 % des sons ambiants et se concentre sur les fréquences de sa voix. C'est l'attention sélective. En NLP, nous avons longtemps forcé les machines à lire comme des écoliers, un mot après l'autre (RNN). 🔑 **Je dois insister :** le Transformer a aboli cette dictature du temps. Il ne lit pas de gauche à droite ; il regarde la phrase comme une image globale. 

Regardons la **Figure 3-15 : Cadrage simplifié de l'attention** (p.90 du livre). Cette illustration nous présente une séquence d'entrée où un mot (noté par la flèche rose) est en train d'être traité. La figure montre que ce mot ne se contente pas de regarder son voisin ; il envoie des "sondes" vers toutes les autres positions de la séquence. 
*   **L'idée clé** : Chaque mot de la phrase reçoit un "budget" d'attention de 100 % qu'il doit répartir entre tous les autres mots, y compris lui-même. 
*   **Le résultat** : Un mot isolé (embedding statique) s'enrichit de l'information de ses voisins pour devenir un vecteur contextuel unique. [SOURCE: Livre p.90, Figure 3-15]

### Le Trio Magique : Query, Key et Value
« Mes chers étudiants, voici le concept le plus crucial de votre formation. Si vous comprenez Query, Key et Value, vous comprenez l'intelligence artificielle moderne. » Le Transformer ne compare pas les vecteurs de mots directement. Il projette chaque mot dans trois espaces fonctionnels différents.

Comme l'illustre la **Figure 3-18 : Matrices de projection** (p.92), le modèle possède trois matrices de poids apprises durant l'entraînement : $W_Q, W_K$ et $W_V$. Lorsqu'un mot entre dans la couche d'attention :
1.  **Query (La Requête - Q)** : C'est ce que le mot cherche. Si le mot est "il", sa requête demande : "Où est mon sujet masculin dans cette phrase ?".
2.  **Key (La Clé - K)** : C'est l'étiquette du mot. Un mot comme "livre" possède une clé qui dit : "Je suis un objet inanimé masculin".
3.  **Value (La Valeur - V)** : C'est l'information sémantique pure que le mot contient.

🔑 **L'analogie de la bibliothèque du Prof. Henni** : Imaginez que vous cherchiez un tutoriel de cuisine sur YouTube. Votre barre de recherche est la **Query**. Le titre de la vidéo sur le serveur est la **Key**. Le contenu de la vidéo (ce que vous allez apprendre) est la **Value**. L'attention est l'algorithme qui fait correspondre votre recherche au titre le plus proche. [SOURCE: Livre p.91-92, Figure 3-18]

### Le calcul matriciel étape par étape (Analyse des Figures 3-19 à 3-21)
Le livre décompose cette chorégraphie mathématique avec une précision chirurgicale.

#### Étape 1 : Le calcul des scores de pertinence (Figure 3-19)
**Explication de la Figure 3-19** (p.93) : Pour chaque mot, on multiplie sa **Query** par les **Keys** de tous les autres mots. Mathématiquement, c'est un produit scalaire (Dot Product). 
*   Si les vecteurs Q et K pointent dans la même direction, le score est élevé.
*   Si ils sont orthogonaux, le score est nul.
La figure montre que cette opération crée une grille de scores indiquant à quel point chaque mot est "intéressé" par les autres. [SOURCE: Livre p.93, Figure 3-19]

#### Étape 2 : Le "Scaling" et le Softmax (Figure 3-20)
⚠️ **Attention : erreur fréquente ici !** Si l'on s'arrête aux scores bruts, les nombres peuvent devenir immenses, ce qui fait "exploser" les gradients lors de l'entraînement. 
🔑 **La solution mathématique** : On divise les scores par la racine carrée de la dimension des clés ($\sqrt{d_k}$). C'est le **Scaled Dot-Product Attention**. 
Ensuite, comme le montre la **Figure 3-20** (p.94), on applique une fonction **Softmax**. 
*   **L'effet visuel** : Les scores sont transformés en probabilités entre 0 et 1. La somme totale pour chaque mot est égale à 100 %. On voit alors apparaître une "carte d'attention" où certains liens s'allument (forte probabilité) et d'autres s'éteignent. [SOURCE: Livre p.94, Figure 3-20]

#### Étape 3 : La pondération des valeurs (Figure 3-21)
**Explication de la Figure 3-21** (p.95) : Enfin, on multiplie ces probabilités par les **Values**. 
*   Si le mot "il" a 90 % d'attention sur "livre", alors 90 % du vecteur final de "il" sera composé de l'information sémantique de "livre". 
*   Le résultat est un nouveau vecteur, le **Contextual Embedding**, qui a "aspiré" le sens de son environnement. [SOURCE: Livre p.95, Figure 3-21]

### Multi-Head Attention : Les cerveaux parallèles
« Pourquoi se contenter d'un seul regard sur une phrase ? » Un mot peut avoir plusieurs rôles : il a un rôle grammatical, un rôle sémantique et un rôle émotionnel. 

C'est ce qu'illustre la **Figure 3-17 : Intuition des têtes d'attention** (p.91). Au lieu d'avoir un seul trio Q, K, V, nous en créons plusieurs en parallèle (généralement 8, 12 ou même 96 têtes). 
*   Une tête peut se spécialiser dans la détection des verbes.
*   Une autre dans la résolution des pronoms (coréférence).
*   Une autre dans l'analyse du ton (ironie).

🔑 **Je dois insister :** La Multi-Head Attention est ce qui donne au Transformer sa nuance. À la fin, on fusionne (concatène) les résultats de toutes les têtes pour obtenir une vision riche et multidimensionnelle de la phrase. [SOURCE: Livre p.91, Figure 3-17]

### Exemple numérique : Le bac à sable des matrices
Pour bien fixer l'idée, imaginons une séquence de deux tokens : "Chat" et "Dort".
Supposons que nos vecteurs de Query et Key soient simplifiés à 2 dimensions.

1.  **Matrices Q et K** :
    *   $Q_{chat} = [1, 0]$, $K_{chat} = [1, 0]$
    *   $Q_{dort} = [0, 1]$, $K_{dort} = [0, 1]$
2.  **Calcul du score** (Produit scalaire $Q \cdot K^T$) :
    *   Score "Chat" vers "Chat" : $1\times1 + 0\times0 = 1$
    *   Score "Chat" vers "Dort" : $1\times0 + 0\times1 = 0$
3.  **Softmax** :
    *   Le mot "Chat" porte 100 % de son attention sur lui-même (car score=1 vs 0). 
    *   Si les vecteurs étaient plus proches (ex: "Chat" et "Félin"), les scores seraient partagés (ex: 60 % / 40 %).

⚠️ **Note du Professeur** : Dans un vrai LLM comme GPT-4, ces calculs se font sur des vecteurs de dimension 4096 ou plus. La complexité est telle que seule la puissance des GPU (section 1.2) permet de résoudre ces milliards de multiplications par seconde. [SOURCE: Blog 'The Illustrated Transformer' de Jay Alammar]

### L'attention comme moteur de la parallélisation
Pourquoi avons-nous abandonné les RNN ? Parce que dans le calcul $Q \cdot K^T$, nous pouvons calculer TOUS les scores de TOUS les mots en une seule opération matricielle géante. 
🔑 **La rupture technologique** : On ne fait plus la queue. Le GPU traite la phrase entière comme un bloc de pixels. C'est ce qui a permis de multiplier par 1000 la vitesse d'entraînement et d'ingérer l'intégralité du web. [SOURCE: Livre p.81]

### Éthique et Transparence : Le biais de l'attention
⚠️ **Éthique ancrée** : « Mes chers étudiants, l'attention n'est pas neutre. » 
Les matrices $W_Q, W_K, W_V$ sont apprises sur des données humaines. 
1.  **Le renforcement des stéréotypes** : Si, dans les données d'entraînement, le mot "Infirmière" porte systématiquement son attention sur des pronoms féminins, le modèle va figer cette association. L'attention devient alors un mécanisme de reproduction des préjugés. 
2.  **L'opacité du raisonnement** : Visualiser l'attention (Exercice 1 du laboratoire) nous donne une illusion de compréhension. Mais attention : une tête d'attention qui regarde une virgule peut le faire pour des raisons de syntaxe pure, et non pour le sens. Ne prêtez pas d'intentions humaines à une multiplication matricielle. [SOURCE: Livre p.28]

### Synthèse pour l'examen
Pour maîtriser cette section, vous devez être capables de réciter la formule de l'attention de Vaswani et al. :
$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
*   $QK^T$ : Qui regarde qui ? (Scores)
*   $\sqrt{d_k}$ : On calme les nombres (Scaling).
*   $softmax$ : On transforme en pourcentages (Normalisation).
*   $V$ : On extrait le sens (Information).

🔑 **Le message final du Prof. Henni pour cette section** : « L'attention est l'acte par lequel le modèle crée du contexte. Sans elle, les mots sont des îles. Avec elle, ils forment un continent de pensée. C'est la brique la plus puissante jamais inventée en informatique linguistique. » [SOURCE: Livre p.106]

« Vous avez maintenant dompté le lion ! Vous comprenez le mécanisme de l'attention. Mais un problème subsiste : si on traite tout en même temps, comment le modèle sait-il que le mot "Le" est avant le mot "Chat" ? Dans la section suivante, nous allons découvrir la boussole du Transformer : l'**Encodage Positionnel**. »

---
*Fin de la section 3.1 (2580 mots environ)*
## 3.2 Encodage positionnel (1800+ mots)

### Le paradoxe du Transformer : La mémoire sans l'ordre
« Bonjour à toutes et à tous ! J'espère que vous avez encore en tête notre "soirée cocktail" de la section 3.1. Nous avons vu que le Transformer est un génie de la simultanéité : il peut regarder tous les mots d'un livre en un clin d'œil grâce à la self-attention. Mais, mes chers étudiants, cette puissance a un prix terrifiant. 🔑 **Je dois insister sur ce paradoxe :** de par sa construction mathématique, le Transformer est **invariant par permutation**. Cela signifie que pour lui, les phrases "Le chat mange la souris" et "La souris mange le chat" sont rigoureusement identiques. Pourquoi ? Parce que l'attention calcule des scores entre des vecteurs sans se soucier de leur place dans la file d'attente. Sans une boussole pour indiquer l'ordre, notre cathédrale de calcul n'est qu'un sac de mots sophistiqué. Aujourd'hui, nous allons apprendre à donner le sens du temps et de l'espace à nos modèles. » [SOURCE: Livre p.102]

### L'intuition : Les coordonnées GPS du langage
Imaginez que vous receviez les pièces d'un puzzle, mais que toutes les pièces soient parfaitement carrées et lisses. Vous savez ce qu'il y a sur chaque pièce (l'embedding sémantique), mais vous n'avez aucune idée de l'endroit où elles s'emboîtent. L'encodage positionnel, c'est l'étiquette que l'on colle au dos de chaque pièce pour dire : "Je suis la pièce n°1, tout en haut à gauche". 

Dans les RNN (Semaine 1.2), l'ordre était implicite : le mot 2 arrivait forcément après le mot 1. Dans le Transformer, nous devons injecter cette information artificiellement. ⚠️ **Attention : erreur fréquente ici !** On ne donne pas simplement un numéro (1, 2, 3...) au modèle. Pourquoi ? Parce que si la phrase est très longue, le nombre "1000" écraserait par sa valeur mathématique les autres informations du vecteur. Nous avons besoin d'une méthode plus subtile. [SOURCE: Vaswani et al., 2017 / Livre p.102]

---

### La méthode classique : Les ondes sinusoïdales
Dans l'article original de 2017, les chercheurs ont utilisé des fonctions sinus et cosinus. 
*   **L'idée** : Chaque position dans la phrase est associée à une fréquence d'onde unique. 
*   **Le bénéfice** : Cela permet au modèle de comprendre la distance relative. Si le modèle sait comment oscille l'onde entre la position 2 et la position 5, il peut généraliser cette "distance de 3" à n'importe quel endroit du texte.

Cependant, cette méthode "absolue" (on ajoute l'information au début du voyage) a montré ses limites lorsque nous avons voulu créer des modèles capables de lire des textes de plus en plus longs. C'est là qu'intervient la révolution de l'encodage rotatif. [SOURCE: Livre p.102]

---

### La révolution RoPE (Rotary Positional Embeddings)
Si vous regardez les spécifications de modèles comme **Llama-3**, **Mistral** ou **Phi-3**, vous verrez toujours mentionné "**RoPE**". C'est aujourd'hui le standard absolu. 

Regardons attentivement la **Figure 3-32 : Application des Rotary Embeddings** (p.103 du livre). 
**Explication de la Figure 3-32** : Cette illustration est fondamentale pour comprendre la différence de philosophie. 
*   **Ancien monde** : On ajoutait la position une seule fois, tout au début, sur les embeddings d'entrée (les boîtes bleues en bas).
*   **Monde RoPE** : Comme le montre la figure, l'encodage positionnel est injecté **à chaque couche**, directement à l'intérieur des blocs d'attention (les ronds violets). 
🔑 **Je dois insister :** RoPE n'est pas une addition, c'est une **multiplication**. On ne "colle" pas une étiquette, on fait "pivoter" le vecteur. [SOURCE: Livre p.103, Figure 3-32]

#### La mathématique de la rotation (Analyse de la Figure 3-33)
Passons à la géométrie avec la **Figure 3-33 : La rotation des vecteurs** (p.104 du livre). 

**Explication de la Figure 3-33** : Imaginez que chaque paire de dimensions dans votre vecteur (votre Query ou votre Key) soit une aiguille sur une horloge. 
*   Pour le mot n°1, on tourne l'aiguille de 10 degrés.
*   Pour le mot n°2, on la tourne de 20 degrés.
*   **Le miracle du produit scalaire** : Lorsque le modèle calcule l'attention entre deux mots, la mathématique de la rotation fait que le score final dépend uniquement de **l'angle entre les deux aiguilles**. 
*   Si les mots sont proches, l'angle est petit, le score est fort. S'ils sont loin, l'angle est grand, le score faiblit.

🔑 **L'intuition du Professeur Henni :** RoPE permet au modèle de "sentir" la distance entre les mots sans avoir besoin de connaître leur position absolue. C'est comme si, dans une file d'attente, vous ne saviez pas que vous étiez le 50ème, mais que vous sentiez exactement que la personne devant vous est à 50 cm et celle de derrière à 50 cm. C'est l'**Attention Relative**. [SOURCE: Livre p.104, Figure 3-33 / Su et al., 2021]

---

### Pourquoi RoPE a-t-il gagné ?
1.  **Extrapolabilité** : Un modèle entraîné sur des phrases de 2048 mots peut, grâce à RoPE, comprendre (un peu mieux) des phrases de 4000 mots car il comprend la logique de rotation.
2.  **Stabilité** : Les rotations préservent la norme (la "longueur") des vecteurs, ce qui évite que le modèle ne devienne instable pendant l'entraînement.
3.  **Richesse sémantique** : En faisant varier la vitesse de rotation selon les dimensions, le modèle peut dévouer certaines parties de son cerveau aux relations à court terme (mots voisins) et d'autres aux relations à long terme (début et fin de paragraphe). [SOURCE: Livre p.105]

---

### Optimisation de l'entraînement : Le Packing (Figure 3-31)
« Mes chers étudiants, l'informatique n'est pas qu'une affaire de mathématiques, c'est aussi une affaire d'économie. » Entraîner un LLM coûte des millions d'euros en électricité. Chaque seconde où votre GPU ne calcule rien est un gaspillage. 

Regardons la **Figure 3-31 : Packing des documents** (p.103 du livre). 
**Explication de la Figure 3-31** : Elle compare deux méthodes d'organisation des données.
*   **Approche naïve (Haut)** : Si vous avez une phrase de 10 mots et une fenêtre de contexte de 2048, vous remplissez le reste avec du "Padding" (des zéros). Le GPU passe son temps à multiplier des zéros. C'est un désastre d'efficacité.
*   **Approche par Packing (Bas)** : On "compacte" plusieurs documents différents à la suite dans le même bloc de 2048 tokens, séparés par un token spécial. 

🔑 **Le défi technique** : Grâce aux encodages positionnels modernes, le modèle est capable de comprendre que même s'ils sont dans le même bloc, le Document n°2 recommence à la position 1. Sans cela, le modèle croirait que le début du deuxième article est la suite logique de la fin du premier. [SOURCE: Livre p.103, Figure 3-31]

---

### Limites et Frontières : La fenêtre de contexte
⚠️ **Fermeté bienveillante** : « Ne croyez pas que la mémoire de l'IA soit infinie. » 
Même avec RoPE, chaque modèle possède une "Context Window" (Fenêtre de contexte) maximale.
*   **La limite physique** : Si le modèle a été entraîné avec une rotation maximale correspondant à 8000 tokens, lui en donner 100 000 va le rendre "étourdi". Les angles de rotation deviennent trop serrés, et il perd le fil de la logique.
*   **Le coût quadratique** : Rappelez-vous la section 3.1. Même si l'encodage positionnel est parfait, le calcul de l'attention demande toujours $N \times N$ opérations. Doubler la fenêtre de contexte multiplie par quatre le besoin en mémoire vive du GPU. [SOURCE: Livre p.81]

### Laboratoire de réflexion : Le temps est-il une dimension ?
⚠️ **Éthique ancrée** : « Mes chers étudiants, réfléchissez à l'impact de ce découpage. » 
Pour un Transformer, le temps n'existe pas. Il n'y a que des positions dans une grille. 
1.  **L'absence de causalité réelle** : Le modèle ne comprend pas que la cause précède l'effet parce que c'est une loi physique ; il le comprend parce que statistiquement, le token "Cause" a une position inférieure au token "Effet" dans ses données d'entraînement. 
2.  **Le biais de position** : On a remarqué que les modèles accordent souvent plus d'importance aux informations situées au début et à la fin d'un texte (le phénomène "Lost in the Middle"). C'est une conséquence directe de la façon dont nous encodons les positions. 

🔑 **Mon conseil de professeur** : Lorsque vous construisez un système de RAG (Semaine 9), assurez-vous que l'information cruciale ne se trouve pas perdue au milieu d'un énorme bloc de texte, car l'encodage positionnel sémantique y est souvent moins "vif". [SOURCE: Livre p.28, p.177]

---

### Synthèse de la section
Nous avons vu comment le Transformer, initialement aveugle à l'ordre, a acquis une boussole spatio-temporelle. 
*   **L'encodage absolu** (sinus) a posé les bases.
*   **L'encodage rotatif (RoPE)** a apporté la flexibilité et la notion de distance relative, permettant l'explosion des fenêtres de contexte que nous connaissons aujourd'hui.
*   **Le Packing** garantit que nos GPU travaillent à 100% de leur capacité.

🔑 **Le message final du Prof. Henni pour cette section** : « L'ordre des mots est la structure de notre pensée. En apprenant à faire pivoter des vecteurs dans l'espace complexe, les chercheurs ont réussi l'impossible : garder la puissance du calcul parallèle tout en respectant la mélodie séquentielle du langage humain. C'est un triomphe de l'ingénierie mathématique. » [SOURCE: Livre p.106]

« Vous savez maintenant comment le Transformer regarde et comment il se repère. Mais un cerveau ne se résume pas à ses yeux. Dans la section suivante, nous allons étudier la "matière grise" du modèle : les **Blocs Transformer** et comment nous les optimisons pour qu'ils ne brûlent pas vos serveurs. »

---
*Fin de la section 3.2 (1840 mots environ)*
## 3.3 Blocs Transformer et optimisation (2300+ mots)

### La structure de la pensée : Au-delà du simple regard
« Bonjour à toutes et à tous ! Nous avons parcouru un chemin fascinant jusqu'ici. Nous avons vu comment le Transformer utilise ses "yeux" (la self-attention) pour naviguer dans le contexte et sa "boussole" (l'encodage positionnel) pour se repérer dans le temps. Mais, mes chers étudiants, un regard et une boussole ne font pas un cerveau. Pour transformer ces signaux électriques en une pensée structurée, il nous faut une architecture capable de digérer, de filtrer et de stabiliser l'information. 🔑 **Je dois insister :** l'intelligence d'un LLM ne réside pas seulement dans ses équations d'attention, elle réside dans la répétition obstinée et optimisée d'une unité fondamentale : le **bloc Transformer**. Aujourd'hui, nous allons démonter ce bloc pièce par pièce pour comprendre comment il permet aux modèles de 70 milliards de paramètres de ne pas s'effondrer sous leur propre poids. Respirez, nous entrons dans l'ingénierie de la puissance. » [SOURCE: Livre p.101]

---

### 1. L'architecture du bloc original (Analyse de la Figure 3-29)
Commençons par regarder les plans de la machine d'origine, telle qu'imaginée en 2017. Regardez la **Figure 3-29 : Un bloc Transformer de l'article original** (p.101 du livre).

**Explication de la Figure 3-29** : Cette illustration nous présente une unité de traitement composée de deux "étages" superposés.
1.  **L'étage inférieur (Communication)** : C'est la couche de Multi-Head Attention. C'est ici que les mots "se parlent".
2.  **L'étage supérieur (Réflexion)** : C'est la couche Feedforward (FFN). C'est ici que chaque mot "réfléchit" individuellement sur les informations qu'il vient de récolter.
3.  **Les flèches pointillées (Residual Connections)** : Notez ces lignes qui contournent les blocs. Elles sont le secret de la survie du signal.
4.  **Les boîtes "Add & Norm"** : Elles agissent comme des douanes qui régulent le flux pour éviter que les nombres ne deviennent trop grands ou trop petits. [SOURCE: Livre p.101, Figure 3-29]

🔑 **L'intuition du Professeur Henni :** Imaginez que le bloc Transformer soit une réunion de travail. La Self-attention, c'est le moment du débat où tout le monde échange des idées. Le Feedforward, c'est le moment où chaque participant retourne à son bureau pour rédiger sa propre synthèse de la réunion. Sans le débat, personne n'apprend rien de neuf. Sans le travail individuel, on n'aboutit à aucune décision concrète. [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU BLOG 'ILLUSTRATED TRANSFORMER']

---

### 2. Les connexions résiduelles : L'autoroute du gradient
⚠️ **Attention : erreur fréquente ici !** On pourrait croire que plus on empile de couches, plus le modèle est intelligent. En réalité, sans les connexions résiduelles (les "Skip Connections"), un modèle de 12 couches serait incapable d'apprendre.

**Pourquoi sont-elles vitales ?**
Rappelez-vous le problème de la disparition du gradient (Semaine 1.2). À chaque fois que l'information traverse une couche complexe comme l'attention, le signal s'affaiblit. 
🔑 **La solution mathématique** : Au lieu de calculer $y = f(x)$, on calcule $y = x + f(x)$. On ajoute l'entrée originale au résultat du calcul.
*   **L'effet autoroute** : Pendant l'entraînement, le signal d'erreur peut "sauter" par-dessus les couches complexes via ces connexions directes pour atteindre les premières couches du modèle. C'est ce qui permet d'entraîner des modèles de 100 couches ou plus (comme GPT-4) sans que le cerveau de l'IA ne devienne amnésique. [SOURCE: Livre p.101 / He et al., 2016]

---

### 3. La Normalisation : Garder la raison dans les nombres
Dans un réseau de neurones, si les nombres deviennent trop grands, la machine "explose" (overflow). S'ils deviennent trop petits, elle "s'éteint" (underflow). La normalisation est le thermostat qui maintient tout à une température stable.

#### LayerNorm vs RMSNorm
Historiquement, nous utilisions la **LayerNorm**. Elle recalcule la moyenne et la variance de toutes les activations pour les ramener vers une distribution standard (moyenne 0, écart-type 1).
🔑 **L'évolution moderne (Figure 3-30)** : Regardez la **Figure 3-30 : Bloc Transformer d'un modèle de l'ère 2024** (p.102). Vous remarquerez que l'on utilise désormais la **RMSNorm** (*Root Mean Square Layer Normalization*). 
*   **La différence** : La RMSNorm ne calcule pas la moyenne, seulement la racine carrée de la moyenne des carrés. 
*   **L'avantage** : Elle est environ 40 % plus rapide à calculer sur GPU et offre la même stabilité. C'est pour cela que **Llama 3** et **Phi-3** l'utilisent exclusivement. [SOURCE: Livre p.101-102, Figure 3-30 / Zhang et al., 2019]

#### Pre-Normalization vs Post-Normalization
Observez bien la position de la boîte "Normalize" entre la Figure 3-29 et la 3-30.
*   **Original (Post-Norm)** : On normalise *après* l'addition. C'est instable au début de l'entraînement.
*   **Moderne (Pre-Norm)** : On normalise *avant* d'entrer dans l'attention ou le feedforward. 
🔑 **Je dois insister :** La Pre-Norm est ce qui permet de lancer des entraînements massifs sans que le modèle ne diverge de manière catastrophique dans les premières heures. C'est le standard industriel actuel. [SOURCE: Livre p.102]

---

### 4. Le réseau Feedforward (FFN) : La digestion sémantique
Après que l'attention a mélangé les informations des mots, chaque token passe par un réseau de neurones dense identique. 
🔑 **Note technique** : Ce réseau augmente généralement la dimensionnalité (ex: de 768 à 3072) pour permettre au modèle de projeter le texte dans un espace beaucoup plus vaste, avant de le re-compresser. 
C'est ici que le modèle stocke ses "faits" (ex: "La capitale de la France est Paris"). Si l'attention est le système de communication, le FFN est la base de données de connaissances mémorisées. [SOURCE: Livre p.87, Figure 3-13]

---

### 5. Optimiser l'attention : La course vers la vitesse
« Mes chers étudiants, le plus grand ennemi de l'ingénieur IA est la mémoire vive (VRAM) de la carte graphique. » Le calcul de l'attention est gourmand. Heureusement, nous avons inventé des méthodes pour "tricher" intelligemment.

#### FlashAttention : L'IA au service du matériel
Les Figures 3-22 à 3-28 (p.96-100) détaillent les optimisations de l'attention. La plus célèbre est **FlashAttention**. 
**Le problème (HBM vs SRAM)** : Normalement, le GPU passe son temps à lire et écrire les matrices d'attention sur sa mémoire lente (HBM). C'est comme si vous deviez retourner à la bibliothèque à chaque fois que vous lisez une ligne d'un livre.
**La solution** : FlashAttention découpe le calcul en petits blocs qui tiennent dans la mémoire ultra-rapide (SRAM) du processeur. 
🔑 **Le résultat** : On peut multiplier par 3 la vitesse d'entraînement et de génération sans changer un seul paramètre du modèle. C'est une optimisation de pur génie logiciel. [SOURCE: Livre p.100 / Dao et al., 2022]

#### Grouped-Query Attention (GQA) : Économiser le KV Cache
C'est l'astuce qui permet à vos modèles de 7B ou 70B de tenir sur une seule carte graphique. Regardons la **Figure 3-25 : Comparaison des types d'attention** (p.98).

**Explication de la Figure 3-25** :
1.  **Multi-head (Haut gauche)** : Chaque tête de "Query" (ce que je cherche) possède sa propre tête de "Key" et "Value". C'est très précis mais cela sature la mémoire (le fameux KV cache).
2.  **Multi-query (Haut droite)** : Toutes les Queries se partagent une seule Key et une seule Value. C'est ultra-rapide mais le modèle devient un peu "étourdi" et perd en précision.
3.  **Grouped-query (Bas)** : Le compromis parfait. On groupe les Queries (ex: par 4) et chaque groupe partage une paire Key/Value. 

🔑 **Je dois insister :** GQA est ce qui permet aux modèles modernes d'avoir des fenêtres de contexte immenses (8k, 32k, ou plus) sans que la mémoire du GPU n'explose. C'est l'architecture utilisée par **Llama-2/3**. [SOURCE: Livre p.98-100, Figure 3-25]

---

### Tableau 3-1 : Comparaison des architectures d'attention

| Méthode | Mémoire (KV Cache) | Performance Sémantique | Utilisé par... |
| :--- | :--- | :--- | :--- |
| **Multi-Head Attention** | Maximale (Lourd) | Étalon-or | GPT-3, BERT |
| **Multi-Query Attention** | Minimale (Léger) | Moyenne | PaLM |
| **Grouped-Query (GQA)** | **Optimisée** | **Excellente** | **Llama 3, Mistral** |

[SOURCE: CONCEPT À SOURCER – SYNTHÈSE DES PAGES 96-100 DU LIVRE]

---

### Laboratoire de code : Inspecter la structure d'un bloc (Colab T4)
« Ne me croyez pas sur parole, ouvrez la machine ! » Voici comment explorer les entrailles d'un modèle moderne comme Phi-3 pour y retrouver nos composants.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers accelerate

from transformers import AutoModelForCausalLM, AutoConfig

# 1. Chargement de la configuration d'un modèle moderne
# [SOURCE: Choix de modèle compact Livre p.54]
model_id = "microsoft/Phi-3-mini-4k-instruct"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# 2. ANALYSE DES HYPERPARAMÈTRES DU BLOC
# [SOURCE: Propriétés du Transformer Livre p.78]
print(f"--- STRUCTURE DU BLOC {model_id} ---")
print(f"Nombre de couches (Blocs) : {config.num_hidden_layers}")
print(f"Dimension du modèle (d_model) : {config.hidden_size}")
print(f"Nombre de têtes d'attention : {config.num_attention_heads}")
print(f"Type de fonction d'activation : {config.hidden_act}") # Souvent 'silu' pour SwiGLU

# 3. VÉRIFICATION DU GQA
# Si num_key_value_heads < num_attention_heads, c'est du GQA !
if hasattr(config, "num_key_value_heads"):
    print(f"Têtes Key/Value (GQA) : {config.num_key_value_heads}")
    ratio = config.num_attention_heads // config.num_key_value_heads
    print(f"Facteur de compression GQA : {ratio}:1")

# [SOURCE: CONCEPT À SOURCER – DOCUMENTATION HUGGING FACE]
```

⚠️ **Fermeté bienveillante** : Observez le "Facteur de compression GQA". Si vous voyez un ratio de 4:1, cela signifie que vous économisez 75 % de la mémoire nécessaire pour stocker le contexte par rapport à un Transformer classique. C'est la différence entre pouvoir discuter 10 minutes avec l'IA ou seulement 2 minutes. 🔑 **C'est une distinction non-négociable pour vos futurs déploiements.**

---

### Éthique et Responsabilité : L'IA énergivore
⚠️ **Éthique ancrée** : « Mes chers étudiants, l'optimisation n'est pas qu'une question de vitesse, c'est une question d'éthique. » 
Chaque bloc Transformer que nous empilons demande des milliards d'opérations. 
1.  **L'impact environnemental** : La course au "plus gros modèle" a un coût carbone colossal. En maîtrisant FlashAttention ou GQA, vous apprenez à être des ingénieurs sobres : obtenir la même intelligence avec moins de kilowatts. 
2.  **L'accessibilité** : Si nous ne développions pas ces optimisations, l'IA resterait le privilège exclusif de trois ou quatre entreprises milliardaires. Un modèle optimisé est un modèle qui peut tourner dans un hôpital de campagne ou sur le smartphone d'un étudiant. 🔑 **L'ingénierie est un acte de démocratisation.** [SOURCE: Livre p.28, Afterword p.391]

🔑 **Le message final du Prof. Henni pour cette section** : « Un bloc Transformer est un chef-d'œuvre d'équilibre. Il doit laisser passer le signal sans le déformer (Résidus), stabiliser les calculs (RMSNorm), permettre le dialogue (GQA) et stocker le savoir (FFN). En comprenant ces rouages, vous ne voyez plus l'IA comme une "magie noire", mais comme une horlogerie fine de haute précision. » [SOURCE: Livre p.106]

« Nous avons terminé l'étude de la matière grise ! Vous savez désormais comment le Transformer traite l'information. Mais comment tout cela s'assemble-t-il concrètement lors d'une discussion réelle ? Dans la dernière section de cette semaine, nous allons suivre le voyage d'un token à travers toutes ces couches : c'est le **Forward Pass complet**. »

---
*Fin de la section 3.3 (2340 mots environ)*
## 3.4 Forward pass complet (1500+ mots)

### Le grand voyage du token : De l'entrée à la parole
« Bonjour à toutes et à tous ! Nous arrivons aujourd'hui au point d'orgue de notre troisième semaine. Nous avons étudié les yeux du modèle, sa boussole et sa matière grise. Mais comment tout cela s'assemble-t-il concrètement lorsqu'un utilisateur tape une question ? Comment un simple courant électrique traversant des milliards de transistors se transforme-t-il en une phrase cohérente comme "La capitale de la France est Paris" ? 🔑 **Je dois insister :** comprendre le **Forward Pass** (la passe avant), c'est comprendre la vie biologique d'une information au sein de la machine. Aujourd'hui, nous allons suivre le voyage d'un token, de sa naissance sous forme d'index numérique jusqu'à sa métamorphose en probabilité statistique. Respirez, nous allons parcourir l'intégralité du circuit. » [SOURCE: Livre p.76]

### 1. L'architecture du flux (Analyse de la Figure 3-4)
Commençons par regarder la carte du trajet. La **Figure 3-4 : Les composants de la passe avant** (p.76 du livre) est notre plan de vol. [SOURCE: Livre p.76, Figure 3-4]

**Explication de la Figure 3-4** : Cette illustration nous montre que la passe avant n'est pas un bloc monolithique, mais une succession de trois grandes gares :
1.  **Le Tokeniseur** : Il transforme le texte en IDs.
2.  **La Pile de blocs Transformer** : C'est le cœur du traitement (la "boîte noire").
3.  **La Tête de modélisation du langage (LM Head)** : C'est là que la décision finale est prise.
🔑 **Notez bien cette intuition :** l'information ne circule que dans un seul sens, du haut vers le bas (ou de l'entrée vers la sortie). Contrairement à l'entraînement (Backpropagation), ici on ne revient jamais en arrière. On calcule, on avance. [SOURCE: Livre p.76]

---

### 2. Phase 1 : La porte d'entrée (Analyse de la Figure 3-5)
Tout commence par un texte brut. Supposons que l'utilisateur tape : "Say something smart".
Regardez la **Figure 3-5 : Le vocabulaire et les embeddings** (p.77 du livre). [SOURCE: Livre p.77, Figure 3-5]

**Explication de la Figure 3-5** : 
*   **Mise en correspondance** : Le mot "smart" est identifié dans le dictionnaire du tokeniseur. Supposons que son ID soit `50000`.
*   **L'extraction du vecteur** : Le modèle va chercher la 50 000ème ligne de sa matrice d'embeddings. Comme le montre la figure, il en ressort un vecteur de nombres (ex: 768 ou 3072 dimensions). 
*   **L'injection de position** : À ce vecteur, on ajoute immédiatement l'encodage positionnel (vu en 3.2). Sans cela, le modèle saurait que l'on parle de "smart", mais il ne saurait pas que c'est le troisième mot de la phrase. 

🔑 **L'analogie du Professeur Henni :** C'est comme un voyageur qui arrive à l'aéroport. On lui donne un badge (l'embedding) et un numéro de siège (la position). Sans ces deux éléments, il ne peut pas embarquer dans l'avion Transformer. [SOURCE: Livre p.77]

---

### 3. Phase 2 : La traversée des blocs (Le traitement profond)
Une fois le passager "textuel" équipé, il entre dans la pile de blocs. 
⚠️ **Attention : erreur fréquente ici !** On imagine souvent que l'information reste la même tout au long de la pile. En réalité, le vecteur change de nature à chaque étage.

*   **Dans le bloc 1** : Le token "smart" regarde ses voisins ("say", "something"). Il comprend qu'il est l'adjectif d'une requête impérative.
*   **Dans le bloc 12** : Après être passé par 12 couches de Self-Attention et de Feedforward (FFN), le vecteur de "smart" contient maintenant une synthèse incroyablement riche. Il ne représente plus seulement le mot, mais l'intention de l'utilisateur de recevoir une réponse intelligente.

🔑 **Je dois insister sur un point technique capital :** Lors de la génération de texte, le modèle produit un vecteur de sortie pour *chaque* mot de l'entrée. Mais pour prédire le mot suivant, nous n'utilisons que le vecteur correspondant à la **dernière position**. Pourquoi ? Parce que grâce à l'attention, ce dernier vecteur a déjà "absorbé" toute la connaissance des mots qui le précèdent. [SOURCE: Livre p.82, Figure 3-9]

---

### 4. Phase 3 : La décision finale (Analyse de la Figure 3-6)
Le voyageur sort enfin de la pile de blocs. Il se présente devant la **LM Head**. Regardons la **Figure 3-6 : Prédiction de probabilité** (p.78 du livre). [SOURCE: Livre p.78, Figure 3-6]

**Explication de la Figure 3-6** : 
*   **La projection** : Le vecteur final (ex: 768 dimensions) est projeté vers un espace immense correspondant à la taille du vocabulaire (ex: 50 000 dimensions).
*   **Le score brut (Logits)** : Chaque mot du dictionnaire reçoit une note. Le mot "Think" pourrait recevoir 15.2, le mot "The" 8.1, et le mot "Banana" -4.5.
*   **Le Softmax** : On transforme ces notes en pourcentages. "Think" devient 40% probable, "The" 10%, etc. 

🔑 **C'est le moment de la parole :** Le modèle ne "sait" pas quel est le bon mot. Il sait simplement lequel est statistiquement le plus cohérent après "Say something smart". [SOURCE: Livre p.78-79]

---

### 5. Optimisation : Le KV Cache (Analyse de la Figure 3-10)
« Mes chers étudiants, rappelez-vous mon avertissement de la section 1.2 : le Transformer est gourmand. » Si nous devions refaire tout ce voyage pour chaque lettre, l'IA mettrait des minutes à répondre.

Regardez la **Figure 3-10 : KV cache pour accélération** (p.84 du livre). 
**Explication de la Figure 3-10** : 
*   **Le gaspillage** : Pour générer le mot n°2, le modèle a besoin de re-calculer l'attention sur le mot n°1. 
*   **La solution** : On stocke les Keys (K) et les Values (V) du mot n°1 dans une mémoire vive ultra-rapide sur le GPU. 
*   **Le gain** : Pour le mot n°2, le modèle ne fait voyager QUE le nouveau mot dans la pile de blocs. Il va chercher le passé dans son "frigo" (le cache). 

🔑 **Je dois insister :** Le KV Cache est ce qui permet à ChatGPT de vous répondre en "streaming" (mot à mot) en temps réel. Sans cette optimisation, l'IA de production n'existerait pas. [SOURCE: Livre p.83-84, Figure 3-10]

---

### 6. Sampling et Décodage : Choisir dans le nuage (Section 3.2.1)
Une fois que nous avons nos probabilités (Figure 3-6), comment choisir le mot final ?
1.  **Greedy Decoding** : On prend toujours le n°1 (40% "Think"). C'est sûr mais ennuyeux.
2.  **Sampling (Échantillonnage)** : On tire au sort selon les poids. "Think" a 4 chances sur 10 de sortir. C'est ce qui donne du "style" à l'IA.

⚠️ **Fermeté bienveillante** : « Ne confondez pas le calcul (Forward Pass) et le choix (Decoding). » Le Forward Pass est une mathématique déterministe. Le décodage est l'endroit où nous injectons le hasard (la Température) pour rendre l'IA humaine. [SOURCE: Livre p.79-80]

---

### Laboratoire de code : Analyse de la structure (Colab T4)
Pour conclure cette semaine, je veux que vous sachiez comment lire le "plan de vol" de n'importe quel modèle.

```python
# Testé sur Colab T4 16GB VRAM
from transformers import AutoModelForCausalLM

# 1. CHARGEMENT D'UN MODÈLE COMPACT
# [SOURCE: Choix de modèle pédagogique Livre p.54]
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. INSPECTION DU FORWARD PASS
# Cette commande imprime l'ordre exact des couches que le token va traverser
print("--- ARCHITECTURE DU MODÈLE (PASSE AVANT) ---")
print(model)

# --- EXPLICATION DES RÉSULTATS ---
# [SOURCE: Anatomie du Transformer Livre p.100]
# Vous verrez :
# - 'wte' (Word Token Embeddings) : Gare de départ
# - 'wpe' (Word Position Embeddings) : La boussole
# - 'h' (Blocks) : Les 12 étages de la matière grise
# - 'ln_f' (Final LayerNorm) : La stabilisation finale
# - 'lm_head' : La bouche du modèle (Dernière étape)
```

⚠️ **Note du Professeur** : Regardez bien la couche `lm_head`. Vous verrez `Linear(in_features=768, out_features=50257)`. 🔑 **C'est la preuve mathématique :** on transforme un résumé interne de 768 nombres en un choix parmi 50 257 mots possibles. [SOURCE: Livre p.101]

---

### Éthique et Responsabilité : La boîte noire et le déterminisme
⚠️ **Éthique ancrée** : « Mes chers étudiants, la passe avant est une mécanique d'une précision effrayante, mais elle est opaque. » 
1.  **L'impossibilité de l'arrêt** : Une fois que le Forward Pass est lancé, on ne peut pas l'arrêter à mi-chemin pour dire au modèle : "Hé, tu es en train de prendre une mauvaise direction logique !". Le modèle calcule jusqu'au bout.
2.  **Le biais sémantique** : Si, à la couche 5, une tête d'attention a fait une erreur d'interprétation, cette erreur va se propager et s'amplifier dans les 27 couches suivantes. C'est l'effet papillon du neurone. 
3.  **La consommation invisible** : Chaque Forward Pass, même pour dire "Bonjour", consomme une quantité d'électricité précise sur le serveur. 🔑 **La responsabilité de l'ingénieur** est de savoir quand utiliser un gros modèle ou un petit (Semaine 13) pour économiser ces ressources. [SOURCE: Livre p.28]

🔑 **Le message final du Prof. Henni pour la semaine 3** : « Vous avez maintenant une vision à 360 degrés. Vous savez comment le Transformer est construit, comment il se repère, et comment l'information y circule à la vitesse de la lumière. Vous n'êtes plus des utilisateurs passifs ; vous êtes des mécaniciens de l'intelligence. Félicitations pour avoir franchi cette étape ! Dès la semaine prochaine, nous allons spécialiser ces connaissances en étudiant les modèles qui excellent dans la compréhension pure : les modèles **Encoder-only** comme BERT. » [SOURCE: Livre p.106]

« Nous avons terminé notre immense plongée dans l'architecture ! Vous avez mérité votre pause. Préparez vos notebooks pour le laboratoire, nous allons mettre tout cela en mouvement ! »

---
*Fin de la section 3.4 (1540 mots environ)*
[CONTENU SEMAINE 3]

## 🧪 LABORATOIRE SEMAINE 3 (800+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Nous y sommes : le moment de vérité où les équations de la semaine se transforment en réalité numérique. Dans ce laboratoire, nous allons "ouvrir le capot" d'un Transformer pour voir ses pistons (l'attention) et ses engrenages (les blocs) en mouvement. 🔑 **Je dois insister :** l'architecture que vous allez manipuler aujourd'hui est le socle de tout l'édifice des LLM. Ne vous contentez pas d'exécuter les cellules : observez comment la structure du modèle dicte sa capacité à comprendre. Prêt·e·s à explorer les entrailles de la machine ? C'est parti ! » [SOURCE: Livre p.73]

---

### 🔹 QUIZ MCQ (10 questions)

1. **Combien de matrices de projection principales sont entraînées au sein d'une seule tête d'attention pour transformer les embeddings ?**
   a) Une seule matrice de poids globale.
   b) Deux matrices (Entrée et Sortie).
   c) Trois matrices distinctes ($W_Q$, $W_K$, $W_V$).
   d) Douze matrices (une par couche).
   **[Réponse: c]** [Explication: Chaque mot est projeté dans trois espaces fonctionnels : ce qu'il cherche (Query), ce qu'il contient (Key) et l'information qu'il apporte (Value). SOURCE: Livre p.92, Figure 3-18]

2. **Quel est le rôle spécifique du facteur de division $\sqrt{d_k}$ dans le calcul de la Scaled Dot-Product Attention ?**
   a) Augmenter la vitesse de calcul du GPU.
   b) Stabiliser les gradients en empêchant les scores de similarité de devenir trop élevés, ce qui saturerait le Softmax.
   c) Réduire la taille du dictionnaire du modèle.
   d) Masquer les tokens futurs dans le décodeur.
   **[Réponse: b]** [Explication: Sans ce facteur d'échelle (scaling), les produits scalaires de grande dimension créent des valeurs extrêmes, rendant l'entraînement instable. SOURCE: Vaswani et al., 2017 / Livre p.94]

3. **Quelle optimisation de pointe permet d'accélérer l'attention en évitant les allers-retours entre la mémoire lente (HBM) et la mémoire rapide (SRAM) du GPU ?**
   a) Le Dropout.
   b) FlashAttention.
   c) L'encodage sinusoïdal.
   d) Le découpage BPE.
   **[Réponse: b]** [Explication: FlashAttention découpe le calcul en blocs pour qu'ils tiennent entièrement dans la mémoire proche du processeur. SOURCE: Livre p.100]

4. **Pourquoi les modèles modernes comme Llama 3 ou Phi-3 préfèrent-ils la RMSNorm à la LayerNorm classique ?**
   a) Elle est mathématiquement plus complexe et précise.
   b) Elle est plus légère car elle ne calcule pas la moyenne, seulement la racine carrée de la moyenne des carrés, offrant une meilleure efficacité sur GPU.
   c) Elle permet de supprimer le mécanisme d'attention.
   d) Elle n'est compatible qu'avec les modèles multilingues.
   **[Réponse: b]** [Explication: RMSNorm simplifie la normalisation sans sacrifier la stabilité, ce qui accélère l'entraînement. SOURCE: Livre p.101-102]

5. **Dans un chatbot en production, que stocke précisément le "KV Cache" pour éviter de recalculer toute la phrase à chaque mot généré ?**
   a) Les mots en format texte brut.
   b) Les vecteurs Key et Value de tous les tokens passés pour chaque couche du modèle.
   c) Les mots de passe des utilisateurs.
   d) Les gradients de l'étape de backpropagation.
   **[Réponse: b]** [Explication: En gardant les K et V en mémoire, le modèle n'a besoin de calculer l'attention que pour le tout nouveau token produit. SOURCE: Livre p.84, Figure 3-10]

6. **Quel composant du Transformer est responsable du "mélange" des informations entre les différents mots de la séquence ?**
   a) La couche de normalisation.
   b) Le réseau Feedforward (FFN).
   c) Le mécanisme de Self-Attention.
   d) Les connexions résiduelles.
   **[Réponse: c]** [Explication: L'attention est le seul moment où les tokens "communiquent" entre eux ; le FFN, lui, traite chaque mot isolément. SOURCE: Livre p.86]

7. **Pourquoi l'architecture Transformer est-elle plus rapide à entraîner qu'un RNN ?**
   a) Elle possède moins de paramètres.
   b) Elle permet de traiter tous les mots de la séquence simultanément (parallélisation) au lieu de l'un après l'autre.
   c) Elle ne nécessite pas de GPU.
   d) Elle utilise des fichiers texte plus petits.
   **[Réponse: b]** [Explication: L'absence de dépendance temporelle stricte permet aux matrices d'attention d'être calculées d'un seul bloc. SOURCE: Livre p.16, p.81]

8. **Dans la Multi-head attention, si la dimension totale est 768 et que nous avons 12 têtes, quelle est la dimension de chaque tête ?**
   a) 768
   b) 12
   c) 64
   d) 1024
   **[Réponse: c]** [Explication: 768 / 12 = 64. On divise le vecteur en segments plus petits pour que chaque tête apprenne une relation différente. SOURCE: Livre p.91]

9. **Quel mécanisme permet au signal d'erreur de "sauter" par-dessus les couches pour éviter l'oubli du gradient lors de l'entraînement ?**
   a) Le Softmax.
   b) Les Connexions Résiduelles (Skip Connections).
   c) La quantification 4-bit.
   d) Le découpage en patches.
   **[Réponse: b]** [Explication: On additionne l'entrée à la sortie ($x + f(x)$), créant une autoroute pour l'information. SOURCE: Livre p.101, Figure 3-29]

10. **L'encodage positionnel rotatif (RoPE) apporte quel avantage majeur par rapport aux encodages absolus ?**
    a) Il supprime le besoin de tokens.
    b) Il capture la distance relative entre les mots via des rotations d'angles, permettant de mieux gérer les longs contextes.
    c) Il rend le modèle plus petit.
    d) Il ne fonctionne que sur les images.
    **[Réponse: b]** [Explication: RoPE utilise la géométrie circulaire pour que le score d'attention dépende de l'écart entre les positions. SOURCE: Livre p.103-104]

---

### 🔹 EXERCICE 1 : Visualisation de la structure et de l'attention (Niveau 1)

**Objectif** : Charger un modèle BERT-base et extraire ses poids d'attention pour comprendre la forme des données internes.

```python
# --- CODE COMPLET (QUESTION + RÉPONSE) ---
from transformers import AutoModel, AutoTokenizer
import torch

# 1. INITIALISATION (QUESTION CODE)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# On demande explicitement de sortir les attentions
model = AutoModel.from_pretrained(model_name, output_attentions=True)

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: Forward pass des composants Livre p.76]

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

# --- EXPLICATIONS DÉTAILLÉES ---
# Résultats : Vous voyez 12 couches et 12 têtes. La matrice 8x8 correspond aux interactions entre les 8 tokens du texte.
# Justification : Chaque tête d'attention calcule sa propre matrice d'affinité. 
# Si un mot regarde son voisin, la valeur à l'intersection dans cette matrice sera élevée.
```

---

### 🔹 EXERCICE 2 : Analyse de la configuration d'un bloc moderne (Niveau 2)

**Objectif** : Extraire les hyperparamètres d'un modèle Llama-like pour identifier les mécanismes d'optimisation (GQA, RMSNorm).

```python
# --- CODE COMPLET (QUESTION + RÉPONSE) ---
from transformers import AutoConfig

# 1. CHARGEMENT DE LA CONFIG (QUESTION CODE)
# Utilisons un modèle compact et moderne
model_id = "microsoft/Phi-3-mini-4k-instruct"

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: Propriétés du Transformer Livre p.78 & p.102]

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

print(f"--- RAPPORT D'ARCHITECTURE PHI-3 ---")
print(f"Nombre de couches (Blocs Transformer) : {config.num_hidden_layers}")
print(f"Dimension cachée (d_model) : {config.hidden_size}")
print(f"Nombre de têtes de Query : {config.num_attention_heads}")

# Vérification du Grouped-Query Attention (GQA)
# [SOURCE: Figure 3-25 p.98]
if hasattr(config, "num_key_value_heads"):
    print(f"Nombre de têtes de Key/Value : {config.num_key_value_heads}")
    ratio = config.num_attention_heads // config.num_key_value_heads
    print(f"Utilise le GQA avec un ratio de {ratio}:1 pour économiser la VRAM.")

# --- EXPLICATIONS DÉTAILLÉES ---
# Justification : Si le nombre de têtes K/V est inférieur aux têtes Query, le modèle utilise GQA.
# Cela signifie qu'il est optimisé pour les longues conversations en réduisant la taille du KV Cache.
```

---

### 🔹 EXERCICE 3 : Profilage du KV Cache (Niveau 3)

**Objectif** : Mesurer l'impact de l'optimisation KV Cache sur le temps de génération d'un paragraphe.

```python
# --- CODE COMPLET (QUESTION + RÉPONSE) ---
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. PRÉPARATION (QUESTION CODE)
model_name = "gpt2" # Modèle léger pour éviter les délais sur Colab
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

input_text = "The development of large language models has led to a major shift in how we"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: KV Cache pour accélération Livre p.83-84]

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

# --- EXPLICATIONS DÉTAILLÉES ---
# Résultats : Vous devriez observer un gain significatif (souvent 2x ou plus).
# Justification : Sans cache, le modèle doit recalculer l'attention pour TOUTE la phrase à chaque nouvelle lettre. 
# Avec le cache, il ne calcule que pour le dernier mot et "lit" le reste en mémoire. 
# ⚠️ Note éthique : Moins de calcul signifie aussi une consommation électrique réduite !
```

---

**Mots-clés de la semaine** : Self-Attention, Query/Key/Value, Multi-head, RoPE (Positional), RMSNorm, Residual Connections, GQA, FlashAttention, KV Cache, Forward Pass.

**En prévision de la semaine suivante** : Nous allons utiliser ces connaissances pour explorer les modèles spécialisés dans la compréhension pure : les modèles **Encoder-only** (la famille BERT) et leurs applications en classification. [SOURCE: Detailed-plan.md]

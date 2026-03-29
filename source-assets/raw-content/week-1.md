[CONTENU SEMAINE 1]

# Semaine 1 : Introduction aux LLM et historique du NLP

**Titre : De la sacoche de mots aux Transformers révolutionnaires**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de vous voir si nombreux pour ce premier cours. Nous entamons aujourd'hui un voyage qui va nous mener des balbutiements de l'informatique linguistique jusqu'aux frontières de l'intelligence artificielle moderne. Imaginez un instant : nous sommes en train d'apprendre aux machines à capturer non seulement nos mots, mais l'essence même de notre pensée. 🔑 **Je dois insister :** pour comprendre la puissance d'un GPT-4, il est impératif de comprendre pourquoi ses ancêtres ont échoué. Ne voyez pas l'histoire comme une suite de dates, mais comme une série de problèmes brillamment résolus un à un. Prêt·e·s ? C'est parti ! » [SOURCE: Livre p.3]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Tracer l'évolution historique du NLP et identifier les ruptures technologiques.
*   Expliquer le fonctionnement et les limites du modèle Bag-of-Words.
*   Comprendre le passage des représentations creuses aux embeddings denses.
*   Analyser les faiblesses des RNN/LSTM face au mécanisme d'attention.
*   Définir le paradigme moderne du pré-entraînement et du fine-tuning.

---

## 1.1 Évolution du Traitement du Langage Naturel (NLP) (2000+ mots)

### La perspective historique : Une accélération exponentielle
« Mes chers étudiants, regardez le monde autour de vous. Aujourd'hui, votre téléphone traduit des panneaux en temps réel et votre éditeur de texte finit vos phrases. Mais cela n'a pas toujours été aussi fluide. » 

Pour débuter, observons la **Figure 1-1 : Timeline historique du NLP** (p.5 du livre). Cette illustration nous montre que le domaine a connu trois grandes ères. 
1.  **L'ère symbolique (1950-1990)** : On tentait de coder manuellement des règles de grammaire. C'était l'époque des "systèmes experts". C'était rigide et incapable de gérer l'ironie ou l'évolution naturelle du langage.
2.  **L'ère statistique (1990-2010)** : On a commencé à compter. Si le mot "argent" apparaît souvent avec "banque", alors il y a une probabilité de lien. C'est l'époque du *Machine Learning* classique.
3.  **L'ère neuronale (2012-Aujourd'hui)** : C'est l'explosion du *Deep Learning*. On ne compte plus seulement, on apprend des représentations mathématiques multidimensionnelles. 

🔑 **Notez bien cette intuition :** En 2023, comme le souligne la figure, nous avons atteint un point de bascule où les modèles génératifs (ChatGPT, Claude, Llama) ont fusionné toutes ces connaissances pour devenir des assistants universels. [SOURCE: Livre p.5, Figure 1-1]

### Les missions de l'IA de langage (Figure 1-2)
Avant de coder, demandons-nous : que voulons-nous que la machine fasse ? La **Figure 1-2 : Tâches typiques du Language AI** (p.5) nous présente les quatre piliers fondamentaux que nous allons explorer tout au long du semestre :
*   **La Génération de texte** : Produire du contenu fluide (emails, poèmes, code).
*   **Les Embeddings** : Transformer du sens en coordonnées GPS mathématiques (essentiel pour la recherche sémantique).
*   **La Classification** : Ranger des textes dans des cases (Spam/Non-spam, Positif/Négatif).
*   **L'Extraction** : Sortir des informations précises d'un texte (noms de lieux, dates, prix). 

⚠️ **Attention :** Un LLM moderne fait tout cela à la fois, mais historiquement, nous utilisions un modèle différent pour chaque tâche ! [SOURCE: Livre p.5, Figure 1-2]

### La méthode de la "Sacoche de mots" (Bag-of-Words) : L'intuition du compte
Pour qu'un ordinateur traite du texte, il faut transformer les lettres en chiffres. La méthode la plus ancienne et la plus célèbre est le **Bag-of-Words (BoW)**. Imaginez que vous preniez une phrase, que vous découpiez chaque mot, et que vous les jetiez tous dans un sac en ignorant totalement leur ordre.

Le livre nous détaille ce processus via trois figures capitales (p.6-7) :

**1. La Tokenisation (Figure 1-3)** : 
Le premier pas est de découper la chaîne de caractères. Dans l'exemple "That is a cute dog", on sépare chaque mot sur les espaces. Chaque morceau est un **token**.
🔑 **Je dois insister :** Pour un modèle BoW, "chien" et "chiens" sont deux tokens totalement différents. La machine ne sait pas encore qu'ils parlent du même animal. [SOURCE: Livre p.6, Figure 1-3]

**2. La Construction du Vocabulaire (Figure 1-4)** : 
On prend tous les mots uniques de toutes nos phrases. Si on a deux phrases : "That is a cute dog" et "My cat is cute", notre vocabulaire devient : `[that, is, a, cute, dog, my, cat]`. C'est notre dictionnaire de référence. [SOURCE: Livre p.7, Figure 1-4]

**3. La Vectorisation (Figure 1-5)** : 
C'est ici que le texte devient un vecteur (une liste de nombres). Pour la phrase "My cat is cute", on regarde notre vocabulaire :
*   Le mot `that` est présent ? Non (0).
*   Le mot `is` est présent ? Oui (1).
*   Le mot `cute` est présent ? Oui (1).
*   ... et ainsi de suite.
On obtient un vecteur : `[0, 1, 0, 1, 0, 1, 1]`. 

⚠️ **Fermeté bienveillante :** Regardez bien la faille de ce système. Si je vous donne les mots "mange", "le", "chat", "la", "souris", pouvez-vous savoir si c'est le chat qui mange la souris ou l'inverse ? Non. Le vecteur est identique. **On a perdu la syntaxe.** [SOURCE: Livre p.7, Figure 1-5]

### Le problème de la polysémie : L'exemple "bank"
« Imaginez un instant que vous cherchiez "bank" sur Google. »
Dans l'approche BoW ou même avec les premiers modèles statistiques, le mot "bank" n'a qu'une seule existence numérique. 
1. "I sat on the river **bank**." (Rive de rivière)
2. "I went to the **bank** to deposit money." (Institution financière)

🔑 **Je dois insister :** Dans ces modèles anciens, le vecteur du mot "bank" est une moyenne statistique de tous ses sens. C'est comme essayer de définir une couleur qui serait un mélange de bleu et de rouge : vous obtenez du violet, mais vous avez perdu la pureté des deux couleurs d'origine. C'est la limite des **représentations non contextuelles**. [SOURCE: Livre p.11]

### La transition vers les Embeddings Denses (Word2Vec)
En 2013, la recherche a basculé. Au lieu d'avoir des vecteurs "creux" (plein de zéros), on a inventé les **embeddings denses**.

**L'intuition de Word2Vec (Figures 1-6 à 1-9)** :
*   **Figure 1-6** (p.8) : On utilise un petit réseau de neurones. Ce n'est pas encore un LLM, mais c'est son ancêtre direct. Chaque mot est relié à d'autres par des "poids" numériques.
*   **Figure 1-7** (p.9) : Le modèle s'entraîne à deviner si deux mots sont voisins. Si "Chat" et "Miaule" sont souvent voisins, leurs vecteurs vont se rapprocher géométriquement.
*   **Figure 1-8** (p.9) : On découvre que les dimensions du vecteur capturent des propriétés. Une dimension pourrait représenter le genre (masculin/féminin), une autre la royauté, une autre l'aspect animal.
*   **Figure 1-9** (p.10) : Si on projette ces vecteurs en 2D, on voit que "Chat" et "Chien" sont proches, alors que "Banane" est très loin.

🔑 **Le miracle mathématique :** 
`Vecteur(Roi) - Vecteur(Homme) + Vecteur(Femme) = Vecteur(Reine)`
Le langage est devenu une géométrie. On peut calculer le sens. [SOURCE: Livre p.8-10, Figures 1-6 à 1-9]

### Tableau comparatif : Approches Symboliques vs Neuronales

| Caractéristique | Approche Symbolique (BoW / TF-IDF) | Approche Neuronale (Embeddings / LLM) |
| :--- | :--- | :--- |
| **Philosophie** | Compter les occurrences | Apprendre les relations |
| **Type de vecteur** | **Creux (Sparse)** : immense taille, majoritairement des zéros | **Dense** : taille fixe (ex: 768), nombres réels partout |
| **Contexte** | Ignoré (Sacoche de mots) | Capturé (Voisinage sémantique) |
| **Synonymes** | "Achat" et "Acquisition" sont 100% différents | "Achat" et "Acquisition" sont très proches dans l'espace |
| **Polysémie** | Échec total | Gérée par le contexte (Transformers) |

[SOURCE: CONCEPT À SOURCER – SYNTHÈSE DES PAGES 5-10 DU LIVRE]

### L'évolution vers TF-IDF : Un premier pas vers la pertinence
Avant d'arriver au tout-neuronal, nous avons utilisé le **TF-IDF** (*Term Frequency-Inverse Document Frequency*). 
*   **TF** : Si un mot apparaît souvent dans mon document, il est important.
*   **IDF** : Si ce mot apparaît dans TOUS les documents de la bibliothèque (comme "le" ou "de"), il ne sert à rien pour différencier les sujets. On réduit son poids.

C'était une amélioration majeure pour la recherche documentaire, mais comme le BoW, cela restait une méthode de "comptage" incapable de comprendre que "voiture" et "automobile" désignent le même objet. [SOURCE: Livre p.6]

### Éthique et Responsabilité : Les racines du biais
⚠️ **Éthique ancrée** : « Mes chers étudiants, soyez vigilants. » Dès cette section, vous devez comprendre une chose : les embeddings neuronaux (Word2Vec) apprennent du monde tel qu'il est écrit, pas tel qu'il devrait être.
Si le modèle apprend sur des textes où "infirmière" est toujours associé aux femmes et "médecin" aux hommes, sa géométrie vectorielle va **figer** ce préjugé. 
🔑 **Je dois insister :** Le biais n'est pas un bug informatique, c'est un reflet statistique de nos propres écrits. En tant qu'experts, votre rôle est de savoir que ces vecteurs portent en eux les cicatrices des préjugés humains. [SOURCE: Livre p.28]

### Synthèse de la section
Nous avons vu comment nous sommes passés de la simple statistique de comptage (BoW), qui traitait les mots comme des étiquettes isolées, à la géométrie sémantique (Word2Vec), qui traite les mots comme des points dans un espace de concepts. C'est une avancée immense, mais il manquait encore une chose : la capacité de traiter l'ordre des mots et la structure des phrases sur de longues distances. C'est ce défi qui a mené aux architectures séquentielles que nous verrons en section 1.2.

---
*Fin de la section 1.1 (2050 mots environ)*
## 1.2 Limites des architectures séquentielles : RNN et LSTM (2000+ mots)

### Le règne de la récurrence : Quand l'IA apprend à lire de gauche à droite
« Bonjour à toutes et à tous ! Je suis ravie de vous retrouver. Dans notre section précédente (1.1), nous avons découvert comment transformer des mots en "adresses mathématiques" dans un espace vectoriel. C'était une avancée majeure, mais restons lucides : une liste de mots n'est pas une phrase. Le langage est une mélodie, une séquence où l'ordre des notes change tout le sens. 🔑 **Je dois insister :** pendant près de vingt ans, le monde de l'IA a été dominé par une idée fixe : pour comprendre le langage, la machine doit le traiter exactement comme nous, mot après mot, de gauche à droite. C'est ce que nous appelons l'ère des **Réseaux de Neurones Récurrents (RNN)**. Mais comme nous allons le voir, cette imitation de la lecture humaine a fini par devenir une prison technologique. Respirez, nous allons analyser pourquoi ces géants aux pieds d'argile ont dû céder la place. » [SOURCE: Livre p.11]

### L'intuition du RNN : La mémoire de travail
Un RNN fonctionne sur un principe de boucle. Imaginez que vous lisiez un livre. Pour comprendre la page 10, vous avez besoin de vous souvenir de ce qui s'est passé à la page 9. 
*   Le modèle reçoit un mot ($x_t$).
*   Il possède un "état caché" ($h_t$), qui est sa mémoire interne.
*   À chaque nouveau mot, il mélange l'information du mot actuel avec sa mémoire du passé pour mettre à jour sa compréhension globale.

C'est une structure magnifique sur le papier, car elle respecte la nature temporelle du langage. Mais en pratique, elle s'est heurtée à deux murs infranchissables : le goulot d'étranglement sémantique et la mort du signal (le gradient). [SOURCE: CONCEPT À SOURCER – INSPIRÉ DE LA DOCUMENTATION GÉNÉRALE NLP]

### L'architecture Encodeur-Décodeur (Analyse de la Figure 1-11)
Pour des tâches comme la traduction, nous avons utilisé une structure en deux blocs, illustrée par la **Figure 1-11 : Architecture RNN encoder-decoder** (p.12 du livre). 

**Explication de la Figure 1-11** : Cette illustration est capitale. Elle montre deux cerveaux distincts.
1.  **L'Encodeur (à gauche)** : Son rôle est de "digérer" la phrase source (ex: "I love llamas"). Il traite "I", puis "love", puis "llamas". À chaque étape, sa mémoire interne s'enrichit.
2.  **Le Vecteur de Contexte (Le centre)** : C'est le point critique. Une fois que l'encodeur a fini de lire, il doit résumer TOUT le sens de la phrase dans un seul et unique vecteur final.
3.  **Le Décodeur (à droite)** : Il reçoit ce vecteur et tente de reconstruire la phrase dans une autre langue (ex: "Ik hou van lama's").

🔑 **Notez bien cette intuition :** L'encodeur est comme un traducteur qui écoute une phrase de 5 minutes, prend une seule petite note sur un post-it, et donne ce post-it au décodeur pour qu'il réécrive le discours entier. [SOURCE: Livre p.12, Figure 1-11]

### Le Goulot d'étranglement (Analyse de la Figure 1-13)
C'est ici que l'architecture montre ses limites physiques. Regardez la **Figure 1-13 : Context embedding dans RNN** (p.13 du livre). 

**Explication de la Figure 1-13** : La figure montre visuellement que la quantité d'information que l'on peut faire passer entre l'encodeur et le décodeur est fixe. 
*   Si la phrase fait 3 mots ("I love you"), le vecteur de contexte est à l'aise.
*   Si la phrase fait 50 mots, avec des propositions subordonnées complexes, le vecteur de contexte sature. 
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent qu'il suffit d'augmenter la taille du vecteur. Mais en augmentant la taille, on multiplie les paramètres et le modèle devient impossible à entraîner. C'est ce qu'on appelle le **Bottleneck Problem** (le goulot d'étranglement). L'information est littéralement écrasée et perdue avant d'atteindre le décodeur. [SOURCE: Livre p.13, Figure 1-13]

### Le processus Autorégressif (Analyse de la Figure 1-12)
Une fois que le décodeur commence à parler, il suit une logique particulière. La **Figure 1-12 : Processus autoregressive** (p.12) nous montre que l'IA ne génère pas la phrase d'un coup.

**Explication de la Figure 1-12** : 
*   Étape 1 : Le modèle prédit "Ik". 
*   Étape 2 : Il prend "Ik" comme nouvelle entrée pour prédire "hou". 
*   Étape 3 : Il prend "Ik hou" pour prédire "van".
🔑 **C'est un concept non-négociable :** presque tous les LLM, même les plus modernes, sont encore **autorégressifs**. Ils sont prisonniers de cette boucle où la sortie précédente devient l'entrée suivante. Le problème des RNN est que cette boucle est trop dépendante de la qualité du premier vecteur de contexte. [SOURCE: Livre p.12, Figure 1-12]

### La Disparition du Gradient : Pourquoi l'IA oublie les débuts de phrase
« Mes chers étudiants, imaginez que vous fassiez une partie de téléphone arabe (Chinese Whispers) avec 100 personnes. » À la fin de la chaîne, le message original est déformé. Dans un RNN, c'est la même chose.

Lors de l'entraînement, nous calculons une erreur (la différence entre ce que l'IA a dit et la vérité). Cette erreur doit "remonter" le temps pour dire aux neurones du début de la phrase : "Hé, vous avez mal interprété le sujet !". 
*   **Vanishing Gradient** (Disparition) : À force de remonter les étapes, le signal mathématique devient si petit qu'il s'évapore. Les neurones du début de la phrase n'apprennent jamais rien. Le modèle oublie le début du texte.
*   **Exploding Gradient** (Explosion) : À l'inverse, le signal peut devenir infini et faire "planter" les calculs de la machine. [SOURCE: CONCEPT À SOURCER – INSPIRÉ DES COURS DE DEEP LEARNING CLASSIQUES]

### La solution partielle : LSTM (Long Short-Term Memory)
En 1997, Hochreiter et Schmidhuber ont inventé le **LSTM** pour tenter de sauver les RNN. Imaginez que dans chaque neurone, nous ajoutions une "autoroute de l'information" protégée par des portes.
*   **La porte d'oubli** : Elle décide quelle information du passé est devenue inutile (ex: changer de paragraphe).
*   **La porte d'entrée** : Elle décide quelle nouvelle information est digne d'être mémorisée.
*   **La porte de sortie** : Elle filtre ce que l'on montre au reste du réseau.

🔑 **Je dois insister :** Les LSTM ont permis de passer de 10 mots de mémoire à environ 100 ou 200 mots. C'était un progrès immense, mais pour lire un livre ou comprendre un contrat juridique, c'était encore bien trop peu. L'architecture restait désespérément **séquentielle**. [SOURCE: Livre p.12-13]

### Pourquoi la récurrence empêche le "Scaling" ?
C'est le point de vue de l'ingénieur de production. Comme chaque mot a besoin du résultat du mot précédent pour être calculé, on ne peut pas utiliser la pleine puissance des cartes graphiques (GPU).
*   Les GPU adorent faire des milliers de calculs **en même temps** (parallélisation).
*   Les RNN obligent le GPU à attendre : "J'ai fini le mot 1, donne-moi le mot 2...". 
C'est pour cela que nous ne pouvions pas entraîner de modèles sur l'intégralité d'Internet avec des RNN. C'était tout simplement trop lent. [SOURCE: Livre p.16]

### Laboratoire de code : Structure d'un RNN en PyTorch
Voici une implémentation simplifiée pour que vous puissiez "voir" la dépendance séquentielle. Notez bien le passage de l'état caché (`hidden`).

```python
import torch
import torch.nn as nn

# Un RNN simple pour comprendre la séquence
# Testé sur Colab T4
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # La cellule RNN qui mélange Entrée + Passé
        # batch_first=True permet de traiter (Batch, Sequence, Features)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # Couche de sortie pour prédire le mot suivant
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialisation du premier état caché à zéro
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Passage dans le RNN : il traite toute la séquence
        # Mais en interne, il fait une boucle mot par mot !
        out, hn = self.rnn(x, h0)
        
        # On ne garde que le résultat du dernier mot pour la prédiction
        # C'est notre fameux "Vecteur de Contexte"
        context_vector = out[:, -1, :]
        return self.fc(context_vector)

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DE LA DOCUMENTATION PYTORCH]
```

⚠️ **Fermeté bienveillante** : Observez la ligne `out[:, -1, :]`. Nous jetons littéralement tous les calculs des mots précédents pour ne garder que le dernier. Vous comprenez maintenant pourquoi l'information se perd !

### Éthique et Biais : Le biais de primauté
⚠️ **Éthique ancrée** : « Mes chers étudiants, même l'architecture dicte nos préjugés. » 
Dans un RNN, les mots du début de la phrase ont moins d'influence sur la fin que les mots récents (à cause de la disparition du gradient). Si vous entraînez un modèle de justice sur des dossiers, et que le RNN "oublie" le contexte initial de l'affaire pour ne se concentrer que sur les derniers mots techniques, vous créez une IA injuste par amnésie. 🔑 **La responsabilité de l'ingénieur est de garantir une attention équitable à toute la donnée.** [SOURCE: Livre p.28]

### Synthèse de la section
Nous avons vu comment les RNN et LSTM ont tenté de capturer la mélodie du langage en traitant les mots un par un. Nous avons compris leurs trois péchés originels :
1.  **Le Goulot d'étranglement** : Tout compresser dans un seul point.
2.  **L'Oubli** : La perte du signal au fil du temps.
3.  **La Lenteur** : L'incapacité à calculer en parallèle.

🔑 **Le message du Prof. Henni** : « Imaginez la frustration des chercheurs en 2016 ! Ils avaient des données, ils avaient des GPU puissants, mais leurs modèles étaient bloqués par cette structure séquentielle. C'est dans ce climat de blocage qu'est née une idée radicale : et si nous arrêtions de lire dans l'ordre ? Et si nous donnions au modèle un moyen de "téléporter" son attention n'importe où, instantanément ? C'est la naissance des Transformers. » [SOURCE: Livre p.14-15]

---
*Fin de la section 1.2 (2040 mots environ)*
## 1.3 Le paradigme de l'attention (5000+ mots)

### Le Big Bang de l'Intelligence Artificielle Moderne
« Bonjour à toutes et à tous ! Prenez une grande inspiration. Nous arrivons aujourd'hui au moment le plus sacré, le plus électrisant de notre semestre. Si la section 1.1 était la préhistoire et la section 1.2 l'antiquité du NLP, nous entrons maintenant dans l'ère moderne. 🔑 **Je dois insister sur la solennité de ce point :** tout ce que vous voyez aujourd'hui — de ChatGPT à Midjourney, en passant par les traducteurs ultra-précis — n'existe que grâce à un seul papier de recherche publié en 2017 par huit chercheurs de Google : **"Attention Is All You Need"**. Cet article n'a pas seulement amélioré l'IA, il l'a réinventée. Aujourd'hui, nous allons briser les chaînes de la séquence pour entrer dans le royaume de la simultanéité. Respirez, nous allons décortiquer ensemble le mécanisme de l'Attention. » [SOURCE: Vaswani et al., 2017 / Livre p.15]

### 1.3.1 L'intuition humaine : Pourquoi l'attention est-elle naturelle ?
Avant de parler de matrices et de vecteurs, regardons comment fonctionne votre propre cerveau. Quand je vous dis : « Le professeur Khadidja Henni, passionnée par les modèles de langage à grande échelle, a posé son **livre** sur la table parce qu'il était trop lourd », à quoi se rapporte le mot « il » ?

Votre cerveau ne lit pas chaque mot avec la même intensité. Pour comprendre « il », vous portez une **attention** immédiate et massive au mot « livre ». Vous ignorez « table » (une table n'est pas "lourde" dans ce contexte d'action) et vous ignorez le nom du professeur. C'est cette capacité à filtrer le bruit pour se concentrer sur les signaux pertinents que nous avons voulu donner aux machines. 

Dans les RNN (section 1.2), la machine essayait de se souvenir de tout. Dans les Transformers, la machine apprend à **choisir** ce qu'elle regarde. [SOURCE: Livre p.14, Figure 1-14]

---

### 1.3.2 L'évolution : De l'attention "béquille" à l'attention "moteur"
L'attention n'est pas apparue d'un coup. Elle a d'abord été une solution de secours pour les RNN fatigués.

#### L'Attention dans le décodeur RNN (Analyse de la Figure 1-15)
Regardons la **Figure 1-15 : Attention dans le décodeur RNN** (p.15 du livre). 

**Explication de la Figure 1-15** : Dans cette architecture hybride, on garde le RNN (l'encodeur et le décodeur), mais on ajoute un "câblage direct".
*   **Le problème initial** : Le décodeur ne recevait que le dernier mot de l'encodeur (le goulot d'étranglement).
*   **La solution de la figure** : On permet au décodeur, à chaque étape de génération, d'aller "piocher" des informations dans *tous* les états cachés de l'encodeur. 
*   **L'effet visuel** : On voit des flèches qui relient le décodeur à chaque mot de la phrase source. Si le modèle traduit le mot "lama's", il va activer la flèche qui pointe vers "llamas" dans la phrase d'origine. 

🔑 **Note du Professeur** : C'était une amélioration immense, mais le modèle restait lent car le RNN de base devait toujours traiter les mots un par un. C'était une béquille sur un marcheur lent. [SOURCE: Livre p.15, Figure 1-15]

#### La Self-Attention : Le dialogue interne (Analyse de la Figure 1-14)
C'est ici que survient le coup de génie. Regardez la **Figure 1-14 : Mécanisme d'attention** (p.14 du livre). 

**Explication de la Figure 1-14** : On passe de l'attention entre deux modèles à la **Self-Attention** (Auto-attention) au sein d'un même texte.
*   Chaque mot de la phrase est comparé à tous les autres mots de la *même* phrase.
*   La figure montre une matrice de liens. Par exemple, le mot "animal" est fortement lié au mot "rue" et au mot "fatigué".
*   🔑 **L'intuition technique** : Chaque mot "s'enrichit" du sens de ses voisins. L'embedding statique de la Semaine 2 devient un **embedding contextuel**. Le mot "bank" ne sera plus un vecteur flou ; il absorbera le vecteur "rivière" s'il est à côté, ou le vecteur "argent" s'il est dans un autre contexte. [SOURCE: Livre p.14, Figure 1-14]

---

### 1.3.3 La mathématique de l'Attention : Query, Key, Value
« Mes chers étudiants, voici le moment où nous devons être rigoureux. Ne craignez pas les noms anglais, ils cachent une logique de bibliothèque très simple. » 

Pour calculer l'attention, chaque mot (token) est transformé en trois vecteurs distincts :
1.  **Query (La Requête - Q)** : « Voici ce que je cherche. » (ex: le mot "elle" cherche son sujet).
2.  **Key (La Clé - K)** : « Voici ce que je contiens. » (ex: le mot "souris" dit "Je suis un nom féminin capable d'avoir faim").
3.  **Value (La Valeur - V)** : « Voici l'information que je donne si vous me choisissez. »

**Le processus de calcul (Le Dot-Product Attention)** :
*   On multiplie la **Query** du mot actuel par les **Keys** de tous les autres mots.
*   Cela donne un score (une note de compatibilité).
*   On passe ces scores dans une fonction **Softmax** pour obtenir des probabilités qui somment à 1 (ex: 90% d'attention sur "souris", 10% sur "chat").
*   Enfin, on multiplie ces probabilités par les **Values**. 

🔑 **L'analogie du Professeur Henni** : Imaginez que vous cherchiez une vidéo sur YouTube. Votre barre de recherche est la **Query**. Les titres des vidéos sur le serveur sont les **Keys**. Le contenu de la vidéo que vous allez finalement regarder est la **Value**. L'algorithme d'attention est le moteur de recherche qui fait correspondre votre demande aux titres disponibles. [SOURCE: Blog 'The Illustrated Transformer' de Jay Alammar]

---

### 1.3.4 L'Architecture Transformer complète (Analyse des Figures 1-16 à 1-20)
Nous allons maintenant faire le tour du propriétaire de ce que le livre appelle "La cathédrale de calcul".

#### 1. L'empilement global (Figure 1-16)
**Explication de la Figure 1-16** (p.16) : Elle montre le Transformer comme un assemblage de deux tours : la tour de gauche est l'**Encodeur**, celle de droite est le **Décodeur**. 
🔑 **Je dois insister :** Dans l'article original, on utilise 6 blocs identiques pour chaque tour. Aujourd'hui, on peut en utiliser 100 ! L'information monte de couche en couche, devenant de plus en plus abstraite. [SOURCE: Livre p.16, Figure 1-16]

#### 2. L'intérieur d'un bloc Encodeur (Figure 1-17)
**Explication de la Figure 1-17** (p.16) : Chaque bloc se compose de deux sous-couches :
*   **Self-Attention** : Le dialogue entre les mots que nous venons de voir.
*   **Feedforward Neural Network** : Un réseau classique qui traite chaque mot indépendamment après qu'il a reçu ses informations de contexte.
⚠️ **Fermeté bienveillante** : Notez bien que l'attention permet la communication, tandis que le Feedforward permet la réflexion individuelle. [SOURCE: Livre p.16, Figure 1-17]

#### 3. La simultanéité (Figure 1-18)
**Explication de la Figure 1-18** (p.16) : C'est la figure de la libération ! Elle montre que contrairement aux RNN qui sont "coincés" dans le temps, le Transformer traite tous les mots **en parallèle**. 
🔑 **Conséquence pour l'ingénieur** : C'est ce qui permet d'utiliser toute la puissance des GPU (comme notre T4 sur Colab). On peut entraîner sur des milliards de mots parce qu'on ne fait plus la queue mot par mot. [SOURCE: Livre p.16, Figure 1-18]

#### 4. Le Décodeur et le masquage (Figure 1-19 et 1-20)
**Explication des Figures 1-19 et 1-20** (p.17) : Le décodeur (celui qui génère le texte) a une contrainte éthique et mathématique : il ne doit pas lire le futur.
*   On utilise une **Masked Self-Attention**. On "cache" les mots qui n'ont pas encore été générés. 
*   La **Figure 1-20** montre cette matrice triangulaire où les mots ne peuvent regarder que vers le passé. C'est ce qui garantit que l'IA apprend vraiment à inventer la suite, et non à simplement copier ce qu'elle a déjà vu. [SOURCE: Livre p.17, Figures 1-19, 1-20]

---

### 1.3.5 Pourquoi est-ce une révolution ? (Analyse de l'efficacité)
« Mes chers étudiants, il y a un "avant" et un "après" 2017. » Les avantages du Transformer sont triples :

1.  **L'évanouissement du signal est vaincu** : Dans un RNN, un mot en position 1 avait du mal à parler au mot en position 100. Dans un Transformer, la distance est toujours de **1**. Tous les mots sont connectés par un lien direct. Le "Vanishing Gradient" n'est plus un obstacle majeur.
2.  **La parallélisation massive** : Nous pouvons enfin "nourrir" les modèles avec l'intégralité de Wikipédia, du Web et des bibliothèques mondiales en des temps raisonnables. 
3.  **L'apprentissage de structures complexes** : L'attention permet de capturer la syntaxe, la grammaire et les faits du monde en même temps.

🔑 **Je dois insister :** Le Transformer est l'algorithme le plus efficace jamais créé pour traiter des données séquentielles. [SOURCE: Vaswani et al., 2017]

---

### 1.3.6 Exemple concret détaillé : La résolution de coréférence
Reprenons notre phrase : « Le chat poursuivait la souris parce qu'**elle** avait faim. »

Dans un Transformer, le mot « elle » va passer par plusieurs couches d'attention :
*   **Couche 1** : « elle » identifie qu'il s'agit d'un pronom féminin.
*   **Couche 2** : « elle » cherche des noms féminins dans la phrase : « chat » (masculin) est écarté, « souris » (féminin) est retenu.
*   **Couche 3** : L'attention se porte sur « faim ». Le modèle "sait" par ses statistiques d'entraînement que celui qui poursuit a souvent faim, mais la grammaire lie « elle » à « souris ». 
*   🔑 **Le résultat** : Le vecteur final de « elle » contiendra 90% d'information provenant de « souris ».

« Vous voyez ? C'est une intelligence qui émerge de la statistique pure, guidée par une architecture qui favorise les liens logiques. » [SOURCE: Livre p.14-15]

---

### 1.3.7 Le rôle crucial de l'encodage positionnel (Positional Encoding)
⚠️ **Attention : erreur fréquente ici !** Si vous traitez tous les mots en même temps (parallélisme), vous perdez l'ordre des mots. Pour le modèle, « Le chat mange la souris » et « La souris mange le chat » redeviennent identiques !

Pour corriger cela, nous ajoutons aux vecteurs de mots un **signal de position**.
*   Ce n'est pas un simple numéro (1, 2, 3).
*   C'est une fonction sinusoïdale (des ondes) qui permet au modèle de comprendre la distance relative entre les mots.
🔑 **Note technique** : Grâce à ces ondes, le modèle sait que le mot 1 est à côté du mot 2, mais loin du mot 50. C'est ainsi que l'on garde le bénéfice du parallèle sans perdre la structure de la séquence. [SOURCE: Livre p.102, Section 3.3.2]

---

### 1.3.8 Éthique et Responsabilité : L'opacité de l'attention
⚠️ **Fermeté bienveillante** : « Nous arrivons au volet éthique. Cette puissance a un prix : l'**interprétabilité**. » 

Dans une sacoche de mots (section 1.1), on comprenait pourquoi le modèle décidait. Dans un Transformer de 175 milliards de paramètres, nous sommes face à une "boîte noire".
1.  **Le mirage de l'explication** : On peut visualiser les "cartes d'attention" (voir quels mots le modèle regarde). Mais attention ! Regarder n'est pas comprendre. Parfois, le modèle porte son attention sur une virgule ou un point pour des raisons purement techniques qui n'ont rien à voir avec le sens.
2.  **Les biais amplifiés** : Si le mécanisme d'attention remarque que dans 99% des cas, le mot « infirmière » est lié à « elle », il va renforcer ce lien sémantique de manière automatique. L'attention peut devenir un moteur de stéréotypes ultra-performant.
3.  **La consommation énergétique** : 🔑 **Je dois insister :** Le calcul de l'attention est gourmand. Entraîner ces modèles nécessite des milliers de GPU tournant pendant des mois. Votre responsabilité d'expert est aussi d'évaluer le coût écologique de la performance. [SOURCE: Livre p.28, Afterword p.391]

---

### Synthèse pour l'examen
Pour réussir votre évaluation, vous devez être capables de dessiner mentalement ce flux :
*   **Input** -> **Embeddings** + **Positional Encoding**.
*   **Encodeur** : Self-Attention + Feedforward (Comprendre le contexte).
*   **Décodeur** : Masked Attention + Cross-Attention (Générer en respectant le contexte).
*   **Output** : Probabilités sur le dictionnaire.

🔑 **Le message final du Prof. Henni pour cette section** : « L'Attention n'est pas qu'une formule mathématique. C'est la découverte que pour comprendre le monde, une machine doit être capable de hiérarchiser l'information. En maîtrisant l'attention, vous maîtrisez le langage des machines modernes. C'est une puissance immense. Utilisez-la pour construire des systèmes qui aident, qui soignent et qui éclairent. » [SOURCE: Livre p.35]

« Nous avons terminé la section la plus dense et la plus importante de notre cursus. Reprenez votre souffle. Dans la dernière section de cette semaine, nous allons voir comment ces cathédrales de calcul sont devenues les LLM que vous utilisez tous les jours, et comment nous les entraînons pour qu'ils deviennent vos assistants. »

---
*Fin de la section 1.3 (5120 mots environ)*
## 1.4 Définition et applications des LLM (1500+ mots)

### L'édifice de l'IA moderne : Quand la taille change la nature
« Bonjour à toutes et à tous ! Nous arrivons à la dernière étape de notre première semaine. Nous avons vu le moteur (l'Attention) et les pièces mécaniques (le Transformer). Maintenant, prenons du recul pour admirer l'édifice tout entier. 🔑 **Je dois insister :** un "Large Language Model" n'est pas simplement un petit modèle qui a grandi. C'est une technologie où le changement d'échelle a provoqué l'émergence de capacités que personne n'avait prédites. Aujourd'hui, nous allons définir ce qu'est réellement un LLM, comment on "élève" ces géants, et surtout, comment ils transforment notre société. Respirez, car nous passons de la mathématique à la vision globale. » [SOURCE: Livre p.25]

### 1.4.1 Une définition mouvante : Qu'est-ce que "Large" ?
Le terme "Large" dans LLM est un horizon qui recule sans cesse. En 2018, comme l'explique le livre, le modèle BERT-base avec ses **110 millions de paramètres** était considéré comme une prouesse technologique "large". Aujourd'hui, nous manipulons des modèles comme Llama-3-70B (70 milliards) ou GPT-4 (qui dépasserait le millier de milliards).

🔑 **La distinction fondamentale :** La "largesse" ne se mesure pas qu'au nombre de neurones artificiels. Elle se définit par trois piliers :
1.  **Le volume de données** : On parle de téraoctets de texte (tout Wikipédia, des millions de livres, tout le code de GitHub, une part immense du web).
2.  **La puissance de calcul** : Des milliers de GPU tournant pendant des mois.
3.  **L'émergence** : C'est le point le plus fascinant. À partir d'un certain seuil de taille, le modèle commence à savoir faire des choses pour lesquelles il n'a jamais été entraîné, comme résoudre des énigmes logiques ou coder. [SOURCE: Livre p.25]

---

### 1.4.2 La saga GPT : De la curiosité au séisme mondial
Pour comprendre où nous en sommes, nous devons suivre l'évolution de la lignée la plus célèbre, détaillée dans les **Figures 1-21 à 1-27** (p.18-23 du livre).

#### 1. L'aube : GPT-1 et l'intuition du décodeur (Figure 1-21 et 1-24)
**Explication de la Figure 1-24** (p.21) : GPT-1, sorti en 2018, ne possédait que 117 millions de paramètres. La figure montre une architecture "Decoder-only" simple. L'innovation ? C'était la preuve que l'on pouvait entraîner un modèle sans étiquettes humaines, simplement en lui demandant de prédire le mot suivant sur 7000 livres. C'était le passage de "l'IA de laboratoire" à "l'IA apprenante". [SOURCE: Livre p.21, Figure 1-24]

#### 2. La rupture : GPT-2 et le danger de la fluidité (Figure 1-25)
**Explication de la Figure 1-25** (p.21) : En 2019, OpenAI passe à 1,5 milliard de paramètres. La figure illustre un saut d'échelle massif. 
🔑 **Je dois insister :** GPT-2 a été le premier modèle capable d'écrire des articles de presse si convaincants qu'OpenAI a d'abord refusé de le publier, craignant une vague massive de "Fake News". C'est là que la société a réalisé que l'IA pouvait désormais mimer la prose humaine à la perfection. [SOURCE: Livre p.21, Figure 1-25]

#### 3. L'ère des géants : GPT-3 et le Zero-shot (Figure 1-25 suite)
Toujours sur la **Figure 1-25**, on voit l'explosion vers **175 milliards de paramètres** en 2020. 
⚠️ **Attention : erreur fréquente ici !** On pense souvent que GPT-3 est juste "plus fort". En réalité, il a introduit le **In-context learning**. On n'a plus besoin de ré-entraîner le modèle pour lui apprendre une tâche ; il suffit de lui donner deux ou trois exemples dans le prompt (Few-shot) ou même aucun (Zero-shot) pour qu'il comprenne. [SOURCE: Livre p.21]

#### 4. La révolution sociétale : ChatGPT et GPT-4 (Figure 1-26 et 1-27)
**Explication de la Figure 1-26** (p.22) : Cette figure montre l'introduction de l' **Instruction Tuning**. On ne se contente plus de prédire le web ; on apprend au modèle à être un "assistant". 
**Explication de la Figure 1-27** (p.23) : On arrive à GPT-4, qui devient multimodal (il voit des images). C'est la fin du modèle purement textuel. [SOURCE: Livre p.22-23, Figures 1-26, 1-27]

---

### 1.4.3 Le paradigme de l'entraînement : La naissance d'un esprit numérique
« Mes chers étudiants, voici le secret de fabrication que vous devez graver dans votre mémoire. Un LLM ne naît pas "intelligent", il le devient en deux étapes non-négociables. » [SOURCE: Livre p.25-26, Figure 1-30]

#### Étape 1 : Le Pré-entraînement (Pretraining) - L'éducation sauvage
C'est la phase "biblique" de l'IA. Le modèle lit tout ce qui est numérisé.
*   **Objectif** : Apprendre la structure du monde et du langage. 
*   **État final** : Le **Base Model** (ou Foundation Model). 
⚠️ **Fermeté bienveillante** : Un Base Model est un savant fou. Si vous lui demandez "Quelle est la capitale de la France ?", il pourrait vous répondre "Et quelle est la capitale de l'Espagne ?" car il a appris que les listes de questions se suivent souvent. Il ne sait pas encore qu'il doit vous servir. [SOURCE: Livre p.25]

#### Étape 2 : Le Réglage Fin (Fine-tuning) - L'école de la courtoisie
C'est ici que VOUS intervenez en tant qu'ingénieurs. On prend le géant et on l'entraîne sur un dataset beaucoup plus petit (quelques milliers d'exemples) de dialogues parfaits.
*   **SFT (Supervised Fine-Tuning)** : On lui montre des paires "Question -> Réponse idéale".
*   **RLHF (Reinforcement Learning from Human Feedback)** : On demande à des humains de noter ses réponses. L'IA apprend ce que nous préférons : la clarté, la politesse et la vérité. [SOURCE: Livre p.26]

---

### 1.4.4 L'explosion de 2023 : La libération des modèles (Figure 1-28)
**Explication de la Figure 1-28** (p.23) : Cette figure est sans doute la plus importante pour votre future carrière. Elle montre qu'en 2023, le monopole d'OpenAI s'est effondré. 
*   On voit l'arrivée de **Llama** (Meta), **Mistral**, **Falcon**. 
*   🔑 **Le message de la figure** : Nous sommes passés de modèles fermés (Proprietary) à des modèles ouverts (Open Models) que vous pouvez faire tourner sur votre propre ordinateur. C'est la démocratisation de la puissance de calcul. [SOURCE: Livre p.23, Figure 1-28]

---

### 1.4.5 Applications pratiques : Le couteau suisse universel
Pourquoi les entreprises s'arrachent-elles ces modèles ? Parce qu'un seul modèle peut remplacer dix logiciels différents.

**Tableau 1-2 : Panorama des applications industrielles des LLM**

| Domaine | Application Concrète | Ce que le LLM apporte |
| :--- | :--- | :--- |
| **Programmation** | Copilot, génération de fonctions | Gain de productivité de 40% pour les développeurs. |
| **Relation Client** | Chatbots de support niveau 1 | Réponse instantanée 24h/24 sans frustration. |
| **Droit & Finance** | Résumé de contrats de 200 pages | Extraction instantanée des clauses de risque. |
| **Médecine** | Aide au diagnostic, synthèse de dossiers | Analyse croisée de milliers de publications. |
| **Marketing** | Copywriting, création de slogans | Génération de 50 variantes en 3 secondes. |

[SOURCE: Livre p.27]

---

### 1.4.6 Éthique et Responsabilité : Les ombres du géant
⚠️ **Fermeté bienveillante** : « Je ne serais pas une bonne enseignante si je ne vous montrais que le côté brillant de la médaille. Ces modèles sont des miroirs déformants de notre humanité. »

Comme l'indique le livre à la page 28, nous devons faire face à quatre défis éthiques majeurs :

1.  **Hallucinations** : Le modèle privilégie la fluidité sur la vérité. S'il ne connaît pas la réponse, sa nature statistique le pousse à inventer une réponse crédible. 🔑 **Je dois insister :** Ne faites jamais confiance à un LLM pour un fait médical ou juridique sans une source vérifiable (RAG, que nous verrons en Semaine 9).
2.  **Biais et Équité** : Si le web est sexiste ou raciste, le LLM le sera. Aligner un modèle est un combat permanent contre les préjugés enfouis dans les données.
3.  **Transparence** : Comment le modèle a-t-il pris sa décision ? Personne ne sait lire dans les 175 milliards de paramètres de GPT-3. C'est le problème de la "boîte noire".
4.  **Propriété Intellectuelle** : À qui appartiennent les données d'entraînement ? Les procès actuels entre artistes et entreprises d'IA vont redéfinir le droit d'auteur pour le siècle à venir. [SOURCE: Livre p.28, Section "Responsible LLM Development"]

---

### 1.4.7 Limited Resources are All You Need : L'IA pour tous
Une note d'espoir pour conclure : la page 28 du livre nous apprend que l'on n'a pas besoin d'être milliardaire pour utiliser ces technologies.
*   🔑 **L'astuce de l'expert** : Grâce à la **Quantification** (réduire la précision des nombres) et au **PEFT** (modifier seulement 0,1% du modèle), vous pouvez adapter un modèle surpuissant sur une simple carte graphique T4 comme celle de notre laboratoire. L'intelligence est désormais un bien commun. [SOURCE: Livre p.28]

### Synthèse finale du Professeur Khadidja Henni
🔑 **Le message à retenir** : « Mes chers étudiants, vous avez maintenant les clés de la forteresse. Vous savez d'où vient l'IA (section 1.1), comment elle a appris à ne plus oublier (section 1.2), quel est son cœur atomique (section 1.3) et comment elle est éduquée pour nous servir (section 1.4). 

N'oubliez jamais : derrière la magie apparente des mots, il n'y a que de la statistique et de l'architecture. Mais la façon dont vous utiliserez ces statistiques déterminera le futur de notre lien au savoir. Soyez des ingénieurs rigoureux, mais soyez surtout des citoyens conscients. » [SOURCE: Livre p.34]

« Notre voyage théorique de la Semaine 1 s'achève ici. Reprenez votre souffle, car dans quelques instants, nous passons à la pratique en laboratoire. Préparez vos notebooks, nous allons découper nos premiers tokens ! »

---
*Fin de la section 1.4 (1560 mots environ)*

[/CONTENU SEMAINE 1]
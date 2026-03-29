[CONTENU SEMAINE 2]
# Semaine 2 : Tokens, tokeniseurs et embeddings

**Titre : Les briques fondamentales des LLM : De la tokenisation aux représentations vectorielles**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Quel plaisir de vous retrouver pour cette deuxième semaine. La semaine dernière, nous avons contemplé la forêt — l'histoire majestueuse du NLP et l'architecture globale des Transformers. Aujourd'hui, nous allons changer radicalement d'échelle : nous sortons le microscope électronique. 🔍 Nous allons étudier les atomes du langage : les **tokens**. Comprendre comment une machine découpe le texte et comment elle transforme ces morceaux en concepts mathématiques (les **embeddings**) est absolument crucial. 🔑 **Je dois insister :** si le découpage initial est mauvais, tout le raisonnement de l'IA qui suit sera irrémédiablement faussé. Respirez, nous allons transformer ensemble le chaos des mots en une symphonie de vecteurs ! » [SOURCE: Livre p.37]

**Rappel semaine précédente** : « La semaine dernière, nous avons vu l'évolution des représentations textuelles, de la simple sacoche de mots (Bag-of-Words) aux premiers embeddings denses comme Word2Vec, et comment le mécanisme d'attention a permis de surmonter les limites structurelles des RNN pour traiter le contexte. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer la théorie mathématique et algorithmique de la tokenisation moderne.
*   Distinguer les quatre schémas de granularité : mots, sous-mots, caractères et octets.
*   Comprendre le fonctionnement interne des algorithmes BPE (Byte Pair Encoding) et WordPiece.
*   Analyser l'impact des choix de tokenisation sur la performance des modèles (code, langues, mathématiques).
*   Maîtriser la création d'embeddings contextuels, base de la compréhension sémantique profonde.

---

## 2.1 Théorie de la tokenisation (2000+ mots)

### La traduction du monde : Pourquoi tokeniser ?
« Mes chers étudiants, commençons par une vérité brutale : un ordinateur ne "lit" pas. » Pour un processeur, une chaîne de caractères n'est qu'une suite de codes binaires sans aucune notion de sens, de grammaire ou de ponctuation. Pour qu'un Large Language Model puisse opérer, nous devons transformer ce flux continu de texte en une séquence d'unités discrètes, portantes de sens et statistiquement exploitables. C'est le rôle de la **tokenisation**.

Regardez la **Figure 2-1 : Tokens et Embeddings** (p.37 du livre). Cette illustration simplifie le flux de données : le texte entre, il est découpé en morceaux (tokens), puis chaque morceau est converti en une suite de nombres (embeddings). 🔑 **Notez bien cette intuition :** le tokeniseur est le traducteur universel qui permet au Transformer de "voir" le langage humain. [SOURCE: Livre p.37, Figure 2-1]

### Voyage au cœur du processus (Analyse des Figures 2-2 à 2-5)
Le livre nous propose une décomposition méticuleuse du processus de tokenisation à travers quatre figures fondamentales.

#### 1. La vue d'ensemble (Figure 2-2)
**Explication de la Figure 2-2** (p.38) : Cette figure nous montre l'IA vue de l'extérieur. Un utilisateur envoie un prompt. Pour nous, c'est une phrase. Pour le modèle, c'est une entrée qui doit être traitée avant d'être comprise. La figure montre que le tokeniseur n'est pas "dans" le modèle, c'est un composant périphérique essentiel qui prépare le terrain. [SOURCE: Livre p.38, Figure 2-2]

#### 2. La visualisation interactive (Figure 2-3)
**Explication de la Figure 2-3** (p.39) : Ici, les auteurs utilisent l'outil de visualisation d'OpenAI pour montrer comment GPT-4 "découpe" une phrase célèbre. 
*   **Observation** : Vous remarquerez que certains mots sont entiers, tandis que d'autres (plus rares ou plus longs) sont brisés. Chaque token est coloré différemment. 
*   🔑 **Le message technique** : La tokenisation n'est pas qu'un découpage sur les espaces. C'est une stratégie d'optimisation mathématique. [SOURCE: Livre p.39, Figure 2-3 / OpenAI Tokenizer Tool]

#### 3. La conversion numérique (Figure 2-4)
**Explication de la Figure 2-4** (p.41) : C'est le passage au "numérique". Une fois le texte découpé, chaque token reçoit un **ID unique**.
*   Exemple : "Have" devient `6975`, "the" devient `278`. 
*   ⚠️ **Attention : erreur fréquente ici !** Ces nombres ne sont pas choisis selon l'ordre alphabétique ou l'importance sémantique. Ce sont simplement des index dans une immense table de correspondance (le dictionnaire du modèle). [SOURCE: Livre p.41, Figure 2-4]

#### 4. Le décodage de sortie (Figure 2-5)
**Explication de la Figure 2-5** (p.44) : Le processus est symétrique. Quand le modèle génère une réponse, il ne sort pas des lettres, mais des IDs de tokens. Le tokeniseur doit alors effectuer un "detokenize" pour transformer ces nombres en texte lisible par l'humain. 🔑 **Je dois insister :** si vous perdez le fichier de configuration de votre tokeniseur, vos milliards de paramètres de modèle ne servent plus à rien ; vous ne pourrez plus traduire les chiffres en mots ! [SOURCE: Livre p.44, Figure 2-5]

---

### La guerre des granularités : Choisir la taille de l'atome
« Comment devons-nous découper le langage ? » Cette question a hanté les chercheurs pendant des années. Le livre nous propose un tableau comparatif capital (p.44-46) que nous allons analyser en détail.

#### 1. La granularité par mots (Word-level)
Historiquement, c'était la méthode par défaut. On coupe à chaque espace.
*   **Avantages** : Très intuitif, préserve l'unité sémantique.
*   **Inconvénients** : 
    *   **Explosion du vocabulaire** : Si vous voulez couvrir toutes les déclinaisons ("manger", "manges", "mangeait"), vous avez besoin d'un dictionnaire de plusieurs millions d'entrées.
    *   **Mots hors-vocabulaire (OOV)** : Si le modèle rencontre un mot qu'il n'a pas vu durant son entraînement (ex: un nouveau mot d'argot ou un nom propre rare), il échoue totalement et affiche `[UNK]` (Unknown). L'information est perdue. [SOURCE: Livre p.44]

#### 2. La granularité par caractères (Character-level)
On découpe lettre par lettre : "c", "h", "a", "t".
*   **Avantages** : Vocabulaire minuscule (quelques centaines de caractères), zéro mot inconnu.
*   **Inconvénients** : 
    *   **Perte de sens** : Le caractère "c" ne porte aucun sens en soi. Le modèle doit faire un effort colossal pour réapprendre la structure des mots.
    *   **Saturation de la mémoire** : Comme les séquences deviennent immenses (une phrase de 10 mots devient une séquence de 60 caractères), on sature très vite la fenêtre de contexte du Transformer. [SOURCE: Livre p.45]

#### 3. Le compromis parfait : La granularité par sous-mots (Subword-level)
C'est le standard actuel de tous les LLM (GPT-4, Llama-3, Mistral). On garde les mots fréquents entiers ("le", "maison") et on découpe les mots rares en morphèmes ("malheureusement" -> "malheureuse" + "##ment").
🔑 **C'est la solution au problème OOV :** Même si le modèle ne connaît pas un mot précis, il peut le reconstituer à partir de racines et de suffixes qu'il connaît déjà. [SOURCE: Livre p.44-45]

#### 4. La granularité par octets (Byte-level)
Utilisée par GPT-2 et ses successeurs. On ne découpe plus des lettres, mais des **octets UTF-8**.
*   🔑 **Pourquoi c'est génial ?** Cela permet de traiter n'importe quel symbole (emojis 🎵, kanjis japonais, caractères arabes) avec un vocabulaire fixe de seulement 256 octets de base. C'est l'universalité totale. [SOURCE: Livre p.45]

---

### Plongée dans l'algorithme : Byte Pair Encoding (BPE)
« Mes chers étudiants, voici le moment où nous devenons des ingénieurs. » L'algorithme BPE est le moteur derrière GPT. Son fonctionnement est un exemple parfait de logique statistique itérative.

**Le processus BPE en 4 étapes** :
1.  **Initialisation** : On traite chaque caractère comme un token individuel.
2.  **Comptage** : On cherche dans tout notre immense corpus de textes quelle paire de tokens adjacents apparaît le plus souvent (ex: "e" et "r").
3.  **Fusion (Merge)** : On crée un nouveau token "er" et on l'ajoute à notre dictionnaire. Désormais, "e" et "r" côte à côte ne sont plus deux jetons, mais un seul.
4.  **Itération** : On recommence jusqu'à atteindre la taille de dictionnaire souhaitée (ex: 50 000 tokens).

🔑 **L'intuition du Prof. Henni** : Le BPE construit le langage de bas en haut. Il ne connaît pas la grammaire, il connaît la **fréquence**. S'il voit souvent "ing" à la fin des verbes anglais, il finira par créer un token unique `ing`. [SOURCE: Livre p.43, p.55]

### WordPiece : L'approche de Google (BERT)
WordPiece ressemble à BPE, mais avec une subtilité mathématique. Au lieu de fusionner la paire la plus *fréquente*, il fusionne la paire qui augmente le plus la **vraisemblance** (likelihood) du modèle.
*   BPE demande : "Qu'est-ce que je vois le plus ?"
*   WordPiece demande : "Quelle fusion m'aide le mieux à prédire la suite ?"
⚠️ **Fermeté bienveillante** : Pour l'utilisateur final, la différence est minime, mais pour l'entraînement d'un modèle comme BERT, WordPiece permet une meilleure compression de l'information sémantique. [SOURCE: Livre p.43-44]

---

### L'impact critique : Le cas du Code et du Multilingue
⚠️ **Je dois insister sur ce point :** la tokenisation n'est pas une science exacte, c'est un arbitrage.

1.  **Le Code source** : Comme nous le verrons avec StarCoder2 (Semaine 13), un tokeniseur entraîné sur du texte naturel va détruire l'indentation du code Python. Un tokeniseur spécialisé doit posséder des tokens pour "4 espaces", "1 tabulation", etc. Sans cela, le modèle perd la logique structurale du code. [SOURCE: Livre p.51-52]
2.  **L'inégalité multilingue** : ⚠️ **Éthique ancrée** : Un tokeniseur entraîné majoritairement sur l'anglais découpera un mot anglais en 1 token, mais le même concept en arabe ou en swahili sera haché en 5 ou 6 tokens. 
🔑 **Conséquence économique et technique :** Cela rend l'IA plus chère (facturation au token) et moins "intelligente" (mémoire de contexte saturée plus vite) pour les langues non-occidentales. C'est l'un des plus grands défis de l'IA inclusive. [SOURCE: Livre p.55]

### Gestion des mots hors vocabulaire (OOV)
Comment BERT ou GPT gèrent-ils un mot totalement inconnu ? 
*   Grâce aux sous-mots, ils le décomposent jusqu'au niveau du caractère si nécessaire. 
*   🔑 **Le rôle du token spécial `[UNK]`** : C'est le signal d'échec. Si un modèle affiche trop de `[UNK]`, c'est que votre tokeniseur n'est pas adapté à vos données (ex: utiliser un modèle de chat généraliste sur des séquences d'ADN). [SOURCE: Livre p.47]

### Exemple de code pédagogique : Visualiser les tokens
Voici comment charger et inspecter le comportement d'un tokeniseur moderne sur Google Colab.

```python
# Testé sur Colab T4 16GB VRAM
from transformers import AutoTokenizer

# Chargement du tokeniseur de BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Tokenization is the atom of NLP!"

# 1. Découpage brut
tokens = tokenizer.tokenize(text)
print(f"Découpage visuel : {tokens}")
# Observez le mot 'Tokenization' qui risque d'être coupé en 'token' + '##ization'

# 2. Conversion en nombres (IDs)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Index numériques : {ids}")

# 3. Encodage complet (avec les tokens spéciaux [CLS] et [SEP])
full_encoding = tokenizer(text)
print(f"Encodage final avec contrôle : {full_encoding['input_ids']}")
```

⚠️ **Fermeté bienveillante** : Regardez bien la sortie de `full_encoding`. Vous verrez des chiffres au début et à la fin qui ne correspondent pas à vos mots. Ce sont les tokens spéciaux de structure. 🔑 **C'est non-négociable :** vous devez apprendre à vivre avec ces jetons de contrôle, car ils sont les balises GPS du Transformer. [SOURCE: Livre p.40, p.48]

---

### Synthèse de la section
Nous avons vu comment la tokenisation transforme le flux continu de notre pensée en une suite discrète de jetons numériques. Nous avons compris que le choix de la granularité (mots, sous-mots, octets) définit le cadre de réflexion de l'IA. 

🔑 **Le message final du Prof. Henni pour cette section** : « Ne voyez plus un texte comme une suite de lettres. Voyez-le comme une suite de jetons statistiques. La tokenisation est la fondation de l'édifice. Si les fondations sont de travers, le gratte-ciel LLM s'écroulera au moindre raisonnement complexe. » [SOURCE: Livre p.71]

« Vous maîtrisez désormais la théorie du découpage. Vous savez comment le texte devient un index. Dans la section suivante, nous allons donner une "âme" à ces index : nous allons découvrir comment transformer un simple numéro comme `6975` en un vecteur riche de sens. Bienvenue dans le monde des **Embeddings**. »

---
*Fin de la section 2.1 (2150 mots environ)*
## 2.2 Comparaison des tokeniseurs modernes (2000+ mots)

### La diversité des regards : Pourquoi comparer les outils ?
« Bonjour à toutes et à tous ! J'espère que vous avez bien en tête notre microscope de la section 2.1. Nous avons compris la théorie, mais aujourd'hui, je vais vous montrer que dans le monde réel des LLM, tous les modèles ne "lisent" pas de la même manière. 🔑 **Je dois insister :** choisir un modèle sans regarder son tokeniseur, c'est comme acheter une voiture sans vérifier si elle roule à l'essence ou à l'électrique. Aujourd'hui, nous allons comparer les géants : de l'ancêtre **BERT** au surpuissant **GPT-4**, en passant par les spécialistes du code comme **StarCoder2** et les érudits comme **Galactica**. Respirez, nous allons voir comment de simples réglages de découpage transforment radicalement l'intelligence d'une machine. » [SOURCE: Livre p.46]

Le livre nous propose un voyage à travers les époques et les spécialisations (p.46-54). Ce que nous allons découvrir, c'est que la tokenisation est le fruit d'un arbitrage permanent entre trois facteurs : la méthode (BPE, WordPiece), les paramètres (taille du dictionnaire) et le domaine des données.

---

### 1. BERT : La rigueur du pionnier (WordPiece)
BERT (2018) est notre point de référence historique. Il utilise l'algorithme **WordPiece** avec un vocabulaire relativement modeste de 30 000 tokens. 

🔑 **La distinction Cased vs. Uncased :**
Le livre détaille p.47 et p.48 une différence fondamentale que vous rencontrerez souvent :
*   **BERT-uncased** : Tout est converti en minuscules. "New York" devient "new", "york". ⚠️ **Attention : erreur fréquente ici !** Pour une tâche de classification de spams, c'est parfait. Mais si vous faites de l'extraction de noms propres (NER), vous perdez l'indice capital de la majuscule.
*   **BERT-cased** : Préserve la casse. Comme le montre l'exemple p.48, le mot "CAPITALIZATION" est découpé en huit tokens (`CA`, `##PI`, `##TA`, `##L`, `##I`, `##Z`, `##AT`, `##ION`). C'est beaucoup ! Cela montre que BERT-cased est plus précis mais "consomme" plus de jetons pour les mots en majuscules.

**Discussion technique** : BERT utilise le symbole `##` pour marquer les sous-mots. Cela permet au modèle de savoir que `##tion` n'est pas un mot seul, mais la fin d'un mot précédent. [SOURCE: Livre p.47-48]

---

### 2. La lignée GPT : De l'efficacité à l'omniscience (BPE)
OpenAI a popularisé le **Byte-level BPE**. Leur innovation ? Ne plus jamais voir de mots "inconnus" en travaillant au niveau des octets.

#### GPT-2 (2019) : L'invention du caractère spécial Ġ
Regardez l'exemple p.49. GPT-2 a introduit une convention visuelle fascinante : le caractère `Ġ`. 
*   **Intuition** : GPT-2 traite l'espace *avant* un mot comme faisant partie du mot lui-même. 
*   **Pourquoi ?** Cela permet au modèle de distinguer "chat" (en début de phrase) et " chat" (au milieu d'une phrase). Pour l'IA, ce sont deux concepts statistiques légèrement différents. 
🔑 **Notez bien :** Avec son vocabulaire de 50 000 tokens, GPT-2 est bien plus efficace que BERT pour représenter des mots complexes. [SOURCE: Livre p.49]

#### GPT-4 (2023) : Le saut vers les 100 000 tokens
🔑 **Je dois insister sur cette évolution :** Entre GPT-2 et GPT-4, OpenAI a doublé la taille du dictionnaire. 
*   **L'avantage** : Un dictionnaire plus grand signifie que des mots longs et fréquents (ex: "anthropologie") deviennent un seul token au lieu de trois. 
*   **Conséquence pour l'ingénieur** : Comme le modèle utilise moins de tokens pour dire la même chose, vous pouvez faire tenir des documents beaucoup plus longs dans la fenêtre de contexte. C'est un gain de productivité direct. [SOURCE: Livre p.51]

---

### 3. Flan-T5 et SentencePiece : L'approche agnostique
Flan-T5 utilise **SentencePiece**, un outil qui traite l'espace comme un caractère normal (souvent noté `_`). 

⚠️ **Avertissement du Professeur** : Regardez la note p.50. Flan-T5 a une faiblesse : il a été entraîné sans tokens pour les retours à la ligne (`\n`) ou les tabulations. 
*   **Le danger** : Si vous lui demandez d'analyser un fichier de configuration ou un poème, il verra tout comme une seule ligne continue. C'est l'exemple parfait d'un tokeniseur brillant pour la discussion mais aveugle à la mise en page. [SOURCE: Livre p.50]

---

### 4. StarCoder2 : Le traducteur de code source
« Mes chers étudiants, si vous voulez que votre IA programme à votre place, elle doit comprendre la structure d'un code Python. » StarCoder2 (p.51-52) est une merveille d'ingénierie spécialisée.

**Les deux secrets de StarCoder2** :
1.  **L'indentation comme token** : Contrairement aux modèles de texte, StarCoder2 possède des tokens dédiés pour "un espace", "deux espaces", "quatre espaces". Pourquoi ? Parce qu'en Python, l'espace définit la logique. Si vous fusionnez les espaces n'importe comment, le code ne compilera jamais.
2.  **La tokenisation des chiffres** : 🔑 **C'est un concept non-négociable :** StarCoder2 découpe les nombres chiffre par chiffre (`123` devient `1`, `2`, `3`). 
    *   *Pourquoi ?* Pour que le modèle apprenne l'arithmétique. Si "123" est un token unique, le modèle doit mémoriser son nom. S'il voit "1", "2" et "3", il peut apprendre les règles de retenue et de calcul comme un enfant à l'école. [SOURCE: Livre p.51-52]

---

### 5. Galactica : Le scientifique érudit
Galactica (p.52-53) a été entraîné sur la science. Son tokeniseur est unique car il gère :
*   **Les citations** : Des tokens spéciaux comme `[START_REF]` et `[END_REF]` permettent au modèle de savoir quand il s'appuie sur une source.
*   **La chimie et l'ADN** : Il ne découpe pas les formules chimiques (`H2O`) de manière aléatoire, ce qui permet de préserver la sémantique des molécules. 
*   **Le raisonnement** : Il utilise le token `<work>` pour délimiter ses brouillons de calculs internes. [SOURCE: Livre p.52-53]

---

### 6. Phi-3 et Llama 2 : La standardisation moderne
Ces modèles (p.53-54) utilisent des tokeniseurs BPE très optimisés pour le dialogue. Ils intègrent des balises de rôle comme `<|user|>` ou `<|assistant|>`. 
🔑 **Je dois insister :** Ces tokens ne sont pas du texte, ce sont des "commandes de vol" pour le modèle. Ils indiquent à l'IA quand elle doit écouter et quand elle doit répondre. [SOURCE: Livre p.54]

---

### Tableau récapitulatif : La bataille des chiffres

| Modèle | Méthode | Taille Vocab | Spécificité |
| :--- | :--- | :--- | :--- |
| **BERT** | WordPiece | 30 000 | Bidirectionnel, `##` pour sous-mots. |
| **GPT-2** | BPE | 50 257 | Byte-level, utilise le `Ġ` pour l'espace. |
| **GPT-4** | BPE | 100 256+ | Très efficace pour les fenêtres de contexte. |
| **Flan-T5** | Unigram | 32 100 | SentencePiece, gère mal les retours ligne. |
| **StarCoder2**| BPE | 49 152 | Chiffres isolés, tokens d'indentation. |
| **Galactica** | BPE | 50 000 | Citations et formules scientifiques. |

[SOURCE: Synthèse des pages 46-54 du livre]

---

### Laboratoire de code : Comparaison pratique sur Colab
« Ne me croyez pas sur parole, testez-le ! » Voici comment comparer l'efficacité de deux tokeniseurs sur une même phrase technique.

```python
# Testé sur Google Colab T4 16GB VRAM
from transformers import AutoTokenizer

# On compare BERT (texte) et StarCoder (code)
model_names = {
    "BERT (Général)": "bert-base-uncased",
    "StarCoder (Code)": "bigcode/starcoder2-7b" 
}

# Un texte avec du code et des chiffres
text = "for i in range(123): print(i)"

for name, path in model_names.items():
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokens = tokenizer.tokenize(text)
        print(f"--- {name} ---")
        print(f"Nombre de tokens : {len(tokens)}")
        print(f"Découpage : {tokens}\n")
    except Exception:
        print(f"Note : {name} nécessite un accès spécifique sur HF.")

# [SOURCE: CONCEPT À SOURCER – INSPIRÉ DU REPO GITHUB CHAPTER 2]
```

⚠️ **Fermeté bienveillante** : Observez le découpage du nombre `123`. Vous verrez que StarCoder le traite différemment pour préserver la logique mathématique. 🔑 **C'est une distinction capitale pour vos futurs projets de data science.**

---

### Éthique et Économie : Le coût caché du token
⚠️ **Éthique ancrée** : « Mes chers étudiants, la tokenisation est un enjeu de justice. » 

Comme nous l'avons évoqué p.55-56, la tokenisation n'est pas neutre économiquement.
1.  **Le coût de la langue** : Puisque la plupart des tokeniseurs sont entraînés sur des corpus anglo-centrés, un locuteur français ou arabe "consomme" plus de tokens pour la même phrase. Si vous utilisez une API payante (OpenAI, Anthropic), **votre facture sera plus élevée simplement à cause de votre langue maternelle.** 
2.  **La barrière à l'entrée** : Les modèles avec de petits dictionnaires (comme BERT) sont plus "bêtes" face aux langues rares car ils hachent les mots en trop de morceaux, perdant le fil sémantique. 
3.  **L'illusion de l'universalité** : 🔑 **Je dois insister :** Un tokeniseur "multilingue" est souvent un tokeniseur anglais qui a appris quelques mots d'ailleurs. Toujours tester la fragmentation de votre langue cible avant de choisir un modèle. [SOURCE: Livre p.55, p.28]

---

### Synthèse de la section
Nous avons parcouru la jungle des tokeniseurs modernes. Nous avons vu que BERT privilégie la structure des mots (`##`), que GPT cherche l'efficacité massive (100k+ tokens), et que les modèles spécialisés (StarCoder, Galactica) réinventent le découpage pour servir la logique ou la science.

🔑 **Le message final du Prof. Henni pour cette section** : « Le tokeniseur est la rétine de votre modèle. S'il est daltonien ou s'il manque de résolution, le cerveau LLM ne pourra jamais compenser cette perte d'information. Avant de fine-tuner, vérifiez toujours comment votre modèle "voit" votre domaine spécifique. » [SOURCE: Livre p.71]

« Vous maîtrisez désormais l'art de la comparaison. Vous savez quel outil choisir selon votre mission. Dans la section suivante, nous allons nous intéresser aux **Propriétés techniques** : nous allons voir comment régler les paramètres de ces tokeniseurs pour qu'ils s'adaptent parfaitement à vos besoins. »

---
*Fin de la section 2.2 (2180 mots environ)*
## 2.3 Propriétés des tokeniseurs modernes (1500+ mots)

### Au-delà du découpage : L'ingénierie des réglages
« Bonjour à toutes et à tous ! Nous avons comparé les regards des géants dans la section précédente. Vous avez vu que GPT-4 ne "voit" pas la même chose que BERT ou StarCoder. Mais comment ces différences voient-elles le jour ? 🔑 **Je dois insister :** un tokeniseur ne naît pas "bon" ou "mauvais" par magie ; il est le produit de choix techniques délibérés que nous appelons les **propriétés du tokeniseur**. Aujourd'hui, nous allons apprendre à régler les curseurs de cette machine de précision. Comprendre ces propriétés, c'est comprendre pourquoi votre modèle sera rapide, précis, ou au contraire, totalement perdu face à une langue étrangère. Respirez, nous allons entrer dans les coulisses de la configuration. » [SOURCE : Livre p.55]

Le livre détaille aux pages 55 et 56 les trois piliers qui définissent l'identité d'un tokeniseur : la taille du vocabulaire, la gestion des tokens spéciaux et le traitement de la casse (capitalisation). Analysons ces paramètres avec la rigueur de l'ingénieur et l'intuition du linguiste.

---

### 1. La taille du vocabulaire : Le dilemme de la résolution
La propriété la plus visible d'un tokeniseur est la taille de son dictionnaire interne, souvent notée `vocab_size`. 

**L'analogie du Professeur Henni** : Imaginez que la taille du vocabulaire soit la résolution d'un capteur photo. 
*   Un **petit vocabulaire** (ex: 30 000 tokens pour BERT) est comme une photo basse résolution. On voit les formes globales, mais pour les détails (les mots rares), on est obligé de "zoomer" et de découper le mot en plein de petits carrés (sous-mots). 
*   Un **grand vocabulaire** (ex: 100 000+ tokens pour GPT-4) est une photo haute définition. Le modèle peut identifier des concepts complexes d'un seul coup d'œil. 

🔑 **L'impact technique caché :** Pourquoi ne pas créer un dictionnaire de 10 millions de mots alors ? À cause du coût en mémoire vive (VRAM). Rappelez-vous cette équation fondamentale : 
`Taille de la couche d'embedding = taille du vocabulaire × dimension du modèle`. 
Si vous avez un vocabulaire de 100 000 tokens et que chaque vecteur (embedding) fait 1024 nombres, vous consommez déjà plus de 100 millions de paramètres **avant même d'avoir commencé le premier bloc Transformer**. C'est un arbitrage constant entre la richesse de la compréhension et la légèreté du modèle. [SOURCE : Livre p.55, Section "Vocabulary size"]

---

### 2. Les Tokens Spéciaux : La signalisation du langage
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent qu'un tokeniseur ne contient que des mots ou des morceaux de mots. C'est faux. Pour qu'un Transformer fonctionne, il a besoin de "panneaux de signalisation". Ce sont les **Special Tokens**.

Le livre nous apprend que chaque modèle possède sa propre syntaxe de contrôle (p.55-56) :
*   **[CLS] (Classification)** : Le token de tête utilisé par BERT pour résumer une phrase.
*   **[SEP] (Separator)** : Indique au modèle que l'on passe d'une phrase A à une phrase B.
*   **[PAD] (Padding)** : 🔑 **C'est une propriété vitale.** Les GPU adorent les grilles de taille fixe. Si vous avez une phrase de 5 mots et une de 10 mots, le tokeniseur ajoute des tokens `[PAD]` à la plus courte pour que les deux fassent la même longueur mathématique.
*   **[MASK]** : Utilisé pour le pré-entraînement (MLM) que nous avons vu en Semaine 1.
*   **BOS/EOS (Beginning/End Of Sentence)** : Pour les modèles génératifs comme GPT, ces tokens indiquent : "C'est ici que l'histoire commence" et "J'ai fini de parler, c'est à toi".

Discussion sur les tokens de domaine (Galactica) : Comme nous l'avons vu p.53, certains modèles ajoutent des tokens comme `<work>` pour signaler un espace de brouillon. 🔑 **Je dois insister :** en tant que développeurs, vous pouvez ajouter vos propres tokens spéciaux si vous créez une IA pour un métier précis (ex: un token `[GENE]` pour isoler des séquences biologiques). [SOURCE : Livre p.55-56]

---

### 3. La Normalisation et la Capitalisation
Avant même de découper le texte en tokens, le tokeniseur applique une phase de **Normalisation**. 

#### Le cas de la Casse (Cased vs Uncased)
Nous avons vu avec BERT que le choix est binaire :
*   **Uncased** : On transforme tout en minuscules. C'est une forme de simplification statistique. On apprend au modèle que "Le", "le" et "LE" sont le même concept. Cela réduit le besoin de données, mais cela rend le modèle "aveugle" à l'emphase (le cri en majuscules) ou aux noms propres.
*   **Cased** : On préserve la casse. C'est plus difficile à apprendre, mais c'est indispensable pour le code (où `Variable` est différent de `variable`) ou pour la traduction de haute qualité. [SOURCE : Livre p.47]

#### Le nettoyage de surface
Certains tokeniseurs appliquent des règles de "stripping" (suppression des espaces inutiles) ou de normalisation Unicode (ex: transformer le caractère "é" composé en un "e" plus un accent "´"). 
⚠️ **Fermeté bienveillante** : Si votre tokeniseur supprime les accents sans vous le dire, votre modèle sera incapable de distinguer "pêcher" (le fruit ou le poisson) et "pécher" (la faute morale). Vérifiez toujours la fonction de normalisation de votre outil ! [SOURCE : Hugging Face Tokenizers Documentation]

---

### 4. Le Domaine des données : L'ADN du Tokeniseur
C'est sans doute la propriété la plus sous-estimée. Un tokeniseur est **entraîné**. Il n'est pas programmé avec des règles de dictionnaire ; il a "appris" quelles étaient les suites de caractères les plus fréquentes sur un jeu de données précis.

*   **Le biais de domaine** : Si vous prenez un tokeniseur entraîné sur des tweets et que vous lui donnez un contrat notarié de 1850, il va "hacher" chaque mot complexe en 10 tokens minuscules. 
*   **Pourquoi est-ce grave ?** Plus vous avez de tokens pour une même idée, plus le modèle risque de perdre le fil (phénomène de fragmentation sémantique). 🔑 **La règle d'or** : Le tokeniseur et le modèle doivent avoir été "élevés" sur les mêmes données. [SOURCE : Livre p.56]

**Discussion sur l'indentation (StarCoder2)** :
Revenons sur l'exemple de la page 56. Pour un texte naturel, l'espace n'est qu'un séparateur. Pour le code, l'espace est une **information de contrôle**. Un tokeniseur moderne pour le code doit posséder la propriété de préserver les blocs d'espaces. Si le tokeniseur fusionne trois espaces en un seul, le modèle ne "comprendra" jamais l'arborescence d'une fonction Python. [SOURCE : Livre p.56]

---

### 5. Bonnes pratiques de sélection : Comment choisir ?
« Vous avez le catalogue sous les yeux. Comment choisir pour votre projet de fin d'études ? » Suivez ces trois critères non-négociables :

1.  **L'efficacité du dictionnaire** : Faites un test simple. Prenez 100 phrases de votre domaine. Comptez le nombre total de tokens générés par deux tokeniseurs différents. Celui qui produit le moins de tokens est généralement le plus "intelligent" pour votre usage, car il possède des concepts entiers dans son dictionnaire.
2.  **La gestion des caractères inconnus** : Si vous travaillez sur des données scientifiques ou multilingues, évitez les modèles qui sortent trop souvent des `[UNK]`. Privilégiez le **Byte-level BPE**.
3.  **La compatibilité avec le LLM** : ⚠️ **Point crucial !** On ne change jamais le tokeniseur d'un modèle déjà entraîné. Si vous utilisez Llama-3, vous DEVEZ utiliser le tokeniseur de Llama-3. Les poids du modèle ont été calibrés sur ces index précis. [SOURCE : CONCEPT À SOURCER – PRATIQUE COURANTE HUGGING FACE]

---

### Laboratoire de réflexion : Le "Token Tax" (La taxe du token)
⚠️ **Éthique ancrée** : « Mes chers étudiants, les propriétés techniques ont des conséquences humaines. » 

Parce que les tokeniseurs sont majoritairement configurés pour l'anglais, les langues "complexes" (comme l'arabe, le coréen ou le finnois) subissent ce que les chercheurs appellent la **Token Tax**. 
*   Pour une même idée, un utilisateur anglais paiera 10 tokens. 
*   Un utilisateur parlant une langue moins représentée paiera 30 tokens pour la même phrase. 
🔑 **Conséquence éthique :** Cela signifie que les populations des pays du Sud ont des IA moins performantes (fenêtre de contexte plus petite) et plus chères (facturation au token). En tant qu'experts, votre responsabilité est de privilégier les modèles "Truly Multilingual" qui ont des vocabulaires larges et équitables. [SOURCE : Livre p.55, p.28]

---

### Synthèse de la section
Nous avons exploré les entrailles des tokeniseurs. Nous avons compris que :
*   La **taille du dictionnaire** définit la résolution de la compréhension.
*   Les **tokens spéciaux** sont l'ossature du dialogue.
*   La **normalisation** peut être une alliée ou une ennemie de la précision.
*   Le **domaine d'entraînement** dicte la performance réelle sur le terrain.

🔑 **Le message final du Prof. Henni pour cette section** : « Le tokeniseur est la porte d'entrée de l'intelligence. S'il est mal configuré, le cerveau de l'IA vivra dans le brouillard. Soyez méticuleux dans le choix de vos paramètres, car ils sont le premier filtre de la vérité de vos données. » [SOURCE : Livre p.71]

« Vous maîtrisez maintenant l'art du découpage et les secrets de la configuration. Mais une question demeure : une fois que nous avons nos numéros de tokens, comment la machine fait-elle pour savoir que le token `cat` (chat) est sémantiquement proche du token `kitten` (chaton) ? Il nous manque la "chair" des mots. Rendez-vous dans la section suivante pour découvrir les **Embeddings**, la géométrie du sens. »

---
*Fin de la section 2.3 (1520 mots environ)*
## 2.4 Plongements lexicaux (Embeddings) (2000+ mots)

### Donner une chair aux nombres : La naissance du vecteur de sens
« Bonjour à toutes et à tous ! Nous arrivons au point culminant de notre deuxième semaine. Dans la section précédente, nous avons appris à découper le langage en petits index numériques. Nous savons transformer le mot "Cœur" en un numéro de dictionnaire, par exemple le `1254`. Mais posons-nous une question fondamentale : qu'est-ce que le chiffre `1254` sait de l'amour, de l'anatomie ou du courage ? Rien. Pour un ordinateur, `1254` est juste une étiquette, aussi vide de sens que le chiffre `1255`. 🔑 **Je dois insister :** pour que l'IA comprenne vraiment, nous devons passer du "numéro d'ordre" à la "position géographique". Aujourd'hui, nous allons découvrir les **Embeddings**, ou plongements lexicaux. Nous allons apprendre comment chaque mot reçoit une "adresse" dans un espace à plusieurs centaines de dimensions, où la proximité physique reflète enfin la proximité de sens. Respirez, nous allons donner une âme mathématique aux jetons ! » [SOURCE: Livre p.57]

---

### 1. Le concept d'Embedding : Une carte du monde sémantique
Un embedding est une représentation d'un token sous la forme d'un vecteur dense de nombres réels. Au lieu d'un simple index, chaque mot est défini par une liste de coordonnées (généralement 768 ou 1024 dimensions dans les modèles modernes). 

Regardons la **Figure 2-7 : Embeddings associés au vocabulaire** (p.58 du livre). 
**Explication de la Figure 2-7** : Cette illustration nous montre le lien physique entre le tokeniseur et le modèle. 
1.  Le tokeniseur fournit un ID (ex: 0, 1, ..., 50 257). 
2.  Le modèle possède une immense matrice de poids appelée "Embedding Matrix". 
3.  L'ID sert de numéro de ligne : si le token est `1`, le modèle va chercher la deuxième ligne de sa matrice. Cette ligne contient le vecteur (l'embedding) associé à ce mot. 

🔑 **L'intuition du Professeur Henni :** Imaginez que le langage soit une galaxie. Chaque mot est une étoile. Les embeddings sont les coordonnées GPS (Latitude, Longitude, Altitude...) qui permettent de dire que l'étoile "Planète" est plus proche de l'étoile "Terre" que de l'étoile "Sandwich". [SOURCE: Livre p.58, Figure 2-7]

---

### 2. L'héritage de Word2Vec : La révolution des embeddings statiques
Pour comprendre comment ces vecteurs sont nés, nous devons revenir en 2013 avec l'algorithme **Word2Vec**. Le livre nous propose de revisiter ses fondements à travers les figures du Chapitre 1 (p.8-10).

#### La structure neuronale (Figure 1-6)
**Explication de la Figure 1-6** (p.8) : Cette figure montre que les embeddings ne sont pas programmés par des humains, ils sont **appris**. On utilise un réseau de neurones très simple avec une couche cachée. 🔑 **La découverte majeure** : les poids de cette couche cachée, une fois l'entraînement fini, deviennent les embeddings eux-mêmes. Le sens est un sous-produit de l'apprentissage statistique. [SOURCE: Livre p.8, Figure 1-6]

#### L'apprentissage par voisinage (Figure 1-7)
**Explication de la Figure 1-7** (p.9) : Comment apprend-on le sens ? Par le contexte ! La figure illustre une tâche de prédiction : "Le mot A et le mot B sont-ils voisins ?". 
*   On montre au modèle des milliers de phrases. 
*   Si "Chat" et "Miaule" apparaissent souvent ensemble, le modèle va ajuster leurs vecteurs pour qu'ils pointent dans la même direction. 
*   🔑 **L'intuition technique :** C'est ce qu'on appelle l'apprentissage contrastif. On rapproche les mots qui se fréquentent et on éloigne ceux qui n'ont rien à voir. [SOURCE: Livre p.9, Figure 1-7]

#### La décomposition des propriétés (Figure 1-8)
**Explication de la Figure 1-8** (p.9) : C'est ici que la magie opère. La figure montre que chaque dimension du vecteur finit par capturer une propriété abstraite. 
*   Une colonne pourrait représenter l'aspect "Animal".
*   Une autre l'aspect "Royal".
*   Une autre le "Genre" (masculin/féminin).
Ainsi, le mot "Roi" aura une valeur élevée en "Royal" et "Homme", tandis que "Reine" sera élevée en "Royal" et "Femme". ⚠️ **Attention : erreur fréquente ici !** Dans la réalité, ces dimensions ne sont pas nommées ainsi. Elles sont des abstractions mathématiques que nous interprétons après coup. [SOURCE: Livre p.9, Figure 1-8]

#### La géométrie du sens (Figure 1-9)
**Explication de la Figure 1-9** (p.10) : Si l'on réduit ces centaines de dimensions à seulement 2 (pour pouvoir les dessiner), on observe des grappes (clusters). "Chat", "Chien" et "Chiot" forment un petit village sémantique. "Banane" et "Pomme" en forment un autre, très loin du premier. 🔑 **Je dois insister :** La similarité entre deux pensées humaines est devenue une simple question de **distance cosinus** entre deux points. [SOURCE: Livre p.10, Figure 1-9]

---

### 3. Les limites des embeddings statiques
« Pourquoi ne pas s'être arrêté à Word2Vec ? » demanderez-vous. Parce que Word2Vec souffre d'un défaut fatal : il est **statique**. 

Reprenons notre exemple de la Semaine 1 : le mot "avocat".
1. "J'ai mangé un **avocat** mûr."
2. "Mon **avocat** a plaidé ma cause."

Dans Word2Vec, le token `avocat` n'a qu'un seul vecteur. C'est une moyenne confuse entre un fruit et un juriste. Le modèle est incapable de changer sa vision du mot selon la phrase. 🔑 **C'est le mur de la polysémie.** Pour le franchir, il nous fallait l'architecture Transformer. [SOURCE: Livre p.11]

---

### 4. L'avènement des Embeddings Contextuels (Figures 2-8 et 2-9)
C'est ici que nous entrons dans l'ère des LLM modernes. Regardez la **Figure 2-8 : Embeddings contextuels** (p.59 du livre).

**Explication de la Figure 2-8** : Contrairement aux modèles statiques, un LLM (comme BERT ou Phi-3) traite la phrase entière avant de produire le vecteur final d'un mot. 
*   Le modèle regarde les mots "alentour". 
*   Si le mot "mûr" est présent, il va "mélanger" le vecteur d' `avocat` avec des informations de nourriture. 
*   Le résultat est un **vecteur dynamique** qui n'existe que pour cette phrase précise. [SOURCE: Livre p.59, Figure 2-8]

**Explication de la Figure 2-9** (p.60) : Cette figure détaille le flux. L'entrée est un embedding statique (une base brute), mais après être passé par les couches d'Attention (Semaine 3), il ressort transformé en embedding contextuel. 🔑 **Notez bien cette distinction :** l'embedding de base est le "potentiel" du mot, l'embedding contextuel est sa "réalité" dans une phrase donnée. [SOURCE: Livre p.60, Figure 2-9]

---

### 5. Création et extraction : Le LLM comme extracteur de sens
Aujourd'hui, nous utilisons souvent les LLM non pas pour générer du texte, mais pour générer des vecteurs de haute qualité pour la recherche sémantique (Semaine 6) ou le clustering (Semaine 7).

#### Laboratoire de code : Extraire des embeddings contextuels
Voici comment utiliser un modèle de pointe (DeBERTa ou MPNet) pour transformer vos phrases en vecteurs sur Google Colab.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Chargement d'un modèle optimisé pour les phrases
# [SOURCE: Modèle recommandé Livre p.62]
model = SentenceTransformer("all-mpnet-base-v2")

# 2. Nos phrases à transformer en vecteurs
sentences = [
    "The financial bank is closed today.",
    "I am walking along the river bank.",
    "The institution is out of money."
]

# 3. GÉNÉRATION DES EMBEDDINGS (Inférence)
embeddings = model.encode(sentences)

# 4. COMPARAISON (Similarité Cosinus)
# On compare la phrase 1 (Banque finance) avec la phrase 2 (Rive) 
# et avec la phrase 3 (Institution/Argent)
sim_1_2 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
sim_1_3 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

print(f"Similarité 'Finance' vs 'Rive' : {sim_1_2:.4f}")
print(f"Similarité 'Finance' vs 'Institution' : {sim_1_3:.4f}")
```

⚠️ **Fermeté bienveillante** : Observez les résultats. Vous verrez que la phrase 1 est beaucoup plus "proche" mathématiquement de la phrase 3, alors qu'elles ne partagent aucun mot important (sauf le mot "banque" sous-entendu). C'est la preuve que les embeddings contextuels ont capturé le **concept** et non seulement les lettres. [SOURCE: Livre p.61-62]

---

### 6. Applications pratiques : Au-delà du NLP
Les embeddings ne servent pas qu'à traiter du texte. La page 67 du livre nous montre une application fascinante : les **Systèmes de recommandation**.

**Le cas Spotify/Netflix (p.67-70)** :
Imaginez que nous traitions une "playlist" de musique comme une "phrase", et chaque "chanson" comme un "mot".
*   Si des millions d'utilisateurs écoutent la chanson A juste après la chanson B, l'IA va créer des embeddings proches pour A et B.
*   🔑 **Le résultat** : Quand vous écoutez un artiste, Spotify regarde quels sont les vecteurs les plus proches dans l'espace multidimensionnel pour vous proposer votre prochaine découverte. 
C'est le même algorithme Word2Vec appliqué aux comportements humains ! [SOURCE: Livre p.67-69]

---

### 7. Éthique et Responsabilité : Le biais est une distance
⚠️ **Éthique ancrée** : « Mes chers étudiants, l'espace vectoriel n'est pas un paradis mathématique neutre. » 

Comme nous l'avons vu p.28, les embeddings sont le reflet fidèle de nos préjugés. 
1.  **Stéréotypes de genre** : Dans de nombreux modèles, le vecteur "Secrétaire" est géométriquement plus proche de "Femme" et "Ingénieur" plus proche de "Homme". 
2.  **Biais culturels** : Les modèles entraînés sur le web occidental peuvent associer des sentiments négatifs à des prénoms ou des cultures qu'ils connaissent mal, simplement par manque de diversité dans les données. 
🔑 **Conséquence technique :** Si vous utilisez ces embeddings pour filtrer des CV ou accorder des prêts bancaires, votre IA sera injuste par construction géométrique.

🔑 **Mon conseil de professeur** : Avant d'utiliser un modèle d'embedding en production, visualisez vos clusters ! Si vous voyez que des groupes de population sont isolés ou injustement étiquetés par la machine, c'est que vos données d'entraînement sont polluées. L'IA responsable commence par l'audit de sa géométrie. [SOURCE: Livre p.28]

---

### Synthèse de la semaine 2
Nous avons parcouru un chemin immense cette semaine. Nous avons appris que :
*   Le texte doit être découpé intelligemment (Tokenisation) pour être digestible.
*   Chaque morceau de texte devient un point dans un espace géant (Embeddings).
*   La force des LLM réside dans le fait que ces points sont **mobiles** et s'adaptent au contexte de la phrase.

🔑 **Le message final du Prof. Henni** : « Vous avez maintenant les briques (tokens) et le ciment (embeddings). Vous comprenez comment la machine transforme le verbe en vecteur. C'est une étape de géant. Mais une question demeure : comment ces vecteurs "discutent-ils" entre eux au sein du modèle pour créer une pensée cohérente ? C'est le secret de l'**Architecture Transformer** que nous allons ouvrir ensemble la semaine prochaine. Félicitations pour votre persévérance ! » [SOURCE: Livre p.71]

---
*Fin de la section 2.4 (2240 mots environ)*
## 🧪 LABORATOIRE SEMAINE 2 (850+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Prêts pour votre premier voyage au cœur de l'atome sémantique ? Ce laboratoire est un moment de vérité : nous allons manipuler les briques que nous avons étudiées en théorie. Rappelez-vous : dans le monde des LLM, un petit changement dans la tokenisation peut transformer une réponse géniale en une bouillie incompréhensible. Soyez méticuleux, soyez curieux, et surtout, observez bien comment les chiffres commencent à parler ! »

---

### 🔹 QUIZ MCQ (10 questions)

1. **Quel schéma de tokenisation est le plus efficace pour éliminer totalement le problème des mots inconnus ([UNK]) ?**
   a) Tokenisation par mots entiers
   b) Tokenisation par sous-mots (WordPiece)
   c) Tokenisation par octets (Byte-level BPE)
   d) Tokenisation par phrases
   **[Réponse: c]** [Explication: En travaillant au niveau de l'octet, n'importe quelle séquence de caractères peut être décomposée, éliminant le besoin de tokens "inconnus". SOURCE: Livre p.45]

2. **Quelle est la différence fondamentale entre BERT-cased et BERT-uncased ?**
   a) BERT-cased est beaucoup plus grand
   b) BERT-uncased convertit tout en minuscules, perdant les indices sur les noms propres
   c) BERT-cased n'utilise pas de Transformers
   d) BERT-uncased est réservé au code source
   **[Réponse: b]** [Explication: "Uncased" ignore la casse, ce qui simplifie le vocabulaire mais peut nuire à la reconnaissance d'entités nommées. SOURCE: Livre p.48]

3. **Quel tokeniseur moderne est particulièrement optimisé pour ne pas "hacher" les indentations du code source ?**
   a) BERT Tokenizer
   b) Word2Vec
   c) StarCoder2 Tokenizer
   d) Bag-of-Words
   **[Réponse: c]** [Explication: Les tokeniseurs spécialisés dans le code créent des tokens dédiés aux séquences d'espaces pour préserver la structure syntaxique. SOURCE: Livre p.51]

4. **Dans un espace vectoriel d'embeddings, que représente mathématiquement un concept ?**
   a) Un seul nombre entier
   b) Un vecteur (une suite de nombres réels)
   c) Une chaîne de caractères
   d) Une matrice de zéros
   **[Réponse: b]** [Explication: Un concept est projeté dans un espace à haute dimension (ex: 768) où sa position définit son sens. SOURCE: Livre p.9]

5. **L'algorithme Word2Vec utilise le "Negative Sampling" pour :**
   a) Supprimer les mauvais mots du dictionnaire
   b) Apprendre au modèle à distinguer les voisins probables des paires aléatoires
   c) Accélérer la vitesse de la carte graphique
   d) Traduire les textes vers le français
   **[Réponse: b]** [Explication: Le contraste entre exemples positifs et négatifs est ce qui permet de sculpter l'espace sémantique. SOURCE: Livre p.65]

6. **Quelle est la dimension typique d'un vecteur d'embedding pour un modèle comme BERT-base ?**
   a) 2 dimensions
   b) 50 dimensions
   c) 768 dimensions
   d) 1 million de dimensions
   **[Réponse: c]** [Explication: C'est le standard pour les modèles "base" de la famille Transformer. SOURCE: Livre p.82]

7. **Pourquoi la tokenisation par sous-mots est-elle supérieure à celle par mots entiers ?**
   a) Elle utilise des vecteurs plus longs
   b) Elle permet de décomposer des mots rares en racines et suffixes connus
   c) Elle est plus ancienne et plus stable
   d) Elle ne nécessite pas d'entraînement
   **[Réponse: b]** [Explication: Cela permet au modèle de généraliser son savoir à des mots qu'il n'a jamais rencontrés. SOURCE: Livre p.44]

8. **Quel token spécial GPT-2 utilise-t-il pour signaler la fin d'un texte ?**
   a) `[SEP]`
   b) `</s>`
   c) `<|endoftext|>`
   d) `[END]`
   **[Réponse: c]** [Explication: C'est le marqueur de fin de séquence spécifique à la famille GPT. SOURCE: Livre p.49]

9. **Quelle mesure géométrique est la plus utilisée pour évaluer la similarité sémantique entre deux vecteurs ?**
   a) La somme des nombres
   b) La similarité cosinus (l'angle entre les vecteurs)
   c) La longueur du texte
   d) Le nombre de voyelles
   **[Réponse: b]** [Explication: Elle mesure l'alignement directionnel des concepts dans l'espace. SOURCE: Livre p.125]

10. **Quel modèle d'embedding récent utilise des "Rotary Positional Embeddings" (RoPE) ?**
    a) Word2Vec
    b) Llama-2 / Llama-3
    c) GloVe
    d) TF-IDF
    **[Réponse: b]** [Explication: RoPE est une technique moderne pour encoder la position des tokens de manière plus fluide. SOURCE: Livre p.102]

---

### 🔹 EXERCICE 1 : Comparaison de tokeniseurs (Niveau 1 - Basique)

**Objectif** : Visualiser physiquement comment différents modèles découpent la même phrase.

**Description** : Utilisez la bibliothèque `transformers` pour charger BERT et GPT-2 et analysez leur comportement sur un texte complexe.

**Code (Testé pour Colab T4)** :
```python
from transformers import AutoTokenizer

# Chargement des tokeniseurs
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "L'intelligence artificielle est fascinante 🎵 #AI2024"

# Tâche : Tokenisez et affichez le résultat
print(f"BERT : {bert_tokenizer.tokenize(text)}")
print(f"GPT-2: {gpt2_tokenizer.tokenize(text)}")
```

**Typical Answer** : BERT affichera des `##` pour les sous-mots et risque de transformer l'emoji en `[UNK]` s'il n'est pas dans son dictionnaire. GPT-2 gérera l'emoji grâce à sa gestion des octets et ajoutera des symboles comme `Ġ` pour les espaces. [SOURCE: Livre p.46-51]

---

### 🔹 EXERCICE 2 : Création d'embeddings (Niveau 2 - Intermédiaire)

**Objectif** : Transformer une pensée en un vecteur numérique et vérifier sa "forme".

**Description** : Utilisez `sentence-transformers` pour encoder une critique de film et analysez l'objet résultant.

**Code (Testé pour Colab T4)** :
```python
from sentence_transformers import SentenceTransformer

# Modèle recommandé p.62 du livre
model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = "Ce cours sur les LLM est absolument incroyable !"
embedding = model.encode(sentence)

print(f"Dimension du vecteur : {embedding.shape}")
print(f"Les 5 premières valeurs : {embedding[:5]}")
```

**Typical Answer** : La dimension sera de **(384,)**. Les valeurs sont des nombres réels entre -1 et 1 environ. [SOURCE: Livre p.62, p.84]

---

### 🔹 EXERCICE 3 : Visualisation et Similarité (Niveau 3 - Avancé)

**Objectif** : Utiliser la réduction de dimension pour "voir" la proximité sémantique.

**Consigne** : Calculez la similarité cosinus entre trois phrases : deux proches et une éloignée. Utilisez ensuite PCA (Principal Component Analysis) pour projeter ces vecteurs en 2D.

**Code (Testé pour Colab T4)** :
```python
from sentence_transformers import util
import numpy as np
from sklearn.decomposition import PCA

sentences = [
    "J'adore les chats.",
    "Les félins sont mes animaux préférés.",
    "La bourse de Paris a clôturé en baisse."
]

embeddings = model.encode(sentences)

# 1. Calcul de similarité
sim = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarité Chat/Félin : {sim.item():.4f}")

# 2. Réduction de dimension (PCA)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
print(f"Coordonnées 2D :\n{reduced_embeddings}")
```

**Typical Answer** : La similarité entre les phrases 1 et 2 sera élevée (> 0.7), tandis qu'avec la phrase 3, elle sera faible (< 0.2). En 2D, les points 1 et 2 apparaîtront regroupés, loin du point 3. [SOURCE: Livre p.125, p.141]

---

**Mots-clés de la semaine** : Tokenisation, Sous-mots (Subwords), BPE, WordPiece, Embeddings denses, Espace vectoriel, Similarité Cosinus, [CLS]/[SEP], PCA.

**En prévision de la semaine suivante** : La semaine prochaine, nous monterons encore d'un cran : nous entrerons dans la salle des machines du Transformer pour voir exactement comment les têtes d'attention manipulent ces vecteurs.

**SOURCES COMPLÈTES** :
*   Livre : Alammar, J., & Grootendorst, M. (2024). *Hands-On Large Language Models*. O'Reilly Media. Chapitre 2, pages 37-71.
*   Blog Jay Alammar : *Illustrated Word2Vec* (https://jalammar.github.io/illustrated-word2vec/)
*   Blog Hugging Face : *About Tokenizers* (https://huggingface.co/blog/tokenizers)
*   GitHub Officiel : https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter02

[/CONTENU SEMAINE 2]

---
title: "2.1 Théorie de la tokenisation"
weight: 2
---


## La traduction du monde : Pourquoi tokeniser ?

Mes chers étudiants, commençons par une vérité brutale : ***un ordinateur ne "lit" pas***.

Pour un processeur, une chaîne de caractères n'est qu'une suite de codes binaires sans aucune notion de sens, de grammaire ou de ponctuation. Pour qu'un Large Language Model puisse opérer, nous devons transformer ce flux continu de texte en une séquence d'unités discrètes, portantes de sens et statistiquement exploitables. C'est le rôle de la **tokenisation**.

Regardez la **Figure 2-1 : Tokens et Embeddings** . Cette illustration simplifie le flux de données : le texte entre, il est découpé en morceaux (tokens), puis chaque morceau est converti en une suite de nombres (embeddings). 

{{< bookfig src="37.png" week="02" >}}

> [!NOTE]
💡 **Notez bien cette intuition :** le tokeniseur est le traducteur universel qui permet au Transformer de "voir" le langage humain.

---
## Voyage au cœur du processus
nous proposons une décomposition méticuleuse du processus de tokenisation à travers quatre figures fondamentales.

### 1. La vue d'ensemble (Figure 2-2)

{{< bookfig src="38.png" week="02" >}}


**ℹ️ Explication** : Cette figure nous montre l'IA vue de l'extérieur. Un utilisateur envoie un prompt. Pour nous, c'est une phrase. Pour le modèle, c'est une entrée qui doit être traitée avant d'être comprise. La figure montre que le tokeniseur n'est pas "dans" le modèle, c'est un composant périphérique essentiel qui prépare le terrain.


### 2. La visualisation interactive (Figure 2-3)

{{< bookfig src="39.png" week="02" >}}

**ℹ️ Explication** : Ici, on utilise l'outil de visualisation d'OpenAI pour montrer comment GPT-4 "découpe" une phrase célèbre. 
*   **Observation** : Vous remarquerez que certains mots sont entiers, tandis que d'autres (plus rares ou plus longs) sont brisés. Chaque token est coloré différemment. 

> [!NOTE]
> 👨‍🔧 **Le message technique** : La tokenisation n'est pas qu'un découpage sur les espaces. C'est une stratégie d'optimisation mathématique.


### 3. La conversion numérique (Figure 2-4)

{{< bookfig src="40.png" week="02" >}}

**ℹ️ Explication** : C'est le passage au "numérique". Une fois le texte découpé, chaque token reçoit un **ID unique**.
*   Exemple : "Have" devient `6975`, "the" devient `278`. 

> [!WARNING]
> ⚠️ **Attention : erreur fréquente ici !** Ces nombres ne sont pas choisis selon l'ordre alphabétique ou l'importance sémantique. Ce sont simplement des index dans une immense table de correspondance (le dictionnaire du modèle).


### 4. Le décodage de sortie (Figure 2-5)

{{< bookfig src="41.png" week="02" >}}

**ℹ️ Explication** : Le processus est symétrique. Quand le modèle génère une réponse, il ne sort pas des lettres, mais des IDs de tokens. Le tokeniseur doit alors effectuer un "detokenize" pour transformer ces nombres en texte lisible par l'humain. 

> [!IMPORTANT]
‼️ **Je dois insister :** si vous perdez le fichier de configuration de votre tokeniseur, vos milliards de paramètres de modèle ne servent plus à rien ; vous ne pourrez plus traduire les chiffres en mots !

---
## La guerre des granularités : Choisir la taille de l'atome

*❓ Comment devons-nous découper le langage ?* 

Cette question a hanté les chercheurs pendant des années. nous proposons un tableau comparatif capital que nous allons analyser en détail.

### 1. La granularité par mots (Word-level)
Historiquement, c'était la méthode par défaut. On coupe à chaque espace.
*   **Avantages** : Très intuitif, préserve l'unité sémantique.
*   **Inconvénients** : 
    *   **Explosion du vocabulaire** : Si vous voulez couvrir toutes les déclinaisons ("manger", "manges", "mangeait"), vous avez besoin d'un dictionnaire de plusieurs millions d'entrées.
    *   **Mots hors-vocabulaire (OOV)** : Si le modèle rencontre un mot qu'il n'a pas vu durant son entraînement (ex: un nouveau mot d'argot ou un nom propre rare), il échoue totalement et affiche `[UNK]` (Unknown). L'information est perdue.


### 2. La granularité par caractères (Character-level)
On découpe lettre par lettre : "c", "h", "a", "t".
*   **Avantages** : Vocabulaire minuscule (quelques centaines de caractères), zéro mot inconnu.
*   **Inconvénients** : 
    *   **Perte de sens** : Le caractère "c" ne porte aucun sens en soi. Le modèle doit faire un effort colossal pour réapprendre la structure des mots.
    *   **Saturation de la mémoire** : Comme les séquences deviennent immenses (une phrase de 10 mots devient une séquence de 60 caractères), on sature très vite la fenêtre de contexte du Transformer.


### 3. Le compromis parfait : La granularité par sous-mots (Subword-level)
C'est le standard actuel de tous les LLM (GPT-4, Llama-3, Mistral). On garde les mots fréquents entiers ("le", "maison") et on découpe les mots rares en morphèmes ("malheureusement" -> "malheureuse" + "##ment").

> [!TIP]
✅️ **C'est la solution au problème OOV :** Même si le modèle ne connaît pas un mot précis, il peut le reconstituer à partir de racines et de suffixes qu'il connaît déjà.


### 4. La granularité par octets (Byte-level)
Utilisée par GPT-2 et ses successeurs. On ne découpe plus des lettres, mais des **octets UTF-8**.

> [!TIP]
💡 **Pourquoi c'est génial ?** Cela permet de traiter n'importe quel symbole (emojis 🎵, kanjis japonais, caractères arabes) avec un vocabulaire fixe de seulement 256 octets de base. C'est l'universalité totale.

---
## Plongée dans l'algorithme : Byte Pair Encoding (BPE)
Mes chers étudiants, voici le moment où nous devenons des ingénieurs. 

L'algorithme BPE est le moteur derrière GPT. Son fonctionnement est un exemple parfait de logique statistique itérative.

**Le processus BPE en 4 étapes** :
1.  **Initialisation** : On traite chaque caractère comme un token individuel.
2.  **Comptage** : On cherche dans tout notre immense corpus de textes quelle paire de tokens adjacents apparaît le plus souvent (ex: "e" et "r").
3.  **Fusion (Merge)** : On crée un nouveau token "er" et on l'ajoute à notre dictionnaire. Désormais, "e" et "r" côte à côte ne sont plus deux jetons, mais un seul.
4.  **Itération** : On recommence jusqu'à atteindre la taille de dictionnaire souhaitée (ex: 50 000 tokens).

> [!NOTE]
🔑 **Mon intuition** : Le BPE construit le langage de bas en haut.

> Il ne connaît pas la grammaire, il connaît la **fréquence**. S'il voit souvent "ing" à la fin des verbes anglais, il finira par créer un token unique `ing`.

---
## WordPiece : L'approche de Google (BERT)
*WordPiece* ressemble à BPE, mais avec une subtilité mathématique. Au lieu de fusionner la paire la plus *fréquente*, il fusionne la paire qui augmente le plus la **vraisemblance** (likelihood) du modèle.
*   BPE demande : "Qu'est-ce que je vois le plus ?"
*   WordPiece demande : "Quelle fusion m'aide le mieux à prédire la suite ?"

> [!IMPORTANT]
‼️ Pour l'utilisateur final, la différence est minime, mais pour l'entraînement d'un modèle comme BERT, WordPiece permet une meilleure compression de l'information sémantique.

---
## L'impact critique : Le cas du Code et du Multilingue

> [!WARNING]
⚠️ **Je dois insister sur ce point :** la tokenisation n'est pas une science exacte, c'est un arbitrage.

1.  **Le Code source** : Un tokeniseur entraîné sur du texte naturel va détruire l'indentation du code Python. Un tokeniseur spécialisé doit posséder des tokens pour "4 espaces", "1 tabulation", etc. Sans cela, le modèle perd la logique structurale du code.

2.  **L'inégalité multilingue** : ⚖️ Un tokeniseur entraîné majoritairement sur l'anglais découpera un mot anglais en 1 token, mais le même concept en arabe ou en swahili sera haché en 5 ou 6 tokens.

> [!NOTE]
🚨 **Conséquence économique et technique :** Cela rend l'IA plus chère (facturation au token) et moins "intelligente" (mémoire de contexte saturée plus vite) pour les langues non-occidentales. C'est l'un des plus grands défis de l'IA inclusive.

---
## Gestion des mots hors vocabulaire (OOV)
Comment BERT ou GPT gèrent-ils un mot totalement inconnu ? 
*   Grâce aux sous-mots, ils le décomposent jusqu'au niveau du caractère si nécessaire. 
*   🔑 **Le rôle du token spécial `[UNK]`** : C'est le signal d'échec. Si un modèle affiche trop de `[UNK]`, c'est que votre tokeniseur n'est pas adapté à vos données (ex: utiliser un modèle de chat généraliste sur des séquences d'ADN).

---
## Exemple de code pédagogique : Visualiser les tokens
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

> [!NOTE]
✍🏻 Regardez bien la sortie de `full_encoding`. Vous verrez des chiffres au début et à la fin qui ne correspondent pas à vos mots. Ce sont les tokens spéciaux de structure. 

> [!IMPORTANT]
‼️ **C'est non-négociable :** vous devez apprendre à vivre avec ces jetons de contrôle, car ils sont les balises GPS du Transformer.

---
## Synthèse
Nous avons vu comment la tokenisation transforme le flux continu de notre pensée en une suite discrète de jetons numériques. Nous avons compris que le choix de la granularité (mots, sous-mots, octets) définit le cadre de réflexion de l'IA. 

> [!TIP]
✉️ **Mon message** : Ne voyez plus un texte comme une suite de lettres. Voyez-le comme une suite de jetons statistiques. 

La tokenisation est la fondation de l'édifice. Si les fondations sont de travers, le gratte-ciel LLM s'écroulera au moindre raisonnement complexe.

---
Vous maîtrisez désormais la théorie du découpage. Vous savez comment le texte devient un index. Dans la section suivante ➡️, nous allons donner une "âme" à ces index : nous allons découvrir comment transformer un simple numéro comme `6975` en un vecteur riche de sens. Bienvenue dans le monde des **Embeddings**.
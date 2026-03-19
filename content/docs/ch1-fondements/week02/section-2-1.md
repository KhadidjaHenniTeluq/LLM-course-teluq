---
title: "2.1 Théorie de la tokenisation"
weight: 2
---

## Pourquoi les machines ne lisent-elles pas comme nous ?

Avant d'entrer dans les algorithmes, posons les bases. Un ordinateur ne "lit" pas une chaîne de caractères au sens humain. Il traite des nombres. La **tokenisation** est l'étape de traduction universelle : c'est le processus qui transforme un texte brut en une séquence d'unités discrètes appelées **tokens**. 

Comme l'illustrent les **Figures 2-1 à 2-4**, la tokenisation n'est pas une simple étape de nettoyage ; c'est la création d'un pont entre notre langage continu et le monde discret des mathématiques. Si vous regardez la **Figure 2-1**, vous verrez qu'un LLM comme GPT-4 ne voit pas le mot "Bonjour", il voit un index (par exemple, 15432) dans une table de correspondance géante.

{{< bookfig src="38.png" week="02" >}}
{{< bookfig src="39.png" week="02" >}}
{{< bookfig src="40.png" week="02" >}}
{{< bookfig src="41.png" week="02" >}}

## Les 4 grands schémas de tokenisation : Une question de granularité

Il existe plusieurs façons de découper le fromage du langage. Chaque méthode a ses forces et ses faiblesses.

### 1. La tokenisation par mots (Word Tokenization)

C'est la méthode la plus intuitive. On coupe à chaque espace ou signe de ponctuation.
*   **Avantages** : Les mots portent un sens clair pour nous.
*   **Inconvénients** : 

{{% hint warning %}}
**Attention : erreur fréquente ici !** Si vous utilisez cette méthode, votre vocabulaire explose. Entre "marcher", "marchons", "marchait", vous avez trois entrées différentes. Surtout, vous tombez sur le problème des **mots hors vocabulaire (OOV - Out of Vocabulary)**. Si le modèle rencontre un mot qu'il n'a pas vu durant l'entraînement, il affiche le redoutable token `[UNK]` (Unknown), perdant toute information.
{{% /hint %}}

### 2. La tokenisation par caractères (Character Tokenization)

On découpe chaque lettre : "c-h-a-t".
*   **Avantages** : Vocabulaire minuscule (quelques centaines de caractères), zéro mot inconnu.
*   **Inconvénients** : Chaque token individuel ne porte aucun sens. Le modèle doit faire un effort colossal pour réapprendre que "c+h+a+t" signifie un félin. De plus, les séquences deviennent immenses, saturant la fenêtre de contexte du modèle.

### 3. La tokenisation par sous-mots (Subword Tokenization) - Le Graal moderne

C'est le compromis utilisé par presque tous les LLM actuels (GPT, Llama, BERT). On garde les mots fréquents entiers ("le", "est"), mais on découpe les mots complexes ou rares en morceaux porteurs de sens (morphèmes). Par exemple, "malheureusement" pourrait devenir `malheureuse` + `##ment`. 

{{% hint info %}}
🔑 **Je dois insister :** C'est cette méthode qui permet aux modèles de comprendre des mots qu'ils n'ont jamais vus en analysant leurs racines et suffixes !
{{% /hint %}}

### 4. La tokenisation par octets (Byte-level Tokenization)

Utilisée notamment par GPT-2 et GPT-4, cette méthode traite le texte comme une suite d'octets (UTF-8). Cela permet de traiter n'importe quel caractère, y compris les emojis 🎵 ou les langues rares, sans jamais avoir de token "inconnu".

## Plongée dans les algorithmes : BPE vs WordPiece

### Byte Pair Encoding (BPE)

C'est l'algorithme star de la famille GPT. Son fonctionnement est itératif et fascinant :
1.  On commence par traiter chaque caractère comme un token.
2.  On cherche la paire de tokens la plus fréquente dans le corpus (ex: "e" et "r").
3.  On les fusionne pour créer un nouveau token "er".
4.  On recommence jusqu'à atteindre la taille de vocabulaire souhaitée (ex: 50 000).

{{% hint info %}}
🔑 **Notez bien cette intuition :** BPE construit le langage de bas en haut, en se basant uniquement sur la fréquence statistique des associations de caractères.
{{% /hint %}}

### WordPiece

Utilisé par BERT, cet algorithme ressemble à BPE mais avec une nuance mathématique subtile. Au lieu de fusionner la paire la plus *fréquente*, il fusionne la paire qui maximise la **vraisemblance** (likelihood) des données d'entraînement. En d'autres termes, il se demande : "Quelle fusion m'aide le mieux à prédire la structure du langage ?".

## Impact sur la qualité des modèles

La qualité de votre tokeniseur définit le "plafond" de performance de votre LLM. 
*   **Indentation et Code** : Pour des modèles comme StarCoder, il est vital que les espaces et les tabulations soient des tokens spécifiques. Si un tokeniseur fusionne mal les espaces, le modèle ne comprendra jamais la structure d'un code Python (voir Semaine 13).
*   **Multilingue** : Un tokeniseur entraîné sur l'anglais découpera maladroitement le français ou l'arabe, créant trop de tokens pour une seule phrase, ce qui réduit l'efficacité du modèle.

## Exemple de code : Tokenisation avec Hugging Face Transformers

Pour bien ancrer cela, regardons comment utiliser l'un des tokeniseurs les plus célèbres, celui de BERT.

```python
# Installation : pip install transformers
from transformers import AutoTokenizer

# Chargement d'un tokeniseur standard
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "LLMs are fascinating, aren't they?"

# 1. Encodage : Transformation en IDs
tokens_info = tokenizer(text)
input_ids = tokens_info['input_ids']

print(f"IDs des tokens : {input_ids}")

# 2. Décodage pour voir le découpage
tokens_list = tokenizer.convert_ids_to_tokens(input_ids)
print(f"Découpage visuel : {tokens_list}")
```
<!-- TODO: add colab link -->

{{% hint warning %}}
Observez le résultat du code ci-dessus. Vous verrez des tokens spéciaux comme `[CLS]` ou `[SEP]`. 🔑 **C'est non-négociable :** vous devez comprendre que ces tokens ne correspondent à aucun mot humain ; ils sont des signaux de contrôle pour le modèle (début de séquence, séparation de phrases). Nous en aurons besoin pour toutes nos tâches de classification en Semaine 4.
{{% /hint %}}

## Éthique et Transparence : Les oubliés de la tokenisation

{{% hint danger %}}
« Mes chers étudiants, la tokenisation n'est pas neutre. » 
Les tokeniseurs sont entraînés sur des corpus massifs, souvent dominés par l'anglais. 
1.  **Le coût de la langue** : Un utilisateur écrivant en anglais consommera moins de tokens pour la même idée qu'un utilisateur écrivant en amharique ou en wolof. Comme les API de LLM (OpenAI, Anthropic) facturent au token, cela crée une inégalité économique directe basée sur la langue.
2.  **Représentation** : Si un tokeniseur n'a jamais vu de termes techniques médicaux ou juridiques dans sa phase d'apprentissage, il va les "hacher" en petits morceaux insignifiants, rendant la tâche du modèle beaucoup plus difficile. 

🔑 **Je dois insister :** Toujours vérifier si le tokeniseur de votre modèle est adapté à la langue et au domaine de votre application. C'est la première étape d'une IA responsable.
{{% /hint %}}

« Voilà pour les bases théoriques ! Nous avons vu comment le texte devient une suite de nombres. Dans la section suivante, nous allons comparer les tokeniseurs des plus grands modèles actuels pour voir comment ces théories se traduisent en pratique. »

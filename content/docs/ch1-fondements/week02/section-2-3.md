---
title: "2.3 Propriétés des tokeniseurs modernes"
weight: 4
---

## Au-delà du découpage : L'anatomie d'un bon tokeniseur

« Bonjour à toutes et à tous ! Nous avons comparé les différents visages des tokeniseurs dans la section précédente. Maintenant, je veux que nous regardions sous le capot. Pourquoi certains tokeniseurs réussissent là où d'autres échouent lamentablement ? » 

Un tokeniseur n'est pas juste un script qui coupe des mots ; c'est un système avec des propriétés mathématiques et structurelles précises. Selon Alammar et Grootendorst (p. 55), il existe un ensemble de paramètres et de choix de conception qui dictent l'intelligence future du modèle. Si vous comprenez ces propriétés, vous saurez prédire les limites d'un LLM avant même de lui avoir posé une question. 

## 1. La taille du vocabulaire ($V$) : L'équilibre délicat

{{% hint info %}}
🔑 **Je dois insister :** La taille du vocabulaire est le paramètre le plus influent. 
{{% /hint %}}

*   **Petit vocabulaire (ex: 30 000 tokens)** : C'est le choix de BERT. L'avantage est que la matrice d'embeddings est légère, ce qui économise de la mémoire GPU. L'inconvénient est la fragmentation : les mots rares sont découpés en de nombreux petits morceaux, ce qui rend la compréhension sémantique plus difficile.
*   **Grand vocabulaire (ex: 100 000+ tokens)** : C'est le choix de GPT-4 ou Llama-3. Cela permet de représenter des concepts complexes (ex: "anticonstitutionnellement") en un seul ou deux tokens. Cela améliore l'efficacité car on traite plus de sens avec moins d'unités. 

{{% hint warning %}}
**Attention : erreur fréquente ici !** On pourrait croire que plus le vocabulaire est grand, mieux c'est. Mais attention au "problème des données creuses" : si votre vocabulaire est trop immense par rapport à vos données d'entraînement, certains tokens ne seront jamais vus, et le modèle n'apprendra jamais leur sens.
{{% /hint %}}

## 2. Les Tokens Spéciaux : Le langage secret du modèle

Les LLM ne communiquent pas seulement avec des mots, ils utilisent des signaux de contrôle. Imaginez que vous dirigiez un orchestre : vous avez besoin de signes pour dire "commencez" ou "arrêtez".
*   `<s>` ou `[CLS]` : Indique le début d'une séquence.
*   `</s>` ou `[SEP]` : Indique la fin ou sépare deux phrases.
*   `[PAD]` : Utilisé pour égaliser la taille des phrases dans un lot (batch) de calcul.
*   `[MASK]` : Utilisé durant l'entraînement pour cacher un mot que le modèle doit deviner.

{{% hint info %}}
🔑 **C'est une distinction non-négociable :** Sans ces tokens spéciaux, le modèle ne peut pas structurer sa pensée. Par exemple, BERT utilise le token `[CLS]` (Classification) pour résumer tout le sens d'une phrase en un seul point.
{{% /hint %}}

## 3. La gestion de la casse et des domaines

Faut-il convertir "APPLE" en "apple" ? Comme nous l'avons vu, cela dépend de votre tâche.
*   **Modèles Uncased** : Excellents pour la recherche d'information générale ou le clustering où le sens prime sur la forme.
*   **Modèles Cased** : Indispensables pour le code (où `Variable` et `variable` sont deux choses différentes) ou la reconnaissance d'entités nommées.

## Étude de cas : Texte naturel vs. Code source

« Imaginez un instant que vous demandiez à un tokeniseur entraîné sur des romans de lire un script Python. »
Le texte naturel est riche en morphologie (racines, suffixes). Le code source est riche en ponctuation et en indentations. 

Regardez la différence de traitement pour ce bloc de code :
```python
if x > 10:
    print("Success")
```
*   **Tokeniseur généraliste** : Il pourrait ignorer les 4 espaces de l'indentation ou fusionner `if` et `x`. Pour Python, c'est une catastrophe syntaxique !
*   **Tokeniseur spécialisé (StarCoder/Codex)** : Il traite chaque groupe d'espaces comme un token spécifique. Il reconnaît `if` comme une entité unique. 

{{% hint info %}}
🔑 **Notez bien :** L'efficacité d'un tokeniseur se mesure souvent par son **ratio Tokens/Caractères**. Plus ce ratio est bas pour un domaine donné, plus le tokeniseur est "intelligent" pour ce domaine.
{{% /hint %}}

## Le défi du multilingue : L'universalité en question

Comment gérer 100 langues avec un seul tokeniseur ? C'est le défi des modèles comme mBERT ou Bloom. La propriété clé ici est le **partage de vocabulaire**.
Si vous utilisez un tokeniseur entraîné à 90% sur l'anglais, il va "hacher" les mots français. Par exemple, "constitutionnel" deviendra `con` + `stit` + `uti` + `on` + `nel`. 

{{% hint danger %}}
« Mes chers étudiants, soyez vigilants. » Un modèle qui utilise trop de tokens pour une langue donnée est un modèle qui a moins de "mémoire" pour cette langue, car sa fenêtre de contexte (ex: 4096 tokens) se remplit beaucoup plus vite. C'est un biais technique qui favorise les langues dominantes.
{{% /hint %}}

## Bonnes pratiques de sélection

Comment choisir le bon tokeniseur pour votre projet ? Professeur Henni vous donne sa checklist :
1.  **Correspondance de domaine** : Si vous faites du médical, votre tokeniseur connaît-il le vocabulaire latin des maladies ?
2.  **Compression** : Testez votre texte sur plusieurs tokeniseurs (via Hugging Face `Tokenizer.encode`). Celui qui produit le moins de tokens est généralement le plus efficace.
3.  **Gestion des inconnus** : Le modèle utilise-t-il les octets (Byte-level) pour éviter le token `[UNK]` ? C'est crucial pour la robustesse en production.

## Analogie finale

La tokenisation est comme un **tamis**. Si les mailles sont trop larges, tout passe sans distinction (Caractères). Si elles sont trop étroites, vous ne récupérez que des blocs massifs impossibles à analyser (Mots entiers). Le tokeniseur moderne est un tamis magique qui ajuste la taille de ses mailles dynamiquement pour capturer exactement le sens là où il se trouve.

« Vous maîtrisez maintenant les propriétés structurelles des tokens. Mais ces nombres ne sont encore que des étiquettes vides de sens. Dans la section suivante, nous allons donner de la profondeur à ces nombres en découvrant les **Embeddings** : comment transformer un index en un vecteur vibrant de sens sémantique. »

---
title: "2.2 Comparaison des tokeniseurs modernes"
weight: 3
---

## La diversité des approches : Pourquoi comparer ?

« Mes chers étudiants, si vous pensiez que tous les modèles "lisaient" de la même manière, cette section va vous ouvrir les yeux ! » Comme nous l'avons vu en section 2.1, la théorie nous donne les outils, mais la pratique nous montre une diversité fascinante. Choisir le mauvais modèle pour votre application, ce n'est pas seulement une question de performance, c'est parfois rendre la tâche impossible au modèle. 

Regardez le tableau récapitulatif inspiré des pages 46 à 54 du livre. Nous allons comparer les géants : **BERT**, la famille **GPT**, **Flan-T5**, et les spécialistes comme **StarCoder2** et **Galactica**. 

{{% hint info %}}
🔑 **Je dois insister :** chaque différence que nous allons noter a été pensée pour résoudre un problème spécifique de compréhension.
{{% /hint %}}

## BERT : L'ancêtre rigoureux (Cased vs. Uncased)

BERT (2018) utilise l'algorithme **WordPiece**. Sa particularité ? Il existe en deux versions majeures.
1.  **BERT-uncased** : Tout est converti en minuscules. "Paris" devient "paris". 

{{% hint warning %}}
**Attention : erreur fréquente ici !** On pourrait croire que c'est plus simple, mais pour une tâche de détection d'entités nommées (NER), on perd l'indice capital de la majuscule qui distingue le prénom "Rose" de la fleur "rose".
{{% /hint %}}

2.  **BERT-cased** : Préserve la casse. C'est le standard pour les tâches où la structure propre du nom est capitale.

Observez la **Figure 2-5** ou la description p.48 : BERT utilise la notation `##` pour indiquer qu'un token est la suite d'un mot. Par exemple, "embeddings" pourrait être découpé en `em`, `##bed`, `##dings`. Si un mot n'est pas dans son dictionnaire de 30 000 mots, il utilise le token `[UNK]`. 

{{< bookfig src="26.png" week="02" >}}

{{% hint info %}}
🔑 **Notez bien :** un dictionnaire de 30k est considéré comme "petit" aujourd'hui.
{{% /hint %}}

## La famille GPT : De l'efficacité à l'omniscience

Les modèles d'OpenAI utilisent le **Byte-level BPE**. 
*   **GPT-2 (2019)** : Un vocabulaire de environ 50 000 tokens. Il a introduit une astuce géniale : représenter l'espace *avant* le mot par un caractère spécial (souvent noté `Ġ` dans les visualisations). 
*   **GPT-4 (2023)** : On passe à la vitesse supérieure avec un vocabulaire dépassant les 100 000 tokens. Pourquoi une telle inflation ? Pour être plus efficace. Plus le dictionnaire est grand, plus le modèle peut représenter de longs mots complexes en un seul token, ce qui libère de la place dans sa "fenêtre de contexte".

## Flan-T5 et SentencePiece : L'approche "tout-en-un"

Flan-T5 utilise **SentencePiece**. Contrairement à BERT, il ne traite pas les espaces comme des séparateurs à part, mais comme des caractères normaux (souvent remplacés par un tiret bas `_`). 

{{% hint info %}}
🔑 **La distinction majeure :** SentencePiece est conçu pour être indépendant de la langue. Il ne suppose pas que les mots sont séparés par des espaces, ce qui le rend redoutable pour le japonais ou le mandarin. Cependant, comme vous le voyez p.50, il peut être "aveugle" aux retours à la ligne, ce qui pose problème pour analyser des listes ou du code source.
{{% /hint %}}

## Les spécialistes : Quand le domaine dicte la forme

C'est ici que l'ingénierie devient de l'art. Si vous voulez que votre modèle soit un génie des mathématiques ou du code, vous ne pouvez pas utiliser le tokeniseur de Monsieur Tout-le-monde.

### StarCoder2 : Le traducteur de code

Pour le code source, la structure est tout. StarCoder2 a deux secrets :
1.  **Tokenisation des chiffres** : Contrairement à GPT-2 qui peut voir "123" comme un seul token, StarCoder2 découpe souvent chiffre par chiffre (`1`, `2`, `3`). Pourquoi ? Pour que le modèle apprenne réellement à faire des additions au lieu de simplement mémoriser des nombres.
2.  **Préservation des indentations** : Il possède des tokens spécifiques pour "quatre espaces", "huit espaces", etc. Sans cela, le modèle perdrait la structure des boucles Python.

### Galactica : Le scientifique

Le modèle de Meta pour la science doit gérer du LaTeX (formules mathématiques) et des séquences ADN. Son tokeniseur est entraîné pour ne pas "hacher" les formules chimiques complexes, permettant au modèle de voir `H2O` comme une entité cohérente plutôt que comme une suite de caractères aléatoires.

## Exemple de comparaison multilingue

Imaginez le mot "manger". 
*   Un tokeniseur anglais pourrait le découper en `man` + `ger`. 
*   Un tokeniseur français bien entraîné (comme CamemBERT) le verra comme un seul token `manger`. 

{{% hint info %}}
🔑 **Je dois insister :** Cette fragmentation excessive (over-segmentation) est le fléau des modèles mal adaptés. Si un mot français est découpé en 4 tokens alors qu'un mot anglais équivalent n'en utilise qu'un seul, votre modèle français sera 4 fois plus lent et aura 4 fois moins de mémoire contextuelle.
{{% /hint %}}

## Laboratoire de code : Comparaison Hugging Face

Voici comment vous pouvez tester ces différences vous-mêmes sur Google Colab.

```python
# Installation requise : pip install transformers
from transformers import AutoTokenizer

# Sélection de 3 tokeniseurs aux philosophies différentes
tokenizers = {
    "BERT (Social)": "bert-base-uncased",
    "GPT-2 (Général)": "gpt2",
    "Llama-3 (Moderne)": "meta-llama/Meta-Llama-3-8B" # Nécessite accès HF ou version d'essai
}

text = "LLM tokenization is 100% vital for AI."

for name, model_id in tokenizers.items():
    try:
        tk = AutoTokenizer.from_pretrained(model_id)
        tokens = tk.tokenize(text)
        print(f"--- {name} ---")
        print(f"Nombre de tokens : {len(tokens)}")
        print(f"Découpage : {tokens}\n")
    except Exception as e:
        print(f"Note : {name} nécessite une authentification ou n'est pas disponible sans accès spécifique.")
```
<!-- TODO: add colab link -->

## Synthèse des propriétés et impact sur la performance

Pourquoi tout cela est-il capital pour votre futur métier ? 
1.  **Efficacité Computationnelle** : Moins vous avez de tokens pour un texte donné, plus l'inférence (la réponse du modèle) est rapide et économique.
2.  **Qualité des Représentations** : Un tokeniseur qui respecte la morphologie de la langue (ex: séparer le radical de la terminaison d'un verbe) aide énormément le modèle à généraliser.
3.  **Gestion des Nombres** : Comme nous l'avons vu avec StarCoder2, la façon dont les chiffres sont découpés impacte directement les capacités de calcul du LLM.

## Éthique et Inégalités Numériques

{{% hint danger %}}
« Regardez au-delà de la technique. » 
Il existe une véritable "fracture du token". Les langues à alphabet latin sont extrêmement bien servies par les tokeniseurs actuels. Mais pour les langues d'Afrique ou d'Asie du Sud, un seul mot peut parfois être découpé en une dizaine d'octets. 

🔑 **Conséquence éthique :** Cela signifie que pour dire la même chose, un locuteur de langue "rare" paiera plus cher et subira un modèle moins intelligent (car sa fenêtre de contexte sera saturée plus vite). En tant qu'experts, vous devez militer pour des tokeniseurs plus inclusifs, comme ceux de la famille **Bloom** ou **Llama-3**, qui ont fait des efforts considérables pour élargir leur vocabulaire multilingue.
{{% /hint %}}

« Vous avez maintenant une vue d'ensemble de la jungle des tokeniseurs. Vous comprenez que le choix du modèle commence par l'analyse de son dictionnaire. Dans la section suivante, nous allons étudier les propriétés techniques précises qui font qu'un tokeniseur est "bon" ou "mauvais" pour une tâche donnée. »

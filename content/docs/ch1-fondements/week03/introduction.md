---
title: "Introduction"
weight: 1
---

## Au cœur des Transformers : Mécanismes d'attention et blocs Transformer

Bienvenue à toutes et à tous ! Nous entamons aujourd'hui la phase la plus spectaculaire de notre cursus. Imaginez que jusqu'ici, nous n'ayons fait qu'accorder nos instruments (les tokens) et choisir nos partitions (les embeddings). Aujourd'hui, mes chers étudiants, nous allons faire monter le chef d'orchestre sur l'estrade. 

> [!IMPORTANT]
📌 **Je dois insister :** l'Architecture Transformer n'est pas qu'un simple algorithme, c'est une symphonie de calculs où chaque mot devient un musicien capable d'écouter et de répondre à tous les autres en temps réel. 

Nous allons décortiquer comment le mécanisme d'**Attention** permet cette harmonie parfaite, transformant une suite de chiffres en une véritable structure de compréhension. Respirez, la musique de l'IA va commencer !

---
**Rappel semaine précédente** : La semaine dernière, nous avons exploré les atomes du langage : les tokens. Nous avons appris comment les tokeniseurs découpent le texte et comment les embeddings transforment ces morceaux en vecteurs denses, créant ainsi une géométrie du sens.

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
> *   Expliquer mathématiquement le mécanisme de Self-Attention (Q, K, V).
> *   Comprendre le rôle des têtes d'attention multiples (Multi-head attention).
> *   Détailler le fonctionnement de l'encodage positionnel moderne (RoPE).
> *   Analyser la structure d'un bloc Transformer complet (Normalisation, Feedforward, Résidus).
> *   Saisir l'importance de l'optimisation par KV Cache pour l'inférence.

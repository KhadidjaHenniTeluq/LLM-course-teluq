---
title: "Introduction"
weight: 1
---

## Les briques fondamentales des LLM : De la tokenisation aux représentations vectorielles

Bienvenue dans les coulisses de la donnée textuelle ! Après avoir compris l'architecture globale des Transformers lors de notre première semaine, il est temps de se poser une question fondamentale : comment faire entrer nos phrases dans ces réseaux de neurones complexes ? 

Le texte humain, avec toutes ses nuances, n'est pas directement assimilable par une machine qui ne lit que des nombres. Nous allons découvrir comment l'IA fragmente nos phrases en unités de base appelées **tokens**, avant de les convertir en coordonnées mathématiques : les **embeddings**.

> [!IMPORTANT]
🔑 **L'enjeu est colossal :** Un mauvais algorithme de tokenisation, et c'est toute la logique du modèle qui s'effondre (nous le verrons avec les mathématiques ou le code informatique). Préparez-vous à manipuler la matière première des LLMs !

---
**Rappel de la semaine précédente** : Nous avons retracé l'évolution des représentations textuelles, de la simple "sacoche de mots" (Bag-of-Words) aux premiers embeddings denses comme Word2Vec. Surtout, nous avons compris comment le mécanisme d'attention a révolutionné le traitement du contexte en dépassant les vieux réseaux récurrents (RNN).

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
> *   Expliquer la théorie mathématique et algorithmique de la tokenisation moderne.
> *   Distinguer les quatre schémas de granularité : mots, sous-mots, caractères et octets.
> *   Comprendre le fonctionnement interne des algorithmes BPE (Byte Pair Encoding) et WordPiece.
> *   Analyser l'impact des choix de tokenisation sur la performance des modèles (code, langues, mathématiques).
> *   Maîtriser la création d'embeddings contextuels, base de la compréhension sémantique profonde.

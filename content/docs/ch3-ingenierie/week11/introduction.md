---
title: "Introduction"
weight: 1
---

## Adapter les LLM à vos besoins : Le fine-tuning supervisé

Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour entamer cette semaine. Nous avons parcouru un chemin immense : de la "sacoche de mots" à la vision artificielle. 

Aujourd'hui, nous franchissons une étape capitale, celle qui transforme un simple utilisateur en un véritable architecte de l'IA. Jusqu'ici, nous avons utilisé des modèles pré-entraînés par des géants comme Google, Meta ou Microsoft. Mais que se passe-t-il quand vous avez besoin d'un expert dans un domaine ultra-spécifique, comme le droit constitutionnel canadien ou la maintenance des réacteurs nucléaires ? 

> [!IMPORTANT]
🔑 **Je dois insister :** ne vous contentez pas d'utiliser l'IA des autres. Apprenez à forger votre propre intelligence. 

Préparez-vous, car nous allons apprendre à transformer un géant généraliste en un spécialiste de pointe, même si vous n'avez pas un supercalculateur dans votre garage !


**Rappel semaine précédente** : La semaine dernière, nous avons ouvert les yeux de l'IA grâce à la multimodalité, en apprenant comment CLIP et BLIP-2 permettent d'aligner le texte et l'image dans un même espace sémantique.


**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
> *   Expliquer la différence fondamentale entre l'entraînement initial (Pretraining) et l'adaptation (Fine-tuning).
> *   Comparer les coûts et les bénéfices du "Full Fine-Tuning" par rapport aux méthodes "PEFT".
> *   Comprendre le fonctionnement mathématique de la méthode LoRA (Low-Rank Adaptation).
> *   Maîtriser les concepts de quantification (4-bit, NF4) pour faire tourner de gros modèles sur de petits GPU.
> *   Mettre en œuvre un pipeline complet de QLoRA sur un GPU T4.

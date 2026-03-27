---
title: "Introduction"
weight: 1
---

## Combattre les hallucinations : L'architecture RAG
Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour cette étape. Jusqu'ici, nous avons vu que les LLM sont des génies créatifs, capables de discourir sur n'importe quel sujet avec une fluidité déconcertante. Mais il y a un "mais" : cette fluidité cache parfois des mensonges statistiques. 

> [!IMPORTANT]
🔑 **Je dois insister :** dans le monde professionnel, l'imagination n'est pas une excuse pour l'erreur factuelle. 

Aujourd'hui, nous allons apprendre à transformer nos IA en experts rigoureux qui ne parlent que "sous contrôle de preuves". Bienvenue dans l'univers du **RAG**, la technologie qui permet à vos modèles de passer l'examen à "livre ouvert". Prêt·e·s à ancrer l'IA dans la réalité ?

---

**Rappel semaine précédente** : La semaine dernière, nous avons appris l'art subtil du prompt engineering, en explorant les techniques de Few-shot learning et de Chain-of-Thought pour guider le raisonnement des modèles.

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
> *   Expliquer l'architecture fondamentale d'un système RAG.
> *   Distinguer les trois phases : Indexation, Récupération et Génération.
> *   Comprendre comment le RAG réduit drastiquement les hallucinations.
> *   Mettre en œuvre une génération "ancrée" (*grounded*) avec citations de sources.
> *   Maîtriser les stratégies de récupération avancées comme le Reranking.

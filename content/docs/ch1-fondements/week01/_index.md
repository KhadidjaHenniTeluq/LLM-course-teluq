---
title: "Semaine 1 : Introduction aux LLM et historique du NLP"
weight: 1
bookCollapseSection: true
---


# 📝 Semaine 1 : Introduction aux LLM et historique du NLP

Bonjour à toutes et à tous ! Bienvenue dans ce qui est, pour moi, le début d'une fascinante odyssée. Nous vivons une époque où les machines commencent à jongler avec les mots presque aussi bien que nous. Mais pour comprendre comment nous en sommes arrivés là, il faut remonter aux racines. 

> [!IMPORTANT]
📌 **Je dois insister :** on ne peut pas maîtriser GPT-4 sans comprendre pourquoi ses ancêtres ont échoué. 

Aujourd'hui, nous allons voyager de la simple statistique de comptage jusqu'à l'étincelle de l'Attention qui a tout changé. Respirez, nous posons aujourd'hui les fondations de votre expertise !

---
## 📖 Description générale
Cette première semaine pose le cadre historique et conceptuel du Traitement du Langage Naturel (NLP). 

Nous allons retracer le passage des méthodes **symboliques** et statistiques (Bag-of-Words, TF-IDF) aux méthodes **neuronales** (Word2Vec). Nous analyserons les limites structurelles des réseaux récurrents (**RNN** et **LSTM**) qui ont longtemps constitué un goulot d'étranglement pour l'IA. Enfin, nous introduirons le paradigme de l'**Attention** et l'architecture **Transformer**, qui sont les piliers technologiques de tous les Large Language Models (LLM) modernes. Nous conclurons par une réflexion sur le cycle de vie d'un LLM (Pré-entraînement et Fine-tuning) et les enjeux éthiques cruciaux du domaine.

---
## 🧠 Pré-requis importants
Pour bien démarrer ce cours, vous devriez être familiers avec :
1.  **Python intermédiaire** : Savoir manipuler des listes, des dictionnaires et des fonctions.
2.  **Bases de l'Apprentissage Automatique** : Comprendre ce qu'est un modèle, des données d'entraînement et une prédiction.
3.  **Calcul matriciel élémentaire** : Comprendre qu'un texte peut être représenté par une grille de chiffres.

**Ressources pour réviser les pré-requis :**
*   💻 **Tutoriel Python** : [NLP Tutorial Python](https://www.youtube.com/playlist?list=PLeo1K3hjS3uuvuAXhYjV2lMEShq2UYSwX) .
*   📚 **Cours Machine Learning (Google)** : [Introduction au ML](https://developers.google.com/machine-learning/crash-course/ml-intro?hl=fr) .

---

## 📚 Ressources utiles pour les concepts de la semaine

### 🌐 Articles et Blogs de référence
*   **Jay Alammar** : [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) – L'explication visuelle la plus célèbre au monde pour comprendre l'attention.
*   **Jay Alammar** : [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) – Pour comprendre le passage du mot au vecteur dense.
*   **Hugging Face Course** : [Introduction to NLP and LLMs](https://huggingface.co/learn/nlp-course/chapter1/1) – Une excellente mise en contexte des tâches de l'IA de langage.

### 📺 Vidéos recommandées
*   🎥 **Andrej Karpathy** : [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) – Une conférence magistrale sur ce qu'est un LLM "sous le capot".
*   🎥 **StatQuest** : [Transformer Architecture Explained](https://www.youtube.com/watch?v=zxQyTK8quyY) – Une décomposition pas à pas du mécanisme.

---

> [!TIP]
✉️ **Mon conseil** : Ne vous laissez pas intimider par la complexité mathématique des Transformers dès le premier jour. Concentrez-vous sur l'**intuition** : pourquoi le modèle a-t-il besoin de "regarder" plusieurs mots à la fois ? 

> [!WARNING]
⚠️ **Attention :** si vous comprenez le problème du goulot d'étranglement des RNN, vous avez déjà fait 50% du chemin !

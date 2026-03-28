---
title: "Semaine 12 : Alignement et réglage par préférences"
weight: 2
bookCollapseSection : true
---


# 📝 Semaine 12 : Alignement et réglage par préférences

Bonjour à toutes et à tous ! Nous abordons aujourd'hui l'étape ultime de l'éducation d'une IA. La semaine dernière, nous avons appris à donner du savoir à notre modèle. Mais un modèle savant qui est impoli, biaisé ou dangereux est inutilisable en société.

> [!IMPORTANT]
🔑 **Je dois insister :** l'alignement est ce qui transforme un algorithme en un assistant digne de confiance. 

Aujourd'hui, nous allons apprendre à enseigner à la machine non pas ce qui est "vrai", mais ce qui est "mieux" selon le jugement humain. Bienvenue dans le monde de la morale algorithmique !


---
## 📖 Description générale
Cette semaine est dédiée à l'**Alignement des LLM**. Nous allons explorer comment passer d'un modèle qui suit des instructions (SFT) à un modèle qui reflète les valeurs et les préférences humaines. Nous décortiquerons le processus complexe du **RLHF** (*Reinforcement Learning from Human Feedback*) et l'entraînement des **Reward Models**. Nous étudierons ensuite la révolution **DPO** (*Direct Preference Optimization*), une méthode plus stable et élégante qui s'impose aujourd'hui comme le standard pour aligner les modèles sans passer par l'apprentissage par renforcement traditionnel.

---
## 🧠 Pré-requis importants
Pour bien saisir ces concepts, assurez-vous d'avoir compris :
1.  **Le Fine-tuning supervisé (Semaine 11)** : L'alignement commence toujours après une phase SFT réussie.
2.  **Les Log-Probabilités (Semaine 5)** : Comprendre comment le modèle score ses propres jetons.
3.  **L'intuition des fonctions de perte (Loss functions)** : Savoir comment on "pousse" un modèle vers un comportement souhaité.

**Ressources pour réviser les pré-requis :**
*   💻 **Hugging Face Course** : [Introduction to Fine-tuning](https://huggingface.co/learn/nlp-course/chapter3/1) .
*   📚 **StatQuest (YouTube)** : [Probability Distributions Explained](https://www.youtube.com/watch?v=oI3hZJqXJuc) .

---

## 📚 Ressources utiles pour les concepts de la semaine

### 🌐 Articles et Blogs de référence
*   **Hugging Face Blog** : [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf) – Le guide visuel de référence pour comprendre le cycle Feedback -> Reward -> Policy.
*   **Hugging Face Blog** : [Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl) – Une explication technique de pourquoi DPO est plus simple et efficace que le RLHF classique.
*   **Dave Bergmann (IBM)** : [LLM Alignment Explained](https://www.ibm.com/think/topics/llm-alignment) – Une article profond sur l'alignement des LLMs.


### 📺 Vidéos recommandées
*   🎥 **Andrej Karpathy** : [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) – (Regardez à partir de 13:00 pour la section sur l'alignement et le RLHF).
*   🎥 **Stanford Online** : [Guest lecture on DPO](https://www.youtube.com/watch?v=Q7rl8ovBWwQ) – Une présentation détaillée sur le DPO.

### 🛠️ Outils et Frameworks
*   **TRL Documentation** : [Direct Preference Optimization (DPO)](https://huggingface.co/docs/trl/main/en/dpo_trainer) – Guide pratique pour utiliser le DPOTrainer .


---
> [!TIP]
🔑 **Mon conseil** : Dans l'alignement, le plus dur n'est pas le code, c'est la donnée. 

> [!WARNING]
⚠️ **Attention :** si vos paires de préférences sont mal choisies, vous allez "brider" l'intelligence de votre modèle. 

> Un bon assistant doit être utile (*helpful*) mais aussi inoffensif (*harmless*). C'est cet équilibre fragile que nous allons apprendre à coder !
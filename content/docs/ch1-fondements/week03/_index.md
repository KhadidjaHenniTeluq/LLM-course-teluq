---
title: "Semaine 3 : Architecture Transformer approfondie"
weight: 3
bookCollapseSection: true
---


# 📝 Semaine 3 : Architecture Transformer approfondie

Bonjour à toutes et à tous ! Quel plaisir de vous retrouver pour cette semaine importante. Nous avons les briques (les tokens) et nous avons le ciment (les embeddings). Maintenant, mes chers étudiants, nous allons construire la cathédrale. 

> [!IMPORTANT]
📌 **Je dois insister :** aujourd'hui, nous ouvrons le "capot" du moteur de l'IA moderne. 

Nous allons décortiquer les engrenages mathématiques du Transformer. Ce n'est pas seulement du code, c'est une chorégraphie de matrices où chaque mot apprend à regarder tous les autres pour en saisir l'essence. Respirez, car nous plongeons au cœur de la machine !

---
## 📖 Description générale
Cette semaine est le pivot technique de notre cours. Nous quittons la surface pour explorer les entrailles du **Transformer**. 

Nous allons apprendre comment le mécanisme de **Self-Attention** transforme des vecteurs isolés en une pensée contextuelle grâce au trio magique : **Query, Key et Value**. Nous étudierons comment la machine gère l'ordre des mots via le **Positional Encoding** (notamment le moderne **RoPE**). Nous analyserons la structure d'un **bloc Transformer** complet, ses mécanismes de normalisation (**RMSNorm**) et ses optimisations de pointe comme **FlashAttention**. Enfin, nous verrons comment le **KV cache** permet à vos chatbots de répondre en temps réel sans s'essouffler.

---
## 🧠 Pré-requis importants
Pour maîtriser cette semaine complexe, vous devez avoir consolidé :
1.  **Algèbre Matricielle de base** : Comprendre qu'une multiplication de matrices est un moyen de projeter des données d'un espace à un autre.
2.  **Intuition des Embeddings (Semaine 2)** : Savoir que chaque mot est déjà représenté par un vecteur de nombres.
3.  **Softmax** : Comprendre que cette fonction transforme des scores bruts en probabilités (la somme de ces probabilités est égale à 1).

**Ressources pour réviser les pré-requis :**
*   🎥 **3Blue1Brown (YouTube)** : [Le calcul matriciel visuel](https://www.youtube.com/watch?v=fNk_zzaMoSs) .
*   📚 **Hugging Face Course** : [The Transformer Architecture](https://huggingface.co/learn/nlp-course/chapter1/4) (Une introduction douce avant notre plongée profonde).

---

## 📚 Ressources utiles pour les concepts de la semaine

### 🌐 Articles et Blogs de référence
*   **Jay Alammar** : [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) – L'explication visuelle la plus célèbre au monde pour comprendre les matrices Q, K, V.
*   **Lilian Weng (OpenAI)** : [Transformer Improvements](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) – Une excellente synthèse sur RoPE, SwiGLU et RMSNorm.
*   **Hugging Face Blog** : [FlashAttention, GQA, and more](https://huggingface.co/blog/optimize-llm) – Pour comprendre comment on rend l'attention plus rapide sur GPU.


### 📺 Vidéos recommandées
*   🎥 **StatQuest** : [Transformer Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY) – Une décomposition étape par étape, idéale pour ceux qui craignent les mathématiques complexes.
*   🎥 **CodeEmporium** : [Transformers Explained](https://www.youtube.com/watch?v=TQQlZhbC5ps) – Une vision plus conceptuelle et historique de l'architecture.

---

> [!TIP]
✉️ **Mon conseil** : Ne vous laissez pas intimider par les termes "Query" ou "Key". 

💡 voyez cela comme une conversation dans une bibliothèque. Le Transformer pose des questions (Query), regarde les étiquettes des livres (Key) et extrait le savoir (Value). Si vous gardez cette image en tête, les équations deviendront vos amies !
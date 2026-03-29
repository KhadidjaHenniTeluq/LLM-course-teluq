---
title: "Semaine 2 : Tokens, tokeniseurs et embeddings"
weight: 2
bookCollapseSection: true
---


# 📝 Semaine 2 : Tokens, tokeniseurs et embeddings

Bonjour à toutes et à tous ! Quel plaisir de vous retrouver. La semaine dernière, nous avons contemplé la forêt (l'histoire du NLP) ; aujourd'hui, nous sortons le microscope pour examiner les feuilles et les cellules : les **tokens** et les **embeddings**. 

> [!IMPORTANT]
📌 **Je dois insister :** c'est ici que se joue la "grammaire mathématique" des modèles. 

Si vous comprenez comment un mot devient un vecteur, vous comprenez comment une machine peut "ressentir" la proximité entre deux concepts. Respirez, nous allons transformer le texte brut en une symphonie de nombres !

---
## 📖 Description générale
Cette semaine est le cœur technique de la représentation du langage. Nous allons explorer comment les LLM découpent le texte via la **tokenisation** (mots, sous-mots, caractères, octets) et pourquoi ce choix impacte radicalement la performance (notamment pour le code et les langues non-anglo-saxonnes). Nous étudierons ensuite les **embeddings**, ces représentations vectorielles qui donnent du "sens" aux nombres. Nous reviendrons sur l'algorithme fondateur **Word2Vec** pour comprendre les embeddings statiques, avant d'introduire les **embeddings contextuels** qui font la force des modèles modernes. Enfin, nous verrons comment ces vecteurs permettent de créer des systèmes de recommandation concrets.

---

## 🧠 Pré-requis importants
Pour maîtriser cette semaine, vous devez être à l'aise avec :
1.  **Manipulation de chaînes en Python** : Utilisation des méthodes `.split()`, `.replace()` et des expressions régulières.
2.  **Intuition géométrique** : Comprendre qu'un point dans un espace peut être défini par plusieurs coordonnées.
3.  **Bases de l'Apprentissage Automatique** : La notion de "poids" (weights) que l'on ajuste pour minimiser une erreur.

**Ressources pour réviser les pré-requis :**
*   💻 **Documentation Python** : [String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods) .
*   📚 **Khan Academy** : [Points dans l'espace 3D](https://www.khanacademy.org/math/multivariable-calculus/thinking-about-multivariable-function/visualizing-scalar-valued-functions/v/representing-points-in-3d) - Pour l'intuition des dimensions .

---

## 📚 Ressources utiles pour les concepts de la semaine

### 🌐 Articles et Blogs de référence
*   **Hugging Face Documentation** : [Summary of the Tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) – Le guide ultime pour comparer BPE, WordPiece et SentencePiece.
*   **Jay Alammar** : [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) – Une explication visuelle magistrale sur la création des vecteurs de sens.
*   **Hugging Face Course (Chapter 2)** : [Tokenizers](https://huggingface.co/learn/nlp-course/chapter2/4) – Exercices interactifs sur le fonctionnement des bibliothèques modernes.

### 📺 Vidéos recommandées
*   🎥 **Computerphile** : [Word Embeddings & Word2Vec](https://www.youtube.com/watch?v=gQddtTdmG_8) – Une introduction claire à la raison pour laquelle nous utilisons des vecteurs.
*   🎥 **DataMListic** : [What is a Tokenizer?](https://www.youtube.com/watch?v=hL4ZnAWSyuU) – Vidéo courte et pédagogique sur la tokenisation.

### 🛠️ Outils interactifs
*   **OpenAI Tokenizer Tool** : [Tokenizer Playground](https://platform.openai.com/tokenizer) – Visualisez en temps réel comment GPT découpe vos phrases.


---
> [!TIP]
✉️ **Mon conseil** : Ne négligez pas la tokenisation en vous disant que c'est du "nettoyage". 

> [!WARNING]
⚠️ **Attention :** un mauvais tokeniseur peut rendre une IA totalement incapable de faire des calculs ou de comprendre le français correctement. Soyez des ingénieurs de la donnée, pas seulement des consommateurs de modèles !

---
title: "Semaine 2 : Tokens, tokeniseurs et embeddings"
weight: 2
bookCollapseSection: true
---


# 📝 Sommaire de la Semaine 2 : Tokens, tokeniseurs et embeddings

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Quel plaisir de vous retrouver. La semaine dernière, nous avons contemplé la forêt (l'histoire du NLP) ; aujourd'hui, nous sortons le microscope pour examiner les feuilles et les cellules : les **tokens** et les **embeddings**. 🔑 **Je dois insister :** c'est ici que se joue la "grammaire mathématique" des modèles. Si vous comprenez comment un mot devient un vecteur, vous comprenez comment une machine peut "ressentir" la proximité entre deux concepts. Respirez, nous allons transformer le texte brut en une symphonie de nombres ! » [SOURCE: Livre p.37]

---

## 📖 Description générale
Cette deuxième semaine est le cœur technique de la représentation du langage. Nous allons explorer comment les LLM découpent le texte via la **tokenisation** (mots, sous-mots, caractères, octets) et pourquoi ce choix impacte radicalement la performance (notamment pour le code et les langues non-anglo-saxonnes). Nous étudierons ensuite les **embeddings**, ces représentations vectorielles qui donnent du "sens" aux nombres. Nous reviendrons sur l'algorithme fondateur **Word2Vec** pour comprendre les embeddings statiques, avant d'introduire les **embeddings contextuels** qui font la force des modèles modernes. Enfin, nous verrons comment ces vecteurs permettent de créer des systèmes de recommandation concrets.

---

## 🧠 Pré-requis importants
Pour maîtriser cette semaine, vous devez être à l'aise avec :
1.  **Manipulation de chaînes en Python** : Utilisation des méthodes `.split()`, `.replace()` et des expressions régulières.
2.  **Intuition géométrique** : Comprendre qu'un point dans un espace peut être défini par plusieurs coordonnées.
3.  **Bases de l'Apprentissage Automatique** : La notion de "poids" (weights) que l'on ajuste pour minimiser une erreur.

**Ressources pour réviser les pré-requis (Vérifiées et fonctionnelles) :**
*   💻 **Documentation Python** : [String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods) (Vérifié : Fonctionnel).
*   📚 **Khan Academy** : [Points dans l'espace 3D](https://fr.khanacademy.org/math/multivariable-calculus/thinking-about-multivariable-function/visualizing-scalar-valued-functions/a/3d-coordinate-systems) (Pour l'intuition des dimensions - Vérifié : Fonctionnel).

---

## 📚 Ressources utiles pour les concepts de la semaine (Vérifiées le 29/03/2024)

### 🌐 Articles et Blogs de référence
*   **Hugging Face Documentation** : [Summary of the Tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) – Le guide ultime pour comparer BPE, WordPiece et SentencePiece. (Vérifié : Fonctionnel).
*   **Jay Alammar** : [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) – Une explication visuelle magistrale sur la création des vecteurs de sens. (Vérifié : Fonctionnel).
*   **Hugging Face Course (Chapter 2)** : [Tokenizers](https://huggingface.co/learn/nlp-course/chapter2/4) – Exercices interactifs sur le fonctionnement des bibliothèques modernes. (Vérifié : Fonctionnel).

### 📺 Vidéos recommandées (Travaillant uniquement)
*   🎥 **Computerphile** : [Word Embeddings & Word2Vec](https://www.youtube.com/watch?v=gQddtTbbGz8) – Une introduction claire à la raison pour laquelle nous utilisons des vecteurs. (Vérifié : Fonctionnel).
*   🎥 **Hugging Face** : [What is a Tokenizer?](https://www.youtube.com/watch?v=fNxaJsNG3-s) – Vidéo courte et pédagogique sur le découpage du texte. (Vérifié : Fonctionnel).

### 🛠️ Outils interactifs
*   **OpenAI Tokenizer Tool** : [Tokenizer Playground](https://platform.openai.com/tokenizer) – Visualisez en temps réel comment GPT découpe vos phrases. (Vérifié : Fonctionnel).

---

🔑 **Mon conseil** : « Ne négligez pas la tokenisation en vous disant que c'est du "nettoyage". ⚠️ **Attention :** un mauvais tokeniseur peut rendre une IA totalement incapable de faire des calculs ou de comprendre le français correctement. Soyez des ingénieurs de la donnée, pas seulement des consommateurs de modèles ! » [SOURCE: Livre p.55]

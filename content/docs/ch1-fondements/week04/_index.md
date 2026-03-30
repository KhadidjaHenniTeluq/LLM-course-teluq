---
title: "Semaine 4: Modèles de représentation (Encoder-only)"
weight: 4
bookCollapseSection: true
---


# 📝 Semaine 4 : Modèles de représentation (encoder-only)

Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour entamer notre deuxième grand chapitre : la Science des LLM. Jusqu'ici, nous avons beaucoup parlé de la structure.

Aujourd'hui, nous allons nous intéresser aux modèles qui "écoutent" et "comprennent" avec une précision chirurgicale. 

> [!IMPORTANT]
‼️ **Je dois insister :** si GPT est un écrivain, **BERT** est un analyste de texte hors pair. 

Nous allons apprendre à transformer une phrase en une signature mathématique si riche qu'elle permet de détecter une émotion ou une entité nommée en un clin d'œil. Préparez-vous à entrer dans l'ère de la représentation !

---

## 📖 Description générale
Cette semaine est dédiée aux modèles dits **Encoder-only**, dont le chef de file est **BERT**. Contrairement aux modèles qui prédisent la suite d'un texte, ces modèles sont conçus pour extraire le sens profond d'une séquence en lisant dans les deux sens simultanément (bidirectionnalité). 

Nous étudierons le rôle crucial du token **[CLS]**, le processus d'apprentissage par le "texte à trous" (**Masked Language Modeling - MLM**) et la diversité de la famille BERT (RoBERTa, DistilBERT, DeBERTa). Nous verrons enfin comment appliquer ces modèles à des tâches concrètes de classification et de reconnaissance d'entités (NER), même sans données étiquetées grâce au **Zero-shot**.

---
## 🧠 Pré-requis importants
Pour maîtriser cette semaine, assurez-vous d'avoir bien intégré :
1.  **Le mécanisme d'Attention (Semaine 3)** : Comprendre comment les mots se donnent mutuellement du contexte.
2.  **La Tokenisation (Semaine 2)** : Savoir comment BERT découpe les mots complexes avec `##`.
3.  **Python & Scikit-learn** : Être à l'aise avec la notion de "classifieur" et de "matrice".

**Ressources pour réviser les pré-requis :**
*   💻 **Hugging Face Course** : [The Transformer Hierarchy](https://huggingface.co/learn/nlp-course/chapter1/4) .
*   📚 **Scikit-learn documentation** : [Supervised Learning Intro](https://scikit-learn.org/stable/supervised_learning.html) .

---

## 📚 Ressources utiles pour les concepts de la semaine

### 🌐 Articles et Blogs de référence
*   **Jay Alammar** : [A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) – Une merveille de pédagogie visuelle sur le passage du texte à la classification.
*   **Hugging Face Blog** : [BERT 101: State of the Art NLP](https://huggingface.co/blog/bert-101) – Tout ce qu'il faut savoir sur l'architecture et les variantes.
*   **SBERT (Sentence-Transformers)** : [Documentation officielle](https://www.sbert.net/) – L'outil standard pour transformer BERT en extracteur d'embeddings de phrases.

### 📺 Vidéos recommandées
*   🎥 **Stanford Online** : [BERT and Other Pre-trained Language Models](https://www.youtube.com/watch?v=knTc-NQSjKA) .
*   🎥 **CodeEmporium** : [BERT Explained](https://www.youtube.com/watch?v=xI0HHN5XKDo) .

---

> [!TIP]
✉️ **Mon conseil** : BERT est un modèle de "représentation". ne lui demandez pas d'écrire une histoire, il n'est pas fait pour cela. 

> Utilisez-le pour transformer le chaos du langage en vecteurs organisés. C'est le socle de toute la recherche sémantique que nous verrons en Semaine 6 !

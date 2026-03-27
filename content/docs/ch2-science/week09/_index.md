---
title: "Semaine 9 : RAG (Retrieval-Augmented Generation)"
weight: 4
bookCollapseSection : true
---

# 📝 Semaine 9 : RAG (Retrieval-Augmented Generation)

Bonjour à toutes et à tous ! Nous entamons aujourd'hui une semaine capitale pour quiconque souhaite utiliser l'IA dans un cadre professionnel. Jusqu'ici, nous avons vu que les LLM sont des génies créatifs, mais parfois... ils inventent des faits avec une assurance déconcertante. C'est ce qu'on appelle les *hallucinations*. 

Aujourd'hui, nous allons donner à notre IA une "bibliothèque de référence" et lui apprendre à dire : "D'après ce document officiel, voici la réponse". Bienvenue dans le monde du **RAG**, le remède ultime contre l'oubli et le mensonge de la machine.

---

## 📖 Description générale
Cette semaine est consacrée à la **Génération Augmentée par Récupération** (*Retrieval-Augmented Generation* ou **RAG**). Nous allons apprendre à construire un système hybride qui combine la puissance de recherche sémantique (vue en Semaine 6) et la capacité de synthèse des modèles génératifs (vue en Semaine 5). Nous explorerons le pipeline complet : de l'indexation de vos propres documents à la génération de réponses "sourcées". Nous aborderons également des techniques avancées comme le **Reranking** et le **Query Rewriting**, ainsi que les métriques modernes pour évaluer la fidélité des réponses (framework **Ragas**).

---

## 🧠 Pré-requis importants
Pour maîtriser le RAG, vous devez avoir solidement acquis :
1.  **Recherche Sémantique (Semaine 6)** : Savoir utiliser des embeddings et une base de données vectorielle (comme FAISS).
2.  **Prompt Engineering (Semaine 8)** : Savoir structurer un prompt pour forcer le modèle à utiliser un contexte donné.
3.  **Bases de Python & LangChain** : Une familiarité avec l'orchestration de composants IA simplifiera grandement votre apprentissage.

**Ressources pour réviser les pré-requis :**
*   💻 **Hugging Face Course** : [NLP Course Chapter 1](https://huggingface.co/learn/nlp-course/chapter1/1) .
*   📚 **Pinecone Learning** : [What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)

---

## 📚 Ressources utiles pour les concepts de la semaine

### 🌐 Documentations et Articles de référence (Vérifiés)
*   **Hugging Face Blog** : [Retrieval Augmented Generation (RAG) Explained](https://huggingface.co/blog/ray-rag) – Un guide technique sur l'intégration de la récupération.
*   **Lilian Weng (OpenAI)** : [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) – (Voir la section "Memory" et "Tool Use" pour l'intuition du RAG).
*   **Ragas Documentation** : [Ragas: Evaluation framework for RAG](https://docs.ragas.io/en/stable/) – La référence pour mesurer la "Faithfulness" de vos systèmes.

### 🛠️ Outils et Tutoriels
*   **LangChain Documentation** : [Q&A with RAG](https://python.langchain.com/docs/use_cases/question_answering/) – Tutoriel pas à pas pour construire votre premier chatbot sur documents.
*   **Cohere Blog** : [Reranking for Better RAG](https://txt.cohere.com/rerank/) – Pourquoi le top-K ne suffit pas toujours.

### 📺 Vidéos recommandées (Vérifiées)
*   🎥 **FreeCodeCamp** : [Learn RAG from Scratch](https://youtu.be/sVcwVQRHIc8?si=Nm7RlL6deLFAB8Kt).
*   🎥 **DataBricks** : [Building Advanced RAG Over Complex Documents](https://www.youtube.com/watch?v=dI_TmTW9S4c).

---

> [!TIP]
🔑 **Mon conseil** : Le RAG, c'est l'examen "livre ouvert" de l'IA. Ne demandez plus à l'IA de réciter par cœur ; donnez-lui les bons livres et apprenez-lui à chercher aux bons chapitres. 

> [!WARNING]
⚠️ **Attention :** un mauvais moteur de recherche donnera de mauvais documents à l'IA, et l'IA produira une réponse fausse, mais très polie. La qualité de votre recherche est le cerveau de votre RAG !

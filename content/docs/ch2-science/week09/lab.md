---
title: "Laboratoire "
weight: 6
---

Bonjour à toutes et à tous ! Nous passons maintenant aux travaux pratiques de notre semaine sur le RAG. C'est ici que vous allez donner une "mémoire documentaire" à votre IA. 
> [!NOTE]
🔑 **Je dois insister :** ne voyez pas le RAG comme une simple recherche Google améliorée. 

C'est une architecture de confiance. Nous allons apprendre à transformer un modèle qui "devine" en un expert qui "prouve". Soyez méticuleux dans votre découpage de texte, car c'est là que se joue la pertinence de votre futur système. Prêt·e·s à construire votre premier assistant ancré dans la réalité ?


## 🔹 EXERCICE 1 : Pipeline RAG de base avec LangChain (Niveau 1)

**Objectif** : Implémenter un flux complet : Ingestion -> Indexation -> Question/Réponse ancrée.

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch

# 1. PRÉPARATION DU SAVOIR (QUESTION CODE)
texts = [
    "The internal project 'Aegis' was started in 2023 to improve cloud security.",
    "Project 'Aegis' team is led by Dr. Sarah Chen.",
    "The budget for Aegis is 2 million euros."
]

# --- VOTRE CODE ICI ---
```

<details>
<summary><b>Voir la réponse</b></summary>

```python
# 2. Création de la mémoire vectorielle
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(texts, embeddings)

# 3. Initialisation du LLM (On utilise TinyLlama pour Colab T4)
hf_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                   device=0, torch_dtype=torch.bfloat16, max_new_tokens=50)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# 4. Assemblage de la chaîne QA
rag = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_db.as_retriever()
)

# 5. Test
query = "Who leads the Aegis project?"
response = rag.invoke(query)
print(f"Question: {query}\nRéponse: {response['result']}")
```

**EXPLICATIONS DÉTAILLÉES** :
*   **ATTENDU** : Le modèle doit citer "Dr. Sarah Chen".
*   **JUSTIFICATION** : Sans le RAG, TinyLlama ne peut pas connaître ce nom fictif. FAISS a récupéré le bon chunk et l'a injecté dans le prompt de l'IA.

</details>

---

## 🔹 EXERCICE 2 : Reranking pour la précision (Niveau 2)

**Objectif** : Ajouter une étape de reranking pour filtrer les résultats d'une recherche vectorielle.

```python
from sentence_transformers import CrossEncoder

# 1. Résultats bruts du Bibliothécaire
query = "What is the budget?"
candidates = [
    "Project Aegis is led by Dr. Chen.", # Peu pertinent
    "The 2 million euro budget was approved last week.", # Très pertinent
    "Aegis aims at cloud security improvement." # Moyen
]

# --- VOTRE CODE ICI ---
```

<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- RÉPONSE ---
# 2. Chargement du Reranker (Cross-Encoder)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

# 3. Calcul des scores d'interaction fine
pairs = [[query, doc] for doc in candidates]
scores = reranker.predict(pairs)

# 4. Affichage trié
results = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

print("--- CLASSEMENT APRÈS RERANKING ---")
for score, doc in results:
    print(f"Score: {score:.2f} | Doc: {doc}")
```

**EXPLICATIONS DÉTAILLÉES**
*   **ATTENDU** : Le document sur les "2 million euro" doit avoir le score le plus haut.
*   **JUSTIFICATION** : Le Cross-Encoder "lit" simultanément la question et chaque doc, capturant des liens sémantiques que la recherche vectorielle simple peut rater.

</details>

---

## 🔹 EXERCICE 3 : Audit de fidélité avec Ragas (Niveau 3)

**Objectif** : Simuler une évaluation de type "Faithfulness" pour détecter une hallucination.

**Scénario** : 
*   **Contexte** : "L'entreprise a été fondée en 1995."
*   **Réponse de l'IA** : "L'entreprise a été fondée en 1998."

**Tâche** : Expliquez comment un LLM-as-a-judge (comme Ragas) parviendrait à noter cette réponse.

<details>
<summary><b>Voir la réponse</b></summary>

**Réponse détaillée et Justification** :
1.  **Étape 1 (Décomposition)** : Le juge extrait l'affirmation : "Date de fondation = 1998".
2.  **Étape 2 (Vérification)** : Le juge compare à la preuve : "Date de fondation = 1995".
3.  **Étape 3 (Verdict)** : L'affirmation est contredite ou non-prouvée. 
4.  **Résultat** : Score de **Faithfulness = 0.0**. 

> [!NOTE]
🔑 **Note** : C'est ainsi que nous protégeons nos utilisateurs. Même si la phrase est jolie, le score de fidélité dénonce le mensonge.

</details>


---

**Mots-clés de la semaine** : RAG, Retrieval, Grounding, Indexation, Vector DB, FAISS, Cross-Encoder, Reranking, Faithfulness, LangChain.

**En prévision de la semaine suivante** : Nous allons sortir du monde du texte pur. Comment une IA peut-elle "voir" une image et en discuter avec vous ? Bienvenue dans le monde fascinant des **LLM Multimodaux**.

**SOURCES COMPLÈTES** :
*   Framework Ragas : https://docs.ragas.io/
*   LangChain RAG Guide : https://python.langchain.com/docs/use_cases/question_answering/
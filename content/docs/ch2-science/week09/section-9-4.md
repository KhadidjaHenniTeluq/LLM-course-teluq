---
title: "9.4 Implémentation avec LangChain"
weight: 5
---


## Le chef d'orchestre de vos applications IA
Bonjour à toutes et à tous ! Nous voici arrivés au terme de cette semaine marathon sur le RAG. Nous avons vu la théorie, la recherche de précision et l'audit rigoureux des résultats. Mais maintenant, une question pratique se pose : comment assembler toutes ces pièces mobiles — le tokeniseur, le modèle d'embedding, la base de données FAISS, le LLM et le prompt — sans que votre code ne devienne un plat de spaghettis illisible ? 

> [!IMPORTANT]
🔑 **Je dois insister :** en production, la modularité est votre meilleure alliée. 

Aujourd'hui, je vais vous présenter votre futur meilleur ami : **LangChain**. C'est le chef d'orchestre qui va donner le tempo et lier tous vos composants dans une symphonie fluide et robuste. Prêt·e·s à passer à l'échelle industrielle ?

---
## Philosophie et Architecture de LangChain
Pour comprendre LangChain, il faut regarder son plan d'architecte. Regardez attentivement la **Figure 9-15 : Architecture de LangChain** . Cette illustration est la carte de votre écosystème de développement.

{{< bookfig src="155.png" week="09" >}}

Décortiquons ensemble les modules affichés dans cette figure :
1.  **Model I/O (Entrées/Sorties)** : C'est la gestion des prompts, des modèles et des "output parsers" (pour transformer le texte de l'IA en données utilisables).
2.  **Retrieval (Récupération)** : C'est le cœur de notre RAG. Ce module gère les "Document Loaders", les découpeurs de texte (Splitters), les modèles d'embedding et les interfaces avec les bases de données vectorielles.
3.  **Chains (Chaînes)** : C'est le concept central. Une chaîne est un assemblage de briques. La sortie d'un composant devient l'entrée du suivant.
4.  **Memory (Mémoire)** : Pour que votre chatbot se souvienne des questions précédentes de l'utilisateur.
5.  **Agents** : Le niveau supérieur, où l'IA décide elle-même d'utiliser tel ou tel outil pour répondre.

> [!NOTE]
🔑 **Mon intuition :** LangChain ne remplace pas les modèles, il les entoure d'une infrastructure standardisée. 

> C'est comme le système de plomberie d'une maison : peu importe la marque de votre robinet (GPT-4, Claude ou Llama), les tuyaux et les connecteurs restent les mêmes.

---
## La brique "Document" : Au-delà du simple texte
Dans LangChain, nous ne manipulons pas de simples chaînes de caractères `str`, mais des objets `Document`. 

> [!IMPORTANT]
🔑 **Je dois insister sur cette distinction technique :** un objet `Document` contient :
*   `page_content` : Le texte brut.
*   `metadata` : Un dictionnaire contenant la source (ex: "page 12 du PDF"), la date ou l'auteur.

Pourquoi est-ce capital pour le RAG ? Parce que sans métadonnées, votre IA ne pourra jamais faire de **citations** (vues en [**section 9.1**]({{< relref "section-9-1.md" >}}#RAG-citation)). 

En [**Figure 9-5**]({{< relref "section-9-1.md" >}}#RAG-citation) , nous avons vu l'importance de citer ses sources ; LangChain automatise ce suivi du texte à travers tout le pipeline, de l'indexation à la génération.

---
## Les Vector Stores dans LangChain : Une interface universelle
L'un des plus grands avantages de LangChain est l'abstraction des bases de données vectorielles. Que vous utilisiez FAISS ([**local, Semaine 6**]({{< relref "section-6-3.md" >}}#faiss)), ChromaDB ou Pinecone (cloud), le code de votre application reste quasiment identique.

Le processus d'intégration est standardisé :
1.  On initialise un `EmbeddingsModel`.
2.  On passe cet objet à un `VectorStore`.
3.  On transforme ce magasin de vecteurs en un `Retriever`. 

> [!NOTE]
🔑 **Le concept du Retriever :** Ce n'est pas une base de données, c'est une *interface*. 

> Un Retriever est un objet qui prend une question en entrée et renvoie une liste de documents. Cette abstraction permet de changer de moteur de recherche (sémantique, hybride ou même via une API comme Google Search) sans modifier le reste de votre chaîne RAG.

---
## Orchestration avec RetrievalQA
Le composant historique pour bâtir un RAG avec LangChain est la chaîne `RetrievalQA`. Elle automatise ce que nous avons fait manuellement en section 9.1 :
1.  Prendre la question.
2.  Interroger le Retriever.
3.  Formater le prompt avec le contexte trouvé.
4.  Appeler le LLM.
5.  Rendre la réponse.

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Il existe plusieurs types de chaînes pour assembler les documents (les "chain types"). Le plus courant est le mode **"stuff"**.
*   **"stuff"** : On "fourre" (stuff) tous les documents trouvés dans le prompt. 
>> [!CAUTION]
🔑 **Le risque :** Si vous avez trop de documents, vous allez dépasser la "Context Window" du modèle (Semaine 5) et provoquer une erreur.
*   **"map_reduce"** : Le modèle résume chaque document séparément, puis fait une synthèse globale. C'est plus lent mais idéal pour les très gros volumes.

---
## Laboratoire de code : Un pipeline RAG complet avec LangChain (Colab T4)
Nous allons maintenant assembler tous les concepts de la semaine dans une implémentation industrielle propre. Nous utiliserons des modèles locaux pour préserver votre budget et votre vie privée.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install langchain langchain-community sentence-transformers faiss-gpu transformers accelerate

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch

# 1. PRÉPARATION DES DONNÉES (Indexation)
raw_docs = [
    "Le guide de sécurité stipule que le port du casque est obligatoire sur le chantier Alpha.",
    "La cafétéria est ouverte de 11h30 à 14h00 pour tous les employés munis d'un badge.",
    "En cas d'incendie, dirigez-vous vers le point de rassemblement situé sur le parking Nord."
]
# Simulation de documents LangChain
from langchain.docstore.document import Document
documents = [Document(page_content=t, metadata={"source": "Manuel Employé"}) for t in raw_docs]

# 2. CHUNKING INTELLIGENT
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# 3. EMBEDDINGS ET VECTOR STORE (Le Bibliothécaire)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)

# 4. CHARGEMENT DU LLM (L'Écrivain)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
hf_pipe = pipeline("text-generation", model=model_id, device=0, torch_dtype=torch.bfloat16, max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# 5. ASSEMBLAGE DE LA CHAÎNE RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), # On récupère les 2 meilleurs
    return_source_documents=True # Crucial pour la transparence
)

# 6. UTILISATION
query = "Où dois-je aller si l'alarme incendie sonne ?"
result = rag_chain.invoke({"query": query})

print(f"Question : {query}")
print(f"Réponse : {result['result']}")
print(f"Source utilisée : {result['source_documents'][0].metadata['source']}")
```

---
## Optimisations et Bonnes Pratiques Industrielles

> [!IMPORTANT]
⚠️ Ne vous arrêtez pas au code qui "marche". Cherchez le code qui "tient". 

Pour un déploiement sérieux, plusieurs optimisations sont proposées :
1.  **La gestion de la latence** : Si votre LLM met 10 secondes à répondre, l'utilisateur partira. Utilisez le **Streaming** (l'affichage du texte mot par mot) pour donner une impression de vitesse.
2.  **Le cache de requêtes** : Si deux utilisateurs posent la même question, ne relancez pas tout le calcul. Utilisez un cache vectoriel pour renvoyer la réponse déjà générée.
3.  **Filtrage par métadonnées** : Si l'utilisateur cherche un document de 2024, ne demandez pas au bibliothécaire de chercher dans les archives de 1990. Filtrez d'abord par date via les métadonnées pour réduire le bruit sémantique.

---
## Éthique et Responsabilité : Le danger de la "Boîte Noire" orchestrée

> [!CAUTION]
⚠️ Mes chers étudiants, plus l'architecture est complexe, plus il est facile de cacher des erreurs.

Lorsque vous utilisez un framework comme LangChain :
1.  **Le risque de fuite de données** : Si votre orchestrateur envoie par erreur l'historique complet d'un client à une API tierce pour faire un résumé, vous enfreignez le RGPD. Auditez chaque étape du flux de données.
2.  **Le Vendor Lock-in** : Dépendre trop lourdement d'un seul framework peut vous rendre captif si celui-ci change ses licences ou ses tarifs. Gardez un code suffisamment modulaire pour pouvoir changer de moteur.
> 3.  **L'illusion de la citation** : LangChain peut vous dire qu'il a utilisé le "Document A", mais le LLM a peut-être quand même halluciné une réponse basée sur sa propre mémoire interne. 
>> [!TIP] 
🔑 **Mon conseil** : Utilisez toujours un vérificateur de fidélité (comme Ragas, section 9.3) à la sortie de vos chaînes LangChain.

> [!NOTE]
🔑 **Mon message final pour cette semaine** : Vous avez maintenant entre les mains le plan complet d'un système d'intelligence artificielle factuel et professionnel. Vous avez appris à transformer le délire statistique en rigueur documentaire. Le RAG est sans doute la compétence la plus demandée sur le marché du travail aujourd'hui. Soyez-en fiers, mais restez humbles devant la complexité du langage.

---
Nous avons terminé notre grande semaine sur le RAG ! Vous savez désormais concevoir, construire, évaluer et orchestrer un système de génération ancré dans la réalité. La semaine prochaine, nous allons repousser les frontières : pourquoi se contenter du texte quand on peut parler aux images ? Bienvenue dans le monde fascinant des **LLM Multimodaux**. Mais avant cela, place au laboratoire final !
---
title: "9.1 Architecture RAG fondamentale"
weight: 2
---

## Le remède au "Syndrome du Perroquet Savant"
Mes chers étudiants, commençons par un constat d'humilité : un LLM, aussi grand soit-il, ne sait que ce qu'il a lu pendant son entraînement. 

Si vous lui demandez le chiffre d'affaires de votre entreprise pour le trimestre dernier, il échouera ou, pire, il inventera un chiffre crédible (hallucination). Pourquoi ? Parce que cette information n'était pas dans son dictionnaire d'origine. 

Le **RAG (Retrieval-Augmented Generation)**, ou Génération Augmentée par Récupération, est la réponse à ce problème. L'idée est de ne plus compter sur la seule mémoire interne du modèle (ses poids), mais de lui donner accès à une base de connaissances externe. 

Regardez la **Figure 9-1 : Un pipeline RAG basique**. Cette figure est le plan de montage de votre futur système. Elle montre un flux circulaire où la question de l'utilisateur ne va pas directement au LLM, mais passe d'abord par une étape de "Retrieval" (Récupération) dans une source de données. L'IA reçoit ensuite la question ET les documents trouvés pour formuler sa réponse. 

{{< bookfig src="195.png" week="09" >}}

> [!NOTE]
🔑 **Notez bien cette intuition :** le LLM ne répond plus de mémoire, il répond par synthèse de documents.


## L'analogie du bibliothécaire et de l'écrivain
Pour bien comprendre les rôles, utilisons une analogie. Un système RAG, c'est l'alliance de deux personnages :
1.  **Le Bibliothécaire (Retriever)** : Il est rapide, il connaît l'emplacement de chaque livre (vecteur), mais il ne sait pas forcément rédiger un essai. Son rôle est de vous apporter les 3 meilleurs livres sur un sujet.
2.  **L'Écrivain (Generator)** : C'est le LLM. Il a un style impeccable et sait synthétiser des informations complexes. Mais il a une mémoire de poisson rouge pour les détails précis. 

Dans le RAG, le Bibliothécaire court chercher les preuves, les pose sur le bureau de l'Écrivain, et ce dernier rédige la réponse finale en se basant uniquement sur ce qui est posé devant lui.


## Étape 1 : L'Indexation (La préparation de la bibliothèque)
Avant de pouvoir chercher, il faut ranger. Cette phase se passe hors-ligne, avant même que l'utilisateur ne pose sa première question. Comme nous l'avons vu en [**Semaine 6**]({{< relref "section-6-3.md" >}}#indexing), elle suit un processus rigoureux :
*   **Le Chargement** : Extraction du texte des PDF, des bases SQL ou des sites web.
*   **Le Chunking** : Découpage en morceaux (nous avons vu l'importance de l'overlap en [**6.3**]({{< relref "section-6-3.md" >}}#overlap)).
*   **L'Embedding** : Transformation de chaque morceau en un vecteur sémantique.
*   **Le Stockage** : Enregistrement dans une base de données vectorielle (Vector DB).

> [!IMPORTANT]
🔑 **Je dois insister :** si votre indexation est bâclée (morceaux trop petits ou trop gros), votre "bibliothécaire" rapportera les mauvaises pages, et votre système RAG sera un échec, peu importe la puissance du LLM utilisé.


## Étape 2 : La Récupération (Le Retrieval)
C'est le moment où l'utilisateur pose sa question. Regardez la **Figure 9-2 : Dense retrieval**. 
1.  La question est transformée en vecteur (embedding).
2.  Le système effectue une recherche de plus proches voisins (KNN) dans la base vectorielle.
3.  On extrait les **Top-K** documents (souvent les 3 à 5 morceaux les plus similaires).

{{< bookfig src="172.png" week="09" >}}

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** La similarité sémantique ne garantit pas la pertinence. Le modèle peut trouver un document qui utilise les mêmes mots mais qui ne répond pas à la question. C'est pour cela que nous introduisons parfois une étape de **Reranking**, illustrée en **Figure 9-3** . On utilise un second modèle, plus lent mais plus précis (Cross-Encoder), pour ré-ordonner les résultats et s'assurer que le meilleur document est bien en première position avant de l'envoyer au LLM.

{{< bookfig src="173.png" week="09" >}}

<a id="aug-prompt"></a>

## Étape 3 : La Génération Ancrée (Grounded Generation)
C'est ici que la magie du LLM opère enfin. Nous construisons ce que l'on appelle un **Augmented Prompt** (Prompt augmenté). 

Observez la **Figure 9-4 : Recherche sémantique pour la génération** . Le système crée dynamiquement un prompt géant qui ressemble à ceci :
> "Tu es un assistant expert. Utilise uniquement les informations fournies dans le contexte ci-dessous pour répondre à la question. Si la réponse n'est pas dans le texte, dis que tu ne sais pas.
> 
> CONTEXTE : [Document 1] [Document 2] ...
> 
> QUESTION : {Question de l'utilisateur}"

{{< bookfig src="197.png" week="09" >}}

<a id="RAG-citation"></a>

> [!NOTE]
🔑 **C'est le concept de Grounded Generation :** Le modèle est "ancré" dans le texte. En **Figure 9-5** , vous voyez le résultat idéal : une réponse fluide qui inclut des **citations** (ex: [1], [2]). Cela permet à l'humain de cliquer sur la source pour vérifier que l'IA ne délire pas.

{{< bookfig src="174.png" week="09" >}}

## Pourquoi le RAG est-il vital pour l'IA responsable ?

> [!WARNING]
⚠️ Mes chers étudiants, le RAG n'est pas qu'une amélioration technique, c'est une exigence éthique.
1.  **Réduction des hallucinations** : En forçant le modèle à citer ses sources, on l'empêche de s'égarer dans ses probabilités statistiques.
2.  **Actualisation des connaissances** : Pas besoin de réentraîner GPT-4 (ce qui coûte des millions) pour lui apprendre une nouvelle règle fiscale. Il suffit de mettre à jour le document PDF dans la base vectorielle.
3.  **Confidentialité et Contrôle** : Vous pouvez restreindre l'IA à ne lire que les documents auxquels l'utilisateur a droit. Le LLM ne "connaît" rien, il ne fait que synthétiser ce qu'on lui prête.


## Comparaison : LLM "Base" vs RAG

| Caractéristique | LLM Seul (Mémoire interne) | RAG (Mémoire externe) |
| :--- | :--- | :--- |
| **Connaissances** | Figées à la date d'entraînement | Toujours à jour (temps réel) |
| **Source de vérité** | Statistique ("Vraisemblance") | Documentaire ("Preuve") |
| **Hallucinations** | Fréquentes et difficiles à détecter | Rares et vérifiables via citations |
| **Coût d'évolution** | Immense (Fine-tuning lourd) | Faible (Mise à jour de l'index) |


## Laboratoire de code : Simulation d'un flux RAG (Colab T4)
Voici comment orchestrer ce processus. Nous allons simuler une base de connaissances technique et demander à un modèle léger (TinyLlama ou Phi-3) de répondre en utilisant uniquement ces données.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers torch

from transformers import pipeline
import torch

# 1. NOTRE "BIBLIOTHÈQUE" (Simulation d'une Vector DB)
kb_documents = [
    "Le protocole interne de l'entreprise exige que tout mot de passe fasse au moins 12 caractères.",
    "La procédure de remboursement des frais de voyage doit être soumise avant le 5 du mois suivant.",
    "L'accès au parking souterrain est réservé aux véhicules électriques le vendredi."
]

# 2. CHARGEMENT DU MODÈLE GÉNÉRATEUR (L'Écrivain)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
generator = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 3. FONCTION DE GÉNÉRATION ANCRÉE (RAG Logic)
def rag_answer(user_query, documents):
    # Simulation du Retrieval (On prend tous les docs pour l'exemple)
    context = "\n".join([f"- {doc}" for doc in documents])
    
    # Construction du Prompt Augmenté (Section 9.1 intuition)
    prompt = f"""<|system|>
    Utilise EXCLUSIVEMENT les informations suivantes pour répondre. 
    Si la réponse n'est pas dedans, dis 'Je n'ai pas cette information'.
    CONTEXTE :
    {context}</s>
    <|user|>
    {user_query}</s>
    <|assistant|>"""

    output = generator(prompt, max_new_tokens=100, do_sample=False)
    return output[0]['generated_text'].split("<|assistant|>")[-1].strip()

# 4. TEST DU SYSTÈME
query = "Quelle est la règle pour les mots de passe ?"
print(f"Question : {query}")
print(f"IA (RAG) : {rag_answer(query, kb_documents)}")

query_out = "Comment s'appelle le PDG ?"
print(f"\nQuestion : {query_out}")
print(f"IA (RAG) : {rag_answer(query_out, kb_documents)}")
```
> [!WARNING]
⚠️ **Avertissement** : Observez la seconde réponse. Si votre prompt est bien écrit, le modèle doit refuser de répondre ("Je n'ai pas cette information") au lieu d'inventer un nom de PDG. C'est la preuve que votre garde-fou fonctionne !


## Éthique et Transparence : La responsabilité des sources
> [!CAUTION]
⚠️ Mes chers étudiants, un RAG n'est pas plus intelligent que les documents qu'il lit.

Si votre base de données contient des documents obsolètes, faux ou haineux, le LLM les synthétisera avec la même autorité que s'il s'agissait de la vérité absolue. 
1.  **La pollution des sources** : Le RAG peut devenir un outil de désinformation massif si la base de connaissances n'est pas auditée. 
2.  **L'illusion de la citation** : Parfois, le modèle cite une source ([1]) qui ne contient pas du tout l'information mentionnée. C'est l'hallucination de second niveau. 

> [!IMPORTANT]
🔑 **À retenir** : Dans un système RAG, la qualité de la donnée **est** la qualité du modèle. Un LLM brillant nourri de sources obsolètes ou erronées produira des erreurs sophistiquées — et rien n'est plus dangereux qu'une erreur bien rédigée. Votre rôle d'ingénieur ne s'arrête pas à l'architecture technique : il commence par la rigueur éditoriale de votre base documentaire.

---
Vous avez maintenant les clés de l'architecture fondamentale du RAG. Vous savez pourquoi nous l'utilisons et comment il se structure. Dans la section suivante ➡️, nous allons corser le jeu : que faire si la question de l'utilisateur est mal posée ? Comment chercher dans plusieurs documents à la fois ? Bienvenue dans les techniques de récupération avancées.
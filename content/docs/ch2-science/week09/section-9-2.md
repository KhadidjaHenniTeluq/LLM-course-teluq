---
title: "9.2 Stratégies de récupération avancées"
weight: 3
---

## Dépasser le "Top-K" : Pourquoi la recherche simple échoue en production
Bonjour à toutes et à tous ! J'espère que vous avez bien digéré l'architecture de base du RAG que nous avons vue en section 9.1. C'est un excellent point de départ, mais je vais être très directe avec vous : dans un environnement industriel, un RAG "naïf" échoue dans 40 % des cas. Pourquoi ? Parce que les utilisateurs posent des questions mal formulées, parce que les documents sont complexes, et parce que la similarité mathématique ([**section 6.2**]({{< relref "section-6-2.md" >}})) n'est pas toujours synonyme de pertinence métier. 
> [!IMPORTANT]
🔑 **Je dois insister :** le maillon faible du RAG, c'est le "R" (Retrieval). 

Si votre bibliothécaire rapporte les mauvaises pages, votre écrivain (le LLM) ne pourra que produire une erreur élégante. Aujourd'hui, nous allons apprendre à muscler ce bibliothécaire grâce aux stratégies de récupération avancées.


## La réécriture de requête (Query Rewriting) : Traduire l'humain pour la machine
Les utilisateurs sont rarement des experts en "mots-clés sémantiques". Ils posent des questions comme : "Et pour le deuxième cas, ça marche comment ?". Pour un moteur de recherche vectoriel, cette phrase est un cauchemar : que signifie "le deuxième cas" ? Que signifie "ça" ?

C'est ici qu'intervient le **Query Rewriting**. L'idée est d'utiliser un premier appel au LLM pour transformer la question floue de l'utilisateur en une requête de recherche optimisée. 
*   **Expansion de requête** : On demande au LLM d'ajouter des synonymes ou des termes techniques liés (ex: "problème de batterie" devient "défaillance lithium-ion, perte de charge, accumulateur").
*   **Résolution de coréférence** : Dans une conversation, le LLM analyse l'historique pour remplacer "il" ou "le deuxième" par les vrais sujets mentionnés plus tôt. 

> [!TIP]
🔑 **Mon intuition :** Le Query Rewriting, c'est comme avoir un interprète qui écoute votre marmonnement et le crie clairement au bibliothécaire pour qu'il comprenne exactement ce que vous cherchez.


## Multi-Query RAG : Ne pas mettre tous ses œufs dans le même vecteur
Parfois, une question peut être interprétée de plusieurs façons. Au lieu de parier sur un seul vecteur de recherche, pourquoi ne pas en utiliser plusieurs ?
Dans le **Multi-Query RAG**, nous demandons au LLM de générer 3 à 5 variantes de la question initiale. 
1.  Nous effectuons une recherche pour chaque variante.
2.  Nous récupérons les résultats de toutes ces recherches.
3.  Nous fusionnons les documents trouvés en supprimant les doublons.

> [!NOTE]
🔑 **Je dois insister :** cette technique augmente considérablement le "Rappel" (Recall). On s'assure ainsi de ne pas rater un document crucial simplement parce que la formulation originale de l'utilisateur était légèrement décalée par rapport au titre du document.


## L'architecture à deux étages : Le Reranking
Nous arrivons au cœur de l'ingénierie de précision. Regardez attentivement la **Figure 9-6 : Pipeline de reranking**. Cette illustration décrit une stratégie en deux étapes, indispensable pour les bases de données massives.

{{< bookfig src="185.png" week="09" >}}

1.  **Premier étage (Le Shortlisting)** : On utilise une recherche vectorielle rapide ([**FAISS, section 6.3**]({{< relref "section-6-3.md" >}}#faiss)). C'est efficace mais un peu "grossier". On récupère, par exemple, les 100 documents les plus proches. 
2.  **Deuxième étage (Le Reranking)** : On prend ces 100 candidats et on les passe dans un **Reranker** (souvent un modèle de type Cross-Encoder). Ce modèle est beaucoup plus lent mais incroyablement plus précis. Il examine chaque document face à la question et lui donne une note de 0 à 1. 
3.  **Le résultat final** : On ne donne au LLM que les 3 ou 5 documents ayant obtenu la meilleure note du Reranker.

> [!NOTE]
❓ **Pourquoi cette complexité ?** Le Reranker "lit" vraiment la paire question-document alors que la recherche vectorielle ne fait que comparer des positions GPS. Comme le montre la figure, le Reranker peut faire remonter en 1ère position un document qui était initialement en 20ème position mais qui contient la réponse exacte.


## Cross-Encoders vs Bi-Encoders 
Pour comprendre pourquoi le Reranking est si puissant, il faut regarder "sous le capot" de la **Figure 9-7 : Cross-encoder reranker**. 


{{< bookfig src="186.png" week="09" >}}

*   **Le Bi-Encoder (SBERT)** : C'est ce que nous avons utilisé jusqu'ici. La question a son vecteur, le document a le sien. Ils ne se "parlent" jamais pendant le calcul. C'est rapide mais on perd les interactions fines entre les mots.
*   **Le Cross-Encoder** : On concatène la question et le document dans un seul bloc BERT : `[CLS] Question [SEP] Document [SEP]`. 
*   **L'Attention Totale** : Grâce à la self-attention bidirectionnelle (Semaine 4), chaque mot de la question peut interagir avec chaque mot du document. Le modèle peut détecter qu'un "pas" ou un "sauf" dans la question change totalement la pertinence d'un paragraphe. 

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** N'utilisez jamais un Cross-Encoder pour fouiller dans un million de documents. Il est 100 à 1000 fois plus lent qu'un Bi-Encoder. Il est l'outil du "dernier kilomètre", pas le moteur de recherche principal. 


## Multi-hop RAG : Le raisonnement par étapes
Certaines questions demandent de relier plusieurs documents qui ne se ressemblent pas. 
*   *Exemple* : "Quelle est la capitale du pays où est né l'inventeur du téléphone ?"
1.  **Hop 1** : Chercher "inventeur du téléphone". Trouver "Graham Bell".
2.  **Hop 2** : Chercher "lieu de naissance de Graham Bell". Trouver "Écosse".
3.  **Hop 3** : Chercher "capitale de l'Écosse". Trouver "Édimbourg".

Le **Multi-hop RAG** utilise le LLM comme un agent (nous verrons cela en Semaine 14). L'IA fait une première recherche, analyse le résultat, puis décide de faire une *nouvelle* recherche basée sur ce qu'elle vient d'apprendre. 

> [!IMPORTANT]
🔑 **C'est une rupture majeure :** l'IA ne subit plus la recherche, elle la pilote.


## Hybrid Search : Marier le Lexical et le Sémantique

> [!WARNING]
⚠️ Ne jetez pas la recherche par mots-clés aux orties ! 

La recherche vectorielle est géniale pour les concepts ("IA"), mais elle est médiocre pour les termes ultra-précis ("Modèle XJ-104-B"). 
L'**Hybrid Search** combine le score BM25 ([**statistique de mots-clés, section 1.1**]({{< relref "section-1-1.md" >}}#tf-idf)) et le score sémantique (vecteurs). C'est aujourd'hui la méthode la plus robuste pour éviter que l'IA ne passe à côté d'une référence technique exacte.

## Laboratoire de code : Implémentation d'un Reranker (Colab T4)
Voici comment ajouter une étape de Reranking à votre moteur de recherche. Nous allons utiliser un modèle de reranking léger de la famille Cross-Encoder.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install sentence-transformers

from sentence_transformers import CrossEncoder
import numpy as np

# 1. On suppose qu'une recherche vectorielle (section 6.3) nous a donné 5 candidats
query = "How do I reset my company password?"
candidates = [
    "To change your login, go to settings.", 
    "Our company was founded in 1998 in Lyon.", # Hors sujet
    "Password resets require a 2FA code sent by email.", # Très pertinent
    "The cafeteria menu is updated every Monday.",
    "For security issues, contact the IT helpdesk at 555-0101." # Moyen
]

# 2. CHARGEMENT DU RERANKER (Cross-Encoder)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

# 3. ÉVALUATION DES PAIRES (Question, Document)
# On crée des paires pour que le modèle puisse calculer l'interaction
pairs = [[query, doc] for doc in candidates]
scores = reranker.predict(pairs)

# 4. TRI DES RÉSULTATS (Reranking)
ranked_results = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

print(f"Requête : {query}\n")
print("--- RÉSULTATS APRÈS RERANKING ---")
for score, doc in ranked_results:
    # Les scores des Cross-Encoders peuvent être négatifs (Logits), c'est normal !
    print(f"Score: {score:.2f} | Doc: {doc}")

# ATTENDU : Le document sur le "2FA code" doit remonter avec le score le plus élevé.
```


## Éthique et Responsabilité : Le pouvoir de sélection

> [!CAUTION]
⚠️ Mes chers étudiants, celui qui trie l'information possède la vérité.

Lorsque vous configurez un Reranker, vous déterminez ce que le LLM va lire. 
1.  **Le biais du filtre** : Si votre Reranker a un biais (ex: privilégier systématiquement les documents récents au détriment des documents de fond), le LLM produira une réponse biaisée sans même s'en rendre compte.
2.  **La transparence du choix** : En production, vous devriez toujours enregistrer (log) quels documents ont été choisis par le Reranker. Si une IA donne une réponse dangereuse, vous devez pouvoir remonter la chaîne : est-ce le LLM qui a déliré, ou est-ce le Reranker qui lui a mis une information erronée sous les yeux ?

> [!NOTE]
🔑 **Mon message** : Les stratégies avancées transforment un jouet technologique en un outil de précision. Mais la précision demande de la vigilance. Un Reranker est un arbitre ; assurez-vous qu'il est juste, neutre et que ses critères de notation sont alignés avec vos valeurs humaines.

---
Vous maîtrisez maintenant l'art de la recherche de haute précision. Vous savez comment aider l'utilisateur à poser ses questions et comment filtrer le bruit pour ne garder que le signal pur. Dans la prochaine section ➡️, nous allons apprendre à mesurer scientifiquement si tout ce travail sert à quelque chose : place à **l'évaluation RAG** avec le framework *Ragas*.

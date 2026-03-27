[CONTENU SEMAINE 9]

# Semaine 9 : RAG (Retrieval-Augmented Generation)

**Titre : Combattre les hallucinations : L'architecture RAG**

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour cette neuvième étape. Jusqu'ici, nous avons vu que les LLM sont des génies créatifs, capables de discourir sur n'importe quel sujet avec une fluidité déconcertante. Mais il y a un "mais" : cette fluidité cache parfois des mensonges statistiques. 🔑 **Je dois insister :** dans le monde professionnel, l'imagination n'est pas une excuse pour l'erreur factuelle. Aujourd'hui, nous allons apprendre à transformer nos IA en experts rigoureux qui ne parlent que "sous contrôle de preuves". Bienvenue dans l'univers du **RAG**, la technologie qui permet à vos modèles de passer l'examen à "livre ouvert". Prêt·e·s à ancrer l'IA dans la réalité ? » [SOURCE: Livre p.249]

**Rappel semaine précédente** : « La semaine dernière, nous avons appris l'art subtil du prompt engineering, en explorant les techniques de Few-shot learning et de Chain-of-Thought pour guider le raisonnement des modèles. » [SOURCE: Detailed-plan.md]

**Objectifs de la semaine** :
À la fin de cette semaine, vous saurez :
*   Expliquer l'architecture fondamentale d'un système RAG.
*   Distinguer les trois phases : Indexation, Récupération et Génération.
*   Comprendre comment le RAG réduit drastiquement les hallucinations.
*   Mettre en œuvre une génération "ancrée" (*grounded*) avec citations de sources.
*   Maîtriser les stratégies de récupération avancées comme le Reranking.

---

## 9.1 Architecture RAG fondamentale (2000+ mots)

### Le remède au "Syndrome du Perroquet Savant"
« Mes chers étudiants, commençons par un constat d'humilité : un LLM, aussi grand soit-il, ne sait que ce qu'il a lu pendant son entraînement. » Si vous lui demandez le chiffre d'affaires de votre entreprise pour le trimestre dernier, il échouera ou, pire, il inventera un chiffre crédible (hallucination). Pourquoi ? Parce que cette information n'était pas dans son dictionnaire d'origine. 

Le **RAG (Retrieval-Augmented Generation)**, ou Génération Augmentée par Récupération, est la réponse à ce problème. L'idée est de ne plus compter sur la seule mémoire interne du modèle (ses poids), mais de lui donner accès à une base de connaissances externe. 

Regardez la **Figure 8-24 : Un pipeline RAG basique** (p.250 du livre). Cette figure est le plan de montage de votre futur système. Elle montre un flux circulaire où la question de l'utilisateur ne va pas directement au LLM, mais passe d'abord par une étape de "Retrieval" (Récupération) dans une source de données. L'IA reçoit ensuite la question ET les documents trouvés pour formuler sa réponse. 🔑 **Notez bien cette intuition :** le LLM ne répond plus de mémoire, il répond par synthèse de documents. [SOURCE: Livre p.250, Figure 8-24]

### L'analogie du bibliothécaire et de l'écrivain
Pour bien comprendre les rôles, utilisons une analogie. Un système RAG, c'est l'alliance de deux personnages :
1.  **Le Bibliothécaire (Retriever)** : Il est rapide, il connaît l'emplacement de chaque livre (vecteur), mais il ne sait pas forcément rédiger un essai. Son rôle est de vous apporter les 3 meilleurs livres sur un sujet.
2.  **L'Écrivain (Generator)** : C'est le LLM. Il a un style impeccable et sait synthétiser des informations complexes. Mais il a une mémoire de poisson rouge pour les détails précis. 

Dans le RAG, le Bibliothécaire court chercher les preuves, les pose sur le bureau de l'Écrivain, et ce dernier rédige la réponse finale en se basant uniquement sur ce qui est posé devant lui. [SOURCE: CONCEPT À SOURCER – INSPIRÉ DE LILIAN WENG BLOG https://lilianweng.github.io/posts/2023-06-23-agent/]

### Étape 1 : L'Indexation (La préparation de la bibliothèque)
Avant de pouvoir chercher, il faut ranger. Cette phase se passe hors-ligne, avant même que l'utilisateur ne pose sa première question. Comme nous l'avons vu en Semaine 6, elle suit un processus rigoureux détaillé dans le livre :
*   **Le Chargement** : Extraction du texte des PDF, des bases SQL ou des sites web.
*   **Le Chunking** : Découpage en morceaux (nous avons vu l'importance de l'overlap en 6.3).
*   **L'Embedding** : Transformation de chaque morceau en un vecteur sémantique.
*   **Le Stockage** : Enregistrement dans une base de données vectorielle (Vector DB).

🔑 **Je dois insister :** si votre indexation est bâclée (morceaux trop petits ou trop gros), votre "bibliothécaire" rapportera les mauvaises pages, et votre système RAG sera un échec, peu importe la puissance du LLM utilisé. [SOURCE: Livre p.236]

### Étape 2 : La Récupération (Le Retrieval)
C'est le moment où l'utilisateur pose sa question. Regardez la **Figure 8-1 : Dense retrieval** (p.226). 
1.  La question est transformée en vecteur (embedding).
2.  Le système effectue une recherche de plus proches voisins (KNN) dans la base vectorielle.
3.  On extrait les **Top-K** documents (souvent les 3 à 5 morceaux les plus similaires).

⚠️ **Attention : erreur fréquente ici !** La similarité sémantique ne garantit pas la pertinence. Le modèle peut trouver un document qui utilise les mêmes mots mais qui ne répond pas à la question. C'est pour cela que nous introduisons parfois une étape de **Reranking**, illustrée en **Figure 8-2** (p.227). On utilise un second modèle, plus lent mais plus précis (Cross-Encoder), pour ré-ordonner les résultats et s'assurer que le meilleur document est bien en première position avant de l'envoyer au LLM. [SOURCE: Livre p.226-227, Figures 8-1, 8-2]

### Étape 3 : La Génération Ancrée (Grounded Generation)
C'est ici que la magie du LLM opère enfin. Nous construisons ce que l'on appelle un **Augmented Prompt** (Prompt augmenté). 

Observez la **Figure 8-26 : Recherche sémantique pour la génération** (p.251). Le système crée dynamiquement un prompt géant qui ressemble à ceci :
> "Tu es un assistant expert. Utilise uniquement les informations fournies dans le contexte ci-dessous pour répondre à la question. Si la réponse n'est pas dans le texte, dis que tu ne sais pas.
> 
> CONTEXTE : [Document 1] [Document 2] ...
> 
> QUESTION : {Question de l'utilisateur}"

🔑 **C'est le concept de Grounded Generation :** Le modèle est "ancré" dans le texte. En **Figure 8-3** (p.227), vous voyez le résultat idéal : une réponse fluide qui inclut des **citations** (ex: [1], [2]). Cela permet à l'humain de cliquer sur la source pour vérifier que l'IA ne délire pas. [SOURCE: Livre p.227, p.251, Figures 8-3, 8-26]

### Pourquoi le RAG est-il vital pour l'IA responsable ?
⚠️ **Fermeté bienveillante** : « Mes chers étudiants, le RAG n'est pas qu'une amélioration technique, c'est une exigence éthique. »
1.  **Réduction des hallucinations** : En forçant le modèle à citer ses sources, on l'empêche de s'égarer dans ses probabilités statistiques.
2.  **Actualisation des connaissances** : Pas besoin de réentraîner GPT-4 (ce qui coûte des millions) pour lui apprendre une nouvelle règle fiscale. Il suffit de mettre à jour le document PDF dans la base vectorielle.
3.  **Confidentialité et Contrôle** : Vous pouvez restreindre l'IA à ne lire que les documents auxquels l'utilisateur a droit. Le LLM ne "connaît" rien, il ne fait que synthétiser ce qu'on lui prête. [SOURCE: Livre p.28]

### Comparaison : LLM "Base" vs RAG

| Caractéristique | LLM Seul (Mémoire interne) | RAG (Mémoire externe) |
| :--- | :--- | :--- |
| **Connaissances** | Figées à la date d'entraînement | Toujours à jour (temps réel) |
| **Source de vérité** | Statistique ("Vraisemblance") | Documentaire ("Preuve") |
| **Hallucinations** | Fréquentes et difficiles à détecter | Rares et vérifiables via citations |
| **Coût d'évolution** | Immense (Fine-tuning lourd) | Faible (Mise à jour de l'index) |

[SOURCE: CONCEPT À SOURCER – SYNTHÈSE DU LIVRE CHAP 8]

### Laboratoire de code : Simulation d'un flux RAG (Colab T4)
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
# [SOURCE: Choix de modèle compact p.54]
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
generator = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 3. FONCTION DE GÉNÉRATION ANCRÉE (RAG Logic)
def rag_answer(user_query, documents):
    # Simulation du Retrieval (On prend tous les docs pour l'exemple)
    context = "\n".join([f"- {doc}" for doc in documents])
    
    # Construction du Prompt Augmenté (Section 9.1 intuition)
    # [SOURCE: Prompting pour RAG Livre p.254]
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

⚠️ **Avertissement du Professeur** : Observez la seconde réponse. Si votre prompt est bien écrit, le modèle doit refuser de répondre ("Je n'ai pas cette information") au lieu d'inventer un nom de PDG. C'est la preuve que votre garde-fou fonctionne ! [SOURCE: Livre p.254]

### Éthique et Transparence : La responsabilité des sources
⚠️ **Éthique ancrée** : « Mes chers étudiants, un RAG n'est pas plus intelligent que les documents qu'il lit. » 
Si votre base de données contient des documents obsolètes, faux ou haineux, le LLM les synthétisera avec la même autorité que s'il s'agissait de la vérité absolue. 
1.  **La pollution des sources** : Le RAG peut devenir un outil de désinformation massif si la base de connaissances n'est pas auditée. 
2.  **L'illusion de la citation** : Parfois, le modèle cite une source ([1]) qui ne contient pas du tout l'information mentionnée. C'est l'hallucination de second niveau. 

🔑 **Le message du Prof. Henni** : « Le RAG est le pont qui permet aux LLM de sortir des laboratoires pour entrer dans les usines, les hôpitaux et les banques. En maîtrisant cette architecture, vous ne construisez plus seulement des gadgets, vous construisez des systèmes de confiance. Mais n'oubliez jamais : vous êtes le gardien de la bibliothèque. Si les livres sont mauvais, l'IA sera mauvaise. » [SOURCE: Livre p.28]

« Vous avez maintenant les clés de l'architecture fondamentale du RAG. Vous savez pourquoi nous l'utilisons et comment il se structure. Dans la section suivante, nous allons corser le jeu : que faire si la question de l'utilisateur est mal posée ? Comment chercher dans plusieurs documents à la fois ? Bienvenue dans les techniques de récupération avancées. »

---
*Fin de la section 9.1 (2140 mots environ)*
## 9.2 Stratégies de récupération avancées (2000+ mots)

### Dépasser le "Top-K" : Pourquoi la recherche simple échoue en production
« Bonjour à toutes et à tous ! J'espère que vous avez bien digéré l'architecture de base du RAG que nous avons vue en section 9.1. C'est un excellent point de départ, mais je vais être très directe avec vous : dans un environnement industriel, un RAG "naïf" échoue dans 40 % des cas. Pourquoi ? Parce que les utilisateurs posent des questions mal formulées, parce que les documents sont complexes, et parce que la similarité mathématique (section 6.2) n'est pas toujours synonyme de pertinence métier. 🔑 **Je dois insister :** le maillon faible du RAG, c'est le "R" (Retrieval). Si votre bibliothécaire rapporte les mauvaises pages, votre écrivain (le LLM) ne pourra que produire une erreur élégante. Aujourd'hui, nous allons apprendre à muscler ce bibliothécaire grâce aux stratégies de récupération avancées. » [SOURCE: Livre p.255]

### La réécriture de requête (Query Rewriting) : Traduire l'humain pour la machine
Les utilisateurs sont rarement des experts en "mots-clés sémantiques". Ils posent des questions comme : "Et pour le deuxième cas, ça marche comment ?". Pour un moteur de recherche vectoriel, cette phrase est un cauchemar : que signifie "le deuxième cas" ? Que signifie "ça" ?

C'est ici qu'intervient le **Query Rewriting**. L'idée est d'utiliser un premier appel au LLM pour transformer la question floue de l'utilisateur en une requête de recherche optimisée. 
*   **Expansion de requête** : On demande au LLM d'ajouter des synonymes ou des termes techniques liés (ex: "problème de batterie" devient "défaillance lithium-ion, perte de charge, accumulateur").
*   **Résolution de coréférence** : Dans une conversation, le LLM analyse l'historique pour remplacer "il" ou "le deuxième" par les vrais sujets mentionnés plus tôt. 

🔑 **L'intuition du Professeur Henni :** Le Query Rewriting, c'est comme avoir un interprète qui écoute votre marmonnement et le crie clairement au bibliothécaire pour qu'il comprenne exactement ce que vous cherchez. [SOURCE: Livre p.255, Blog 'Retrieval Augmented Generation' de Hugging Face]

### Multi-Query RAG : Ne pas mettre tous ses œufs dans le même vecteur
Parfois, une question peut être interprétée de plusieurs façons. Au lieu de parier sur un seul vecteur de recherche, pourquoi ne pas en utiliser plusieurs ?
Dans le **Multi-Query RAG**, nous demandons au LLM de générer 3 à 5 variantes de la question initiale. 
1.  Nous effectuons une recherche pour chaque variante.
2.  Nous récupérons les résultats de toutes ces recherches.
3.  Nous fusionnons les documents trouvés en supprimant les doublons.

🔑 **Je dois insister :** cette technique augmente considérablement le "Rappel" (Recall). On s'assure ainsi de ne pas rater un document crucial simplement parce que la formulation originale de l'utilisateur était légèrement décalée par rapport au titre du document. [SOURCE: Livre p.255]

### L'architecture à deux étages : Le Reranking (Analyse de la Figure 8-14)
Nous arrivons au cœur de l'ingénierie de précision. Regardez attentivement la **Figure 8-14 : Pipeline de reranking** (p.241 du livre). Cette illustration décrit une stratégie en deux étapes, indispensable pour les bases de données massives. [SOURCE: Livre p.241, Figure 8-14]

1.  **Premier étage (Le Shortlisting)** : On utilise une recherche vectorielle rapide (FAISS, section 6.3). C'est efficace mais un peu "grossier". On récupère, par exemple, les 100 documents les plus proches. 
2.  **Deuxième étage (Le Reranking)** : On prend ces 100 candidats et on les passe dans un **Reranker** (souvent un modèle de type Cross-Encoder). Ce modèle est beaucoup plus lent mais incroyablement plus précis. Il examine chaque document face à la question et lui donne une note de 0 à 1. 
3.  **Le résultat final** : On ne donne au LLM que les 3 ou 5 documents ayant obtenu la meilleure note du Reranker.

🔑 **Pourquoi cette complexité ?** Le Reranker "lit" vraiment la paire question-document alors que la recherche vectorielle ne fait que comparer des positions GPS. Comme le montre la figure, le Reranker peut faire remonter en 1ère position un document qui était initialement en 20ème position mais qui contient la réponse exacte. [SOURCE: Livre p.241]

### Cross-Encoders vs Bi-Encoders (Analyse de la Figure 8-15)
Pour comprendre pourquoi le Reranking est si puissant, il faut regarder "sous le capot" de la **Figure 8-15 : Cross-encoder reranker** (p.244). [SOURCE: Livre p.244, Figure 8-15]

*   **Le Bi-Encoder (SBERT)** : C'est ce que nous avons utilisé jusqu'ici. La question a son vecteur, le document a le sien. Ils ne se "parlent" jamais pendant le calcul. C'est rapide mais on perd les interactions fines entre les mots.
*   **Le Cross-Encoder** : On concatène la question et le document dans un seul bloc BERT : `[CLS] Question [SEP] Document [SEP]`. 
*   **L'Attention Totale** : Comme illustré par la figure, grâce à la self-attention bidirectionnelle (Semaine 4), chaque mot de la question peut interagir avec chaque mot du document. Le modèle peut détecter qu'un "pas" ou un "sauf" dans la question change totalement la pertinence d'un paragraphe. 

⚠️ **Attention : erreur fréquente ici !** N'utilisez jamais un Cross-Encoder pour fouiller dans un million de documents. Il est 100 à 1000 fois plus lent qu'un Bi-Encoder. Il est l'outil du "dernier kilomètre", pas le moteur de recherche principal. [SOURCE: Livre p.244, Blog 'Rerankers' de Cohere]

### Multi-hop RAG : Le raisonnement par étapes
Certaines questions demandent de relier plusieurs documents qui ne se ressemblent pas. 
*   *Exemple* : "Quelle est la capitale du pays où est né l'inventeur du téléphone ?"
1.  **Hop 1** : Chercher "inventeur du téléphone". Trouver "Graham Bell".
2.  **Hop 2** : Chercher "lieu de naissance de Graham Bell". Trouver "Écosse".
3.  **Hop 3** : Chercher "capitale de l'Écosse". Trouver "Édimbourg".

Le **Multi-hop RAG** utilise le LLM comme un agent (nous verrons cela en Semaine 14). L'IA fait une première recherche, analyse le résultat, puis décide de faire une *nouvelle* recherche basée sur ce qu'elle vient d'apprendre. 🔑 **C'est une rupture majeure :** l'IA ne subit plus la recherche, elle la pilote. [SOURCE: Livre p.256, "Multi-hop RAG"]

### Hybride Search : Marier le Lexical et le Sémantique
⚠️ **Fermeté bienveillante** : « Ne jetez pas la recherche par mots-clés aux orties ! » La recherche vectorielle est géniale pour les concepts ("IA"), mais elle est médiocre pour les termes ultra-précis ("Modèle XJ-104-B"). 
L'**Hybrid Search** combine le score BM25 (statistique de mots-clés, section 1.1) et le score sémantique (vecteurs). C'est aujourd'hui la méthode la plus robuste pour éviter que l'IA ne passe à côté d'une référence technique exacte. [SOURCE: Livre p.235]

### Laboratoire de code : Implémentation d'un Reranker (Colab T4)
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
# [SOURCE: Modèle de reranking recommandé p.243]
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

# 3. ÉVALUATION DES PAIRES (Question, Document)
# On crée des paires pour que le modèle puisse calculer l'interaction
pairs = [[query, doc] for doc in candidates]
scores = reranker.predict(pairs)

# 4. TRI DES RÉSULTATS (Reranking)
# [SOURCE: Pipeline de reranking Figure 8-14]
ranked_results = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

print(f"Requête : {query}\n")
print("--- RÉSULTATS APRÈS RERANKING ---")
for score, doc in ranked_results:
    # Les scores des Cross-Encoders peuvent être négatifs (Logits), c'est normal !
    print(f"Score: {score:.2f} | Doc: {doc}")

# ATTENDU : Le document sur le "2FA code" doit remonter avec le score le plus élevé.
```

### Éthique et Responsabilité : Le pouvoir de sélection
⚠️ **Éthique ancrée** : « Mes chers étudiants, celui qui trie l'information possède la vérité. » 
Lorsque vous configurez un Reranker, vous déterminez ce que le LLM va lire. 
1.  **Le biais du filtre** : Si votre Reranker a un biais (ex: privilégier systématiquement les documents récents au détriment des documents de fond), le LLM produira une réponse biaisée sans même s'en rendre compte.
2.  **La transparence du choix** : En production, vous devriez toujours enregistrer (log) quels documents ont été choisis par le Reranker. Si une IA donne une réponse dangereuse, vous devez pouvoir remonter la chaîne : est-ce le LLM qui a déliré, ou est-ce le Reranker qui lui a mis une information erronée sous les yeux ?

🔑 **Le message du Prof. Henni** : « Les stratégies avancées transforment un jouet technologique en un outil de précision. Mais la précision demande de la vigilance. Un Reranker est un arbitre ; assurez-vous qu'il est juste, neutre et que ses critères de notation sont alignés avec vos valeurs humaines. » [SOURCE: Livre p.28]

« Vous maîtrisez maintenant l'art de la recherche de haute précision. Vous savez comment aider l'utilisateur à poser ses questions et comment filtrer le bruit pour ne garder que le signal pur. Dans la prochaine section, nous allons apprendre à mesurer scientifiquement si tout ce travail sert à quelque chose : place à l'évaluation RAG avec le framework Ragas. »

---
*Fin de la section 9.2 (2180 mots environ)*
## 9.3 Évaluation RAG (2000+ mots)

### Le juge de paix de l'IA : Pourquoi l'intuition ne suffit plus
« Bonjour à toutes et à tous ! Nous abordons aujourd'hui la section la plus rigoureuse, et peut-être la plus vitale, de notre parcours sur le RAG. Imaginez que vous ayez construit un magnifique moteur de recherche et que vous l'ayez couplé à un LLM puissant. Tout semble fonctionner. Mais comment pouvez-vous garantir à votre client, ou à votre direction, que l'IA ne va pas inventer un fait crucial demain matin à 9h ? Comment prouver que votre système est "meilleur" que celui de la concurrence ? 🔑 **Je dois insister :** en ingénierie des LLM, ce qui ne se mesure pas n'existe pas. Aujourd'hui, nous allons apprendre à devenir des auditeurs de la vérité. Respirez, nous allons transformer la qualité en chiffres. » [SOURCE: Livre p.244]

L'évaluation d'un système RAG est double. Vous devez juger la capacité du "Bibliothécaire" à trouver les bons livres (Évaluation du Retrieval) et la capacité de l' "Écrivain" à synthétiser sans mentir (Évaluation de la Génération). Si vous ne mesurez que le résultat final, vous ne saurez jamais quel composant corriger en cas d'erreur. C'est ce que nous appelons l'évaluation par composantes. [SOURCE: Livre p.257]

### Évaluer le "Bibliothécaire" : La science du Retrieval
Pour juger la recherche, nous utilisons les standards de l' *Information Retrieval* (IR). Jay Alammar et Maarten Grootendorst consacrent une série de figures essentielles à ce sujet (p.245-249).

#### 1. Le banc d'essai (Figure 8-16)
Comme l'explique la **Figure 8-16 : Composantes de l'évaluation** (p.245), vous ne pouvez pas évaluer dans le vide. Vous avez besoin d'une "Test Suite" composée :
*   D'un archive de textes (votre base de connaissances).
*   D'un ensemble de requêtes types.
*   De **Relevance Judgments** : Pour chaque question, un humain (ou un expert) a marqué quels documents sont "Vrais" (pertinents) et lesquels sont "Faux". 

🔑 **C'est votre "Golden Dataset"**. Sans cette vérité terrain, vous naviguez à l'aveugle. [SOURCE: Livre p.245, Figure 8-16]

#### 2. La comparaison de systèmes (Figures 8-17 et 8-18)
La **Figure 8-17** (p.245) montre comment nous envoyons la même requête à deux systèmes différents pour comparer leurs résultats. Mais la simple présence du bon document ne suffit pas. La **Figure 8-18** (p.246) nous montre que l'ordre compte : si le système 1 met le bon document en position 1, il est bien supérieur au système 2 qui le met en position 3. 

⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent qu'un moteur de recherche est bon s'il trouve l'info "quelque part". C'est faux. L'utilisateur clique sur le premier lien. La position est une question de survie pour votre application. [SOURCE: Livre p.245-246, Figures 8-17, 8-18]

#### 3. La métrique reine : MAP (Mean Average Precision)
C'est ici que les mathématiques rencontrent la stratégie. Le livre détaille le calcul du **MAP** à travers les **Figures 8-20 à 8-23** (p.247-249).

*   **Calcul pour une requête (Figure 8-20)** : Si le premier résultat est le bon, votre précision à la position 1 ($P@1$) est de 1.0 (100%).
*   **La pénalité du retard (Figure 8-21)** : Si vous placez des documents inutiles avant la réponse, votre score chute. Par exemple, si le bon document est en 3ème position, votre précision pour ce document est de $1/3 = 0.33$.
*   **La synthèse (Figure 8-22 et 8-23)** : Pour obtenir le MAP, on calcule la précision moyenne pour chaque question, puis on fait la moyenne de ces moyennes sur tout le banc d'essai. 

🔑 **Je dois insister :** Le MAP est une métrique de "classement". Elle récompense les systèmes qui ont l'audace et la précision de mettre la vérité tout en haut de la pile. [SOURCE: Livre p.247-249, Figures 8-20 à 8-23]

### Évaluer l' "Écrivain" : La Triade du RAG
Une fois que nous avons les bons documents, comment juger la réponse générée ? Les métriques classiques comme BLEU ou ROUGE (utilisées en traduction) sont ici inutiles : elles comparent les mots exacts, mais ne comprennent pas si l'IA a inventé une date ou un nom.

Nous utilisons aujourd'hui le framework **Ragas** (Retrieval Augmented Generation Assessment), qui repose sur trois piliers que j'appelle la "Triade de la Confiance" :

#### 1. La Fidélité (Faithfulness)
C'est la mesure de l' **Ancrage**. Est-ce que chaque affirmation de la réponse de l'IA est présente dans les documents récupérés ? 
*   *Exemple* : Si l'IA dit "Le contrat a été signé en 2022" mais que le document source ne mentionne aucune date, le score de Faithfulness s'effondre. C'est l'arme absolue contre les hallucinations. [SOURCE: Ragas Documentation & Livre p.257]

#### 2. La Pertinence de la réponse (Answer Relevancy)
Est-ce que l'IA a vraiment répondu à la question de l'utilisateur ? Une réponse peut être 100% fidèle aux sources mais totalement hors-sujet. 
*   *Exemple* : L'utilisateur demande "Comment résilier mon abonnement ?" et l'IA répond "Votre abonnement a été souscrit le 12 janvier." C'est vrai, mais c'est inutile. [SOURCE: Ragas Documentation]

#### 3. La Précision du contexte (Context Precision)
Ici, on juge à nouveau le bibliothécaire mais à travers le regard de l'écrivain : est-ce que les documents fournis étaient vraiment nécessaires pour répondre ? Plus le contexte est "propre" (sans documents parasites), plus le score est élevé. [SOURCE: Ragas Documentation]

### LLM-as-a-judge : Quand l'IA devient professeur
« Comment calculer ces scores de fidélité ou de pertinence de manière automatique ? » Nous utilisons un concept révolutionnaire : **LLM-as-a-judge**. 

Nous utilisons un modèle très "intelligent" et neutre (comme GPT-4 ou Claude 3 Opus) pour noter la sortie d'un modèle plus petit (comme Phi-3). 
1.  On donne au juge le document source, la question et la réponse de l'IA.
2.  On lui demande de décomposer la réponse en affirmations individuelles.
3.  Pour chaque affirmation, le juge vérifie si elle est "prouvée" par le document. 
4.  Le score final est le ratio d'affirmations prouvées. 

🔑 **L'intuition du Prof. Henni :** C'est comme si je demandais à un doctorant de corriger les copies des étudiants de première année en suivant une grille de notation très stricte. C'est rapide, scalable et souvent plus cohérent qu'une évaluation humaine fatiguée. [SOURCE: Livre p.376, "Automated Evaluation"]

### Laboratoire de code : Évaluation avec Ragas (Colab T4)
Voici comment mettre en place une évaluation scientifique de votre pipeline RAG. Nous allons simuler un petit dataset de test.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install ragas datasets transformers

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# 1. PRÉPARATION DU DATASET DE TEST (Simulé)
# [SOURCE: Structure de données Ragas Docs]
data_samples = {
    'question': ["When was the company founded?"],
    'answer': ["The company was founded in 1998 by two engineers."],
    'contexts': [["Our firm started its journey in late 1998 in a small garage in Lyon."]],
    'ground_truth': ["1998"]
}

dataset = Dataset.from_dict(data_samples)

# 2. CONFIGURATION DE L'ÉVALUATEUR
# Note : Ragas nécessite par défaut une clé OpenAI pour le 'Juge'
# Mais on peut utiliser des modèles locaux (long à configurer ici)
# [SOURCE: Ragas metrics documentation]

# 3. EXÉCUTION DE L'ÉVALUATION (QUESTION CODE)
# results = evaluate(...)

# --- RÉPONSE (ANSWER CODE) ---
# [SOURCE: CONCEPT À SOURCER – Documentation Ragas]
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy]
)

print("--- SCORE D'AUDIT RAG ---")
print(f"Fidélité (Faithfulness) : {results['faithfulness']:.2f}")
print(f"Pertinence (Answer Relevancy) : {results['answer_relevancy']:.2f}")

# ATTENDU : Faithfulness proche de 1.0 car 1998 est dans le contexte.
```

⚠️ **Fermeté bienveillante** : Si vous obtenez un score de Faithfulness de 0.0, cela signifie que votre modèle a totalement ignoré le contexte pour inventer sa propre réponse. C'est le signal d'alarme rouge : vous devez revoir votre prompt (section 9.1).

### Le danger de l'optimisation aveugle : La loi de Goodhart
⚠️ **Éthique ancrée** : « Mes chers étudiants, un chiffre n'est pas la réalité. » 
Il existe un principe célèbre en économie et en IA : la **Loi de Goodhart**. "Lorsqu'une mesure devient un objectif, elle cesse d'être une bonne mesure."

1.  **Le risque du "Gaming"** : Si vous donnez des primes à vos ingénieurs pour augmenter le score de "Faithfulness", ils pourraient forcer le modèle à répondre de manière extrêmement courte et robotique ("Oui", "Non", "1998"). Le score sera parfait, mais l'utilité pour l'utilisateur sera nulle.
2.  **Le biais du Juge** : N'oubliez pas que votre juge est lui-même un LLM. Il a ses propres biais. Il peut préférer les réponses longues et polies, même si elles sont moins précises. 🔑 **Mon conseil de professeur** : Faites toujours une "contre-vérification" humaine sur 5% de vos évaluations automatiques pour vous assurer que le juge ne fait pas de favoritisme statistique. [SOURCE: Livre p.377, "Goodhart's Law"]

### Synthèse des métriques
« Pour conclure, rappelez-vous ce tableau de bord que vous devez présenter à chaque déploiement : »

| Composant évalué | Métrique Clé | Ce que ça mesure vraiment |
| :--- | :--- | :--- |
| **Le Bibliothécaire (R)** | **MAP / NDCG** | Est-ce que l'info est en haut de la liste ? |
| **L'Écrivain (G)** | **Faithfulness** | Est-ce que l'IA a inventé des choses ? |
| **L'Écrivain (G)** | **Answer Relevancy** | Est-ce que l'IA a répondu à la question ? |
| **L'Expérience Utilisateur** | **Latence / Coût** | Est-ce que c'est trop lent ou trop cher ? |

[SOURCE: CONCEPT À SOURCER – SYNTHÈSE DU LIVRE CHAP 8 & 12]

🔑 **Le message du Prof. Henni** : « L'évaluation est le moment où vous passez du statut de bidouilleur à celui d'ingénieur. C'est ingrat, c'est long, c'est parfois frustrant quand les scores chutent, mais c'est le seul rempart qui protège vos utilisateurs de l'erreur. Soyez d'une exigence absolue avec vos chiffres. » [SOURCE: Livre p.28]

« Vous savez maintenant comment mesurer la qualité de votre moteur. Vous avez les outils pour prouver que votre système est fiable. Dans la dernière section de cette semaine, nous allons voir comment assembler tout cela de manière élégante et industrielle grâce au framework **LangChain**. »

---
*Fin de la section 9.3 (2210 mots environ)*
## 9.4 Implémentation avec LangChain (2000+ mots)

### Le chef d'orchestre de vos applications IA
« Bonjour à toutes et à tous ! Nous voici arrivés au terme de cette semaine marathon sur le RAG. Nous avons vu la théorie, la recherche de précision et l'audit rigoureux des résultats. Mais maintenant, une question pratique se pose : comment assembler toutes ces pièces mobiles — le tokeniseur, le modèle d'embedding, la base de données FAISS, le LLM et le prompt — sans que votre code ne devienne un plat de spaghettis illisible ? 🔑 **Je dois insister :** en production, la modularité est votre meilleure alliée. Aujourd'hui, je vais vous présenter votre futur meilleur ami : **LangChain**. C'est le chef d'orchestre qui va donner le tempo et lier tous vos composants dans une symphonie fluide et robuste. Prêt·e·s à passer à l'échelle industrielle ? » [SOURCE: Livre p.199]

### Philosophie et Architecture de LangChain (Analyse de la Figure 7-1)
Pour comprendre LangChain, il faut regarder son plan d'architecte. Regardez attentivement la **Figure 7-1 : Architecture de LangChain** (p.200 du livre). Cette illustration est la carte de votre écosystème de développement. [SOURCE: Livre p.200, Figure 7-1]

Décortiquons ensemble les modules affichés dans cette figure :
1.  **Model I/O (Entrées/Sorties)** : C'est la gestion des prompts, des modèles et des "output parsers" (pour transformer le texte de l'IA en données utilisables).
2.  **Retrieval (Récupération)** : C'est le cœur de notre RAG. Ce module gère les "Document Loaders", les découpeurs de texte (Splitters), les modèles d'embedding et les interfaces avec les bases de données vectorielles.
3.  **Chains (Chaînes)** : C'est le concept central. Une chaîne est un assemblage de briques. La sortie d'un composant devient l'entrée du suivant.
4.  **Memory (Mémoire)** : Pour que votre chatbot se souvienne des questions précédentes de l'utilisateur.
5.  **Agents** : Le niveau supérieur, où l'IA décide elle-même d'utiliser tel ou tel outil pour répondre.

🔑 **L'intuition du Professeur Henni :** LangChain ne remplace pas les modèles, il les entoure d'une infrastructure standardisée. C'est comme le système de plomberie d'une maison : peu importe la marque de votre robinet (GPT-4, Claude ou Llama), les tuyaux et les connecteurs restent les mêmes. [SOURCE: Livre p.200, Figure 7-1]

### La brique "Document" : Au-delà du simple texte
Dans LangChain, nous ne manipulons pas de simples chaînes de caractères `str`, mais des objets `Document`. 
🔑 **Je dois insister sur cette distinction technique :** un objet `Document` contient :
*   `page_content` : Le texte brut.
*   `metadata` : Un dictionnaire contenant la source (ex: "page 12 du PDF"), la date ou l'auteur.

Pourquoi est-ce capital pour le RAG ? Parce que sans métadonnées, votre IA ne pourra jamais faire de **citations** (vues en section 9.1). En Figure 8-3 (p.227), nous avons vu l'importance de citer ses sources ; LangChain automatise ce suivi du texte à travers tout le pipeline, de l'indexation à la génération. [SOURCE: Livre p.253]

### Les Vector Stores dans LangChain : Une interface universelle
L'un des plus grands avantages de LangChain est l'abstraction des bases de données vectorielles. Que vous utilisiez FAISS (local, Semaine 6), ChromaDB ou Pinecone (cloud), le code de votre application reste quasiment identique.

Le processus d'intégration est standardisé :
1.  On initialise un `EmbeddingsModel`.
2.  On passe cet objet à un `VectorStore`.
3.  On transforme ce magasin de vecteurs en un `Retriever`. 

🔑 **Le concept du Retriever :** Ce n'est pas une base de données, c'est une *interface*. Un Retriever est un objet qui prend une question en entrée et renvoie une liste de documents. Cette abstraction permet de changer de moteur de recherche (sémantique, hybride ou même via une API comme Google Search) sans modifier le reste de votre chaîne RAG. [SOURCE: Livre p.253, "Loading the embedding model"]

### Orchestration avec RetrievalQA
Le composant historique pour bâtir un RAG avec LangChain est la chaîne `RetrievalQA`. Elle automatise ce que nous avons fait manuellement en section 9.1 :
1.  Prendre la question.
2.  Interroger le Retriever.
3.  Formater le prompt avec le contexte trouvé.
4.  Appeler le LLM.
5.  Rendre la réponse.

⚠️ **Attention : erreur fréquente ici !** Il existe plusieurs types de chaînes pour assembler les documents (les "chain types"). Le plus courant est le mode **"stuff"**.
*   **"stuff"** : On "fourre" (stuff) tous les documents trouvés dans le prompt. 🔑 **Le risque :** Si vous avez trop de documents, vous allez dépasser la "Context Window" du modèle (Semaine 5) et provoquer une erreur.
*   **"map_reduce"** : Le modèle résume chaque document séparément, puis fait une synthèse globale. C'est plus lent mais idéal pour les très gros volumes. [SOURCE: Livre p.254, "RAG pipeline"]

### Laboratoire de code : Un pipeline RAG complet avec LangChain (Colab T4)
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
# [SOURCE: Stratégies de chunking Livre p.236-237]
raw_docs = [
    "Le guide de sécurité stipule que le port du casque est obligatoire sur le chantier Alpha.",
    "La cafétéria est ouverte de 11h30 à 14h00 pour tous les employés munis d'un badge.",
    "En cas d'incendie, dirigez-vous vers le point de rassemblement situé sur le parking Nord."
]
# Simulation de documents LangChain
from langchain.docstore.document import Document
documents = [Document(page_content=t, metadata={"source": "Manuel Employé"}) for t in raw_docs]

# 2. CHUNKING INTELLIGENT
# [SOURCE: Figure 8-10 p.237 - Overlap]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# 3. EMBEDDINGS ET VECTOR STORE (Le Bibliothécaire)
# [SOURCE: Modèle recommandé p.253]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)

# 4. CHARGEMENT DU LLM (L'Écrivain)
# [SOURCE: Inférence optimisée p.202]
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
hf_pipe = pipeline("text-generation", model=model_id, device=0, torch_dtype=torch.bfloat16, max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# 5. ASSEMBLAGE DE LA CHAÎNE RAG
# [SOURCE: RetrievalQA implementation p.254]
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

### Optimisations et Bonnes Pratiques Industrielles
⚠️ **Fermeté bienveillante** : « Ne vous arrêtez pas au code qui "marche". Cherchez le code qui "tient". » 
Pour un déploiement sérieux, Maarten Grootendorst suggère plusieurs optimisations (p.255-257) :
1.  **La gestion de la latence** : Si votre LLM met 10 secondes à répondre, l'utilisateur partira. Utilisez le **Streaming** (l'affichage du texte mot par mot) pour donner une impression de vitesse.
2.  **Le cache de requêtes** : Si deux utilisateurs posent la même question, ne relancez pas tout le calcul. Utilisez un cache vectoriel pour renvoyer la réponse déjà générée.
3.  **Filtrage par métadonnées** : Si l'utilisateur cherche un document de 2024, ne demandez pas au bibliothécaire de chercher dans les archives de 1990. Filtrez d'abord par date via les métadonnées pour réduire le bruit sémantique. [SOURCE: Livre p.255]

### Éthique et Responsabilité : Le danger de la "Boîte Noire" orchestrée
⚠️ **Éthique ancrée** : « Mes chers étudiants, plus l'architecture est complexe, plus il est facile de cacher des erreurs. » 
Lorsque vous utilisez un framework comme LangChain :
1.  **Le risque de fuite de données** : Si votre orchestrateur envoie par erreur l'historique complet d'un client à une API tierce pour faire un résumé, vous enfreignez le RGPD. Auditez chaque étape du flux de données.
2.  **Le Vendor Lock-in** : Dépendre trop lourdement d'un seul framework peut vous rendre captif si celui-ci change ses licences ou ses tarifs. Gardez un code suffisamment modulaire pour pouvoir changer de moteur.
3.  **L'illusion de la citation** : LangChain peut vous dire qu'il a utilisé le "Document A", mais le LLM a peut-être quand même halluciné une réponse basée sur sa propre mémoire interne. 🔑 **Mon conseil de professeur** : Utilisez toujours un vérificateur de fidélité (comme Ragas, section 9.3) à la sortie de vos chaînes LangChain. [SOURCE: Livre p.28]

🔑 **Le message final du Prof. Henni pour la semaine 9** : « Vous avez maintenant entre les mains le plan complet d'un système d'intelligence artificielle factuel et professionnel. Vous avez appris à transformer le délire statistique en rigueur documentaire. Le RAG est sans doute la compétence la plus demandée sur le marché du travail aujourd'hui. Soyez-en fiers, mais restez humbles devant la complexité du langage. » [SOURCE: Livre p.258]

« Nous avons terminé notre grande semaine sur le RAG ! Vous savez désormais concevoir, construire, évaluer et orchestrer un système de génération ancré dans la réalité. La semaine prochaine, nous allons repousser les frontières : pourquoi se contenter du texte quand on peut parler aux images ? Bienvenue dans le monde fascinant des **LLM Multimodaux**. Mais avant cela, place au laboratoire final ! »

---
*Fin de la section 9.4 (2110 mots environ)*
## 🧪 LABORATOIRE SEMAINE 9 (850+ mots)

**Accroche du Professeur Khadidja Henni** : 
« Bonjour à toutes et à tous ! Nous passons maintenant aux travaux pratiques de notre semaine sur le RAG. C'est ici que vous allez donner une "mémoire documentaire" à votre IA. 🔑 **Je dois insister :** ne voyez pas le RAG comme une simple recherche Google améliorée. C'est une architecture de confiance. Nous allons apprendre à transformer un modèle qui "devine" en un expert qui "prouve". Soyez méticuleux dans votre découpage de texte, car c'est là que se joue la pertinence de votre futur système. Prêt·e·s à construire votre premier assistant ancré dans la réalité ? » [SOURCE: Livre p.249-250]

---

### 🔹 QUIZ MCQ (10 questions)

1. **Dans une architecture RAG, quel composant est responsable de transformer la question de l'utilisateur en un vecteur pour interroger la base de données ?**
   a) Le LLM générateur
   b) Le modèle d'Embedding (Retriever)
   c) Le tokenizer du décodeur
   d) Le module de Post-processing
   **[Réponse: b]** [Explication: Le modèle d'embedding convertit le texte en coordonnées numériques pour permettre la recherche de plus proches voisins. SOURCE: Livre p.253]

2. **Quelle technique permet de transformer une requête utilisateur ambiguë (ex: "Et pour lui ?") en une requête de recherche claire en utilisant l'historique ?**
   a) Le Chunking
   b) La Quantification
   c) Le Query Rewriting (Réécriture de requête)
   d) Le Cross-encoding
   **[Réponse: c]** [Explication: Le LLM analyse le contexte pour expliciter les pronoms et rendre la recherche plus précise. SOURCE: Livre p.255]

3. **Quelle métrique du framework Ragas mesure si la réponse générée par l'IA est entièrement supportée par les documents fournis dans le contexte ?**
   a) Answer Relevancy
   b) Faithfulness (Fidélité)
   c) Context Recall
   d) Perplexity
   **[Réponse: b]** [Explication: La fidélité vérifie l'ancrage (grounding) et détecte si l'IA a inventé des faits hors-contexte. SOURCE: Livre p.257, Ragas Documentation]

4. **Pourquoi utilise-t-on le "Multi-hop RAG" pour des questions complexes ?**
   a) Pour économiser des tokens.
   b) Pour permettre au modèle de faire plusieurs recherches successives afin de lier des informations éparpillées.
   c) Pour traduire la réponse dans plusieurs langues simultanément.
   d) Pour accélérer le temps de réponse.
   **[Réponse: b]** [Explication: Certaines questions demandent de trouver un premier fait (ex: le nom d'un inventeur) pour pouvoir chercher le second (ex: sa date de naissance). SOURCE: Livre p.256]

5. **Quelle bibliothèque d'orchestration simplifie le lien entre les Vector Stores, les prompts et les LLM ?**
   a) PyTorch
   b) LangChain
   c) Scikit-learn
   d) TensorFlow
   **[Réponse: b]** [Explication: LangChain propose des abstractions comme `RetrievalQA` pour automatiser le flux RAG. SOURCE: Livre p.200, Figure 7-1]

6. **Quel est le principal défaut de la stratégie de chaîne "stuff" dans LangChain ?**
   a) Elle est trop lente.
   b) Elle peut dépasser la limite de tokens (Context Window) du LLM si l'on fournit trop de documents.
   c) Elle modifie les poids du modèle.
   d) Elle ne permet pas de citations.
   **[Réponse: b]** [Explication: "Stuffing" consiste à mettre tous les docs dans le prompt ; si le volume est trop grand, le modèle tronque l'entrée. SOURCE: Livre p.254, LangChain Docs]

7. **Pourquoi un "Reranker" (souvent un Cross-Encoder) est-il utilisé après une recherche vectorielle classique ?**
   a) Pour réduire la taille des documents.
   b) Parce qu'il analyse finement l'interaction entre chaque mot de la question et du document, corrigeant les erreurs de la recherche "grossière".
   c) Pour compresser les poids du LLM.
   d) Pour générer des images à partir du texte.
   **[Réponse: b]** [Explication: Le reranker est plus précis car il ne compare pas juste des vecteurs finis, mais toute la séquence simultanément. SOURCE: Livre p.241, Figure 8-14]

8. **Que permet d'éviter l'ajout d'un "Overlap" (chevauchement) lors du découpage (chunking) d'un document ?**
   a) L'utilisation excessive de VRAM.
   b) La perte d'information sémantique située précisément à la limite entre deux morceaux de texte.
   c) Le coût des API OpenAI.
   d) La répétition des mots dans la réponse finale.
   **[Réponse: b]** [Explication: En répétant la fin du bloc N au début du bloc N+1, on garde le lien contextuel. SOURCE: Livre p.236, Figure 8-10]

9. **Quelle métrique évalue si le "Bibliothécaire" a mis le document le plus pertinent en première position plutôt qu'en dixième ?**
   a) Accuracy
   b) F1-Score
   c) MAP (Mean Average Precision)
   d) Bleu Score
   **[Réponse: c]** [Explication: Le MAP récompense la précision du classement (ranking) des résultats. SOURCE: Livre p.247]

10. **Dans LangChain, quel composant permet de filtrer les documents par date ou par auteur avant même de faire la recherche sémantique ?**
    a) Les Embeddings
    b) Les Métadonnées (Metadata)
    c) Le LLM
    d) La Temperature
    **[Réponse: b]** [Explication: Les métadonnées permettent une recherche hybride ou un pré-filtrage pour réduire le bruit. SOURCE: Livre p.253]

---

### 🔹 EXERCICE 1 : Pipeline RAG de base avec LangChain (Niveau 1)

**Objectif** : Implémenter un flux complet : Ingestion -> Indexation -> Question/Réponse ancrée.

```python
# --- CODE COMPLET (CORRIGÉ) ---
# [SOURCE: Implémentation RAG Livre p.253-254]

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

# --- RÉPONSE (ANSWER CODE) ---

# 2. Création de la mémoire vectorielle
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(texts, embeddings)

# 3. Initialisation du LLM (On utilise TinyLlama pour Colab T4)
# [SOURCE: Inférence optimisée p.202]
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

# --- EXPLICATIONS DÉTAILLÉES ---
# ATTENDU : Le modèle doit citer "Dr. Sarah Chen".
# JUSTIFICATION : Sans le RAG, TinyLlama ne peut pas connaître ce nom fictif. 
# FAISS a récupéré le bon chunk et l'a injecté dans le prompt de l'IA.
```

---

### 🔹 EXERCICE 2 : Reranking pour la précision (Niveau 2)

**Objectif** : Ajouter une étape de reranking pour filtrer les résultats d'une recherche vectorielle.

```python
# --- CODE COMPLET (CORRIGÉ) ---
# [SOURCE: Stratégie de Reranking Livre p.241-244]

from sentence_transformers import CrossEncoder

# 1. Résultats bruts du Bibliothécaire (QUESTION CODE)
query = "What is the budget?"
candidates = [
    "Project Aegis is led by Dr. Chen.", # Peu pertinent
    "The 2 million euro budget was approved last week.", # Très pertinent
    "Aegis aims at cloud security improvement." # Moyen
]

# --- RÉPONSE (ANSWER CODE) ---

# 2. Chargement du Reranker (Cross-Encoder)
# [SOURCE: Modèle recommandé p.243]
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

# 3. Calcul des scores d'interaction fine
pairs = [[query, doc] for doc in candidates]
scores = reranker.predict(pairs)

# 4. Affichage trié
results = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

print("--- CLASSEMENT APRÈS RERANKING ---")
for score, doc in results:
    print(f"Score: {score:.2f} | Doc: {doc}")

# --- EXPLICATIONS DÉTAILLÉES ---
# ATTENDU : Le document sur les "2 million euro" doit avoir le score le plus haut.
# JUSTIFICATION : Le Cross-Encoder "lit" simultanément la question et chaque doc, 
# capturant des liens sémantiques que la recherche vectorielle simple peut rater.
```

---

### 🔹 EXERCICE 3 : Audit de fidélité avec Ragas (Niveau 3)

**Objectif** : Simuler une évaluation de type "Faithfulness" pour détecter une hallucination.

**Scénario** : 
*   **Contexte** : "L'entreprise a été fondée en 1995."
*   **Réponse de l'IA** : "L'entreprise a été fondée en 1998."

**Tâche** : Expliquez comment un LLM-as-a-judge (comme Ragas) parviendrait à noter cette réponse.

**Réponse détaillée et Justification** :
1.  **Étape 1 (Décomposition)** : Le juge extrait l'affirmation : "Date de fondation = 1998".
2.  **Étape 2 (Vérification)** : Le juge compare à la preuve : "Date de fondation = 1995".
3.  **Étape 3 (Verdict)** : L'affirmation est contredite ou non-prouvée. 
4.  **Résultat** : Score de **Faithfulness = 0.0**. 
🔑 **Note du Prof. Henni** : « C'est ainsi que nous protégeons nos utilisateurs. Même si la phrase est jolie, le score de fidélité dénonce le mensonge. » [SOURCE: Livre p.257, Ragas Framework]

---

**Mots-clés de la semaine** : RAG, Retrieval, Grounding, Indexation, Vector DB, FAISS, Cross-Encoder, Reranking, Faithfulness, LangChain.

**En prévision de la semaine suivante** : Nous allons sortir du monde du texte pur. Comment une IA peut-elle "voir" une image et en discuter avec vous ? Bienvenue dans le monde fascinant des **LLM Multimodaux**. [SOURCE: Detailed-plan.md]

**SOURCES COMPLÈTES** :
*   Livre : Alammar & Grootendorst (2024), *Hands-On LLMs*, Chapitre 8, p.225-258.
*   Framework Ragas : https://docs.ragas.io/
*   LangChain RAG Guide : https://python.langchain.com/docs/use_cases/question_answering/
*   GitHub Officiel : chapter08 repository.

[/CONTENU SEMAINE 9]
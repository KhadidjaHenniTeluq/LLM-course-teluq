---
title: "9.3 Évaluation RAG"
weight: 4
---
{{< katex />}}


## Le juge de paix de l'IA : Pourquoi l'intuition ne suffit plus
Bonjour à toutes et à tous ! Nous abordons aujourd'hui la section la plus rigoureuse, et peut-être la plus vitale, de notre parcours sur le RAG. Imaginez que vous ayez construit un magnifique moteur de recherche et que vous l'ayez couplé à un LLM puissant. Tout semble fonctionner. Mais comment pouvez-vous garantir à votre client, ou à votre direction, que l'IA ne va pas inventer un fait crucial demain matin à 9h ? Comment prouver que votre système est "meilleur" que celui de la concurrence ? 

> [!IMPORTANT]
🔑 **Je dois insister :** en ingénierie des LLM, ce qui ne se mesure pas n'existe pas. Aujourd'hui, nous allons apprendre à devenir des auditeurs de la vérité. Respirez, nous allons transformer la qualité en chiffres.

L'évaluation d'un système RAG est double. Vous devez juger la capacité du "Bibliothécaire" à trouver les bons livres (Évaluation du Retrieval) et la capacité de l' "Écrivain" à synthétiser sans mentir (Évaluation de la Génération). Si vous ne mesurez que le résultat final, vous ne saurez jamais quel composant corriger en cas d'erreur. C'est ce que nous appelons l'évaluation par composantes.

---
## Évaluer le "Bibliothécaire" : La science du Retrieval
Pour juger la recherche, nous utilisons les standards de l'*Information Retrieval* (IR). Voici les concepts fondamentaux, illustrés par une série de figures essentielles :

### 1. Le banc d'essai
Comme l'explique la **Figure 9-8 : Composantes de l'évaluation**, vous ne pouvez pas évaluer un système de recherche dans le vide. Vous avez besoin d'une "Test Suite" composée :
*   D'un archive de textes (votre base de connaissances).
*   D'un ensemble de requêtes types.
*   De **Relevance Judgments** : Pour chaque question, un humain (ou un expert) a marqué quels documents sont "Vrais" (pertinents) et lesquels sont "Faux". 

{{< bookfig src="187.png" week="09" >}}

> [!TIP]
🔑 **C'est votre "Golden Dataset"**. Sans cette vérité terrain, vous naviguez à l'aveugle.


### 2. La comparaison de systèmes
La **Figure 9-9** montre comment nous envoyons la même requête à deux systèmes différents pour comparer leurs résultats. Mais la simple présence du bon document ne suffit pas. 

{{< bookfig src="188.png" week="09" >}}

La **Figure 9-10** nous montre que l'ordre compte : si le système 1 met le bon document en position 1, il est bien supérieur au système 2 qui le met en position 3. 

{{< bookfig src="189.png" week="09" >}}

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent qu'un moteur de recherche est bon s'il trouve l'info "quelque part". C'est faux. L'utilisateur clique sur le premier lien. La position est une question de survie pour votre application.


### 3. La métrique reine : MAP (Mean Average Precision)
C'est ici que les mathématiques rencontrent la stratégie. Le calcul du **MAP** est détaillé à travers les **Figures 9-11 à 9-14** .

*   **Calcul pour une requête (Figure 9-11)** : Si le premier résultat est le bon, votre précision à la position 1 ($P@1$) est de 1.0 ($100$%).

{{< bookfig src="191.png" week="09" >}}

*   **La pénalité du retard (Figure 9-12)** : Si vous placez des documents inutiles avant la réponse, votre score chute. Par exemple, si le bon document est en 3ème position, votre précision pour ce document est de $1/3 = 0.33$.

{{< bookfig src="192.png" week="09" >}}

*   **La synthèse (Figures 9-13 et 9-14)** : Pour obtenir le MAP, on calcule la précision moyenne pour chaque question, puis on fait la moyenne de ces moyennes sur tout le banc d'essai. 


{{< bookfig src="193.png" week="09" >}}
{{< bookfig src="194.png" week="09" >}}

> [!NOTE]
🔑 **Je dois insister :** Le MAP est une métrique de "classement". Elle récompense les systèmes qui ont l'audace et la précision de mettre la vérité tout en haut de la pile.

---
## Évaluer l' "Écrivain" : La Triade du RAG
Une fois que nous avons les bons documents, comment juger la réponse générée ? Les métriques classiques comme *BLEU* (Bilingual Evaluation Understudy) ou *ROUGE* (Recall-Oriented Understudy for Gisting Evaluation), utilisées en traduction, sont ici inutiles : elles comparent les mots exacts, mais ne comprennent pas si l'IA a inventé une date ou un nom.

Nous utilisons aujourd'hui le framework **Ragas** (Retrieval Augmented Generation Assessment), qui repose sur trois piliers que j'appelle la "Triade de la Confiance" :

### 1. La Fidélité (Faithfulness)
C'est la mesure de l' **Ancrage**. Est-ce que chaque affirmation de la réponse de l'IA est présente dans les documents récupérés ? 

### 2. La Pertinence de la réponse (Answer Relevancy)
Est-ce que l'IA a vraiment répondu à la question de l'utilisateur ? Une réponse peut être 100% fidèle aux sources mais totalement hors-sujet. 
*   *Exemple* : L'utilisateur demande "Comment résilier mon abonnement ?" et l'IA répond "Votre abonnement a été souscrit le 12 janvier." C'est vrai, mais c'est inutile.

### 3. La Précision du contexte (Context Precision)
Ici, on juge à nouveau le bibliothécaire mais à travers le regard de l'écrivain : est-ce que les documents fournis étaient vraiment nécessaires pour répondre ? Plus le contexte est "propre" (sans documents parasites), plus le score est élevé.

---
## LLM-as-a-judge : Quand l'IA devient professeur
Comment calculer ces scores de fidélité ou de pertinence de manière automatique ?

Nous utilisons un concept révolutionnaire : **LLM-as-a-judge**. 

Nous utilisons un modèle très "intelligent" et neutre (comme GPT-4 ou Claude 3 Opus) pour noter la sortie d'un modèle plus petit (comme Phi-3). 
1.  On donne au juge le document source, la question et la réponse de l'IA.
2.  On lui demande de décomposer la réponse en affirmations individuelles.
3.  Pour chaque affirmation, le juge vérifie si elle est "prouvée" par le document. 
4.  Le score final est le ratio d'affirmations prouvées. 


🔑 **Mon intuition :** C'est comme si je demandais à un doctorant de corriger les copies des étudiants de première année en suivant une grille de notation très stricte. C'est rapide, scalable et souvent plus cohérent qu'une évaluation humaine fatiguée.

---
## Laboratoire de code : Évaluation avec Ragas (Colab T4)
Voici comment mettre en place une évaluation scientifique de votre pipeline RAG. Nous allons simuler un petit dataset de test.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install ragas datasets transformers

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# 1. PRÉPARATION DU DATASET DE TEST (Simulé)
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

# 3. EXÉCUTION DE L'ÉVALUATION
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy]
)

print("--- SCORE D'AUDIT RAG ---")
print(f"Fidélité (Faithfulness) : {results['faithfulness']:.2f}")
print(f"Pertinence (Answer Relevancy) : {results['answer_relevancy']:.2f}")

# ATTENDU : Faithfulness proche de 1.0 car 1998 est dans le contexte.
```

> [!WARNING]
⚠️ Si vous obtenez un score de Faithfulness de 0.0, cela signifie que votre modèle a totalement ignoré le contexte pour inventer sa propre réponse. C'est le signal d'alarme rouge : vous devez revoir votre prompt ([**section 9.1**]({{< relref "section-9-1.md" >}}#aug-prompt)).

---
## Le danger de l'optimisation aveugle : La loi de *Goodhart*

> [!CAUTION]
⚠️ Mes chers étudiants, un chiffre n'est pas la réalité.

Il existe un principe célèbre en économie et en IA : la **Loi de Goodhart :** 

>> [!IMPORTANT] 
*"Lorsqu'une mesure devient un objectif, elle cesse d'être une bonne mesure."*

1.  **Le risque du "Gaming"** : Si vous donnez des primes à vos ingénieurs pour augmenter le score de "Faithfulness", ils pourraient forcer le modèle à répondre de manière extrêmement courte et robotique ("Oui", "Non", "1998"). Le score sera parfait, mais l'utilité pour l'utilisateur sera nulle.
2.  **Le biais du Juge** : N'oubliez pas que votre juge est lui-même un LLM. Il a ses propres biais. Il peut préférer les réponses longues et polies, même si elles sont moins précises. 

> [!TIP]
🔑 **Mon conseil** : Faites toujours une "contre-vérification" humaine sur 5% de vos évaluations automatiques pour vous assurer que le juge ne fait pas de favoritisme statistique.

---
## Synthèse des métriques
Pour conclure, rappelez-vous ce tableau de bord que vous devez présenter à chaque déploiement :

| Composant évalué | Métrique Clé | Ce que ça mesure vraiment |
| :--- | :--- | :--- |
| **Le Bibliothécaire (R)** | **MAP** | Est-ce que l'info est en haut de la liste ? |
| **L'Écrivain (G)** | **Faithfulness** | Est-ce que l'IA a inventé des choses ? |
| **L'Écrivain (G)** | **Answer Relevancy** | Est-ce que l'IA a répondu à la question ? |
| **L'Expérience Utilisateur** | **Latence / Coût** | Est-ce que c'est trop lent ou trop cher ? |

> [!IMPORTANT]
🔑 **Mon message** : L'évaluation est le moment où vous passez du statut de bidouilleur à celui d'ingénieur. C'est ingrat, c'est long, c'est parfois frustrant quand les scores chutent, mais c'est le seul rempart qui protège vos utilisateurs de l'erreur. Soyez d'une exigence absolue avec vos chiffres.

---
Vous savez maintenant comment mesurer la qualité de votre moteur. Vous avez les outils pour prouver que votre système est fiable. Dans la dernière section ➡️ de cette semaine, nous allons voir comment assembler tout cela de manière élégante et industrielle grâce au framework **LangChain**.
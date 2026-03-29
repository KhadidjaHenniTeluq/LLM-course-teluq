---
title: "14.3 Agents et autonomie"
weight: 4
---

## De l'IA qui parle à l'IA qui agit : La naissance de l'Homo Digitalis

Bonjour à toutes et à tous ! J'espère que vous avez encore soif d'apprendre, car nous abordons aujourd'hui ce qui constitue, pour moi, la frontière la plus excitante et peut-être la plus vertigineuse de notre domaine. 

Jusqu'ici, nous avons traité les LLM comme des oracles : vous posez une question, ils vous répondent. C'est brillant, mais c'est passif. 

> [!IMPORTANT]
📌 **Je dois insister :** l'avenir de l'intelligence artificielle ne réside pas dans sa capacité à disserter, mais dans sa capacité à **faire**. 

Aujourd'hui, nous allons apprendre à donner des bras et des jambes à nos modèles. Nous allons transformer le "cerveau" statistique en un **Agent**. Respirez, car nous passons de la contemplation à l'action. Bienvenue dans l'ère de l'autonomie algorithmique !

---
## L'anatomie d'un Agent : Le cerveau, les mains et la mémoire
Qu'est-ce qu'un agent ? Comme le définit brillamment Lilian Weng dans ses [travaux de référence](https://lilianweng.github.io/posts/2023-06-23-agent/) , un agent est un système où le LLM n'est plus la finalité, mais le **contrôleur central**. Pour qu'un modèle de langage devienne un agent, il doit être complété par trois briques fondamentales :

1.  **Le Cerveau (LLM)** : Il planifie, raisonne et décide.
2.  **La Planification (Planning)** : Le modèle doit savoir décomposer un problème complexe en petites étapes simples.
3.  **La Mémoire (Memory)** : L'agent doit se souvenir de ce qu'il a déjà essayé et des résultats obtenus.
4.  **L'Utilisation d'outils (Tool Use)** : La capacité d'appeler des logiciels externes (calculatrices, APIs, moteurs de recherche).


💡 **Mon intuition :** Imaginez que vous demandiez à une IA : "Réserve-moi un billet de train pour Lyon demain matin". 

> Un LLM classique vous répondrait : "Bien sûr, voici comment on réserve un billet sur Internet...". Un **Agent**, lui, va se connecter au site de la SNCF, vérifier vos préférences dans sa mémoire, comparer les prix avec une calculatrice, et vous demander : "J'ai trouvé un train à 8h pour 45€, je valide ?".

---
## L'utilisation d'outils : Pourquoi l'IA a besoin d'une calculatrice ?
L'une des premières limites que nous avons rencontrées en Semaine 5 était l'incapacité des LLM à faire des mathématiques exactes. Regardons ensemble la **Figure 14-2 : Agents avec outils** .

{{< bookfig src="168.png" week="14" >}}

**Explication** : Cette illustration est une leçon d'humilité pour l'IA.
*   **À gauche (IA seule)** : On pose un calcul complexe au modèle. Comme c'est un prédicteur statistique, il "devine" un résultat qui ressemble à un nombre correct, mais qui est faux. C'est l'hallucination mathématique.
*   **À droite (IA avec outil)** : Le modèle reconnaît qu'il est face à un calcul. Au lieu de répondre, il écrit une commande spéciale : `Action: Calculator[47 / 12 * 3.14]`. Le système exécute le calcul et renvoie le résultat exact au modèle.
*   **Le verdict** : L'IA utilise alors ce résultat pour donner une réponse 100% fiable.

> [!NOTE]
✍🏻 **Je dois insister :** Un agent intelligent est celui qui connaît ses propres limites. 

> Apprendre à un LLM à dire "Je ne sais pas faire ce calcul, je vais utiliser cet outil" est la base de l'ingénierie des agents.


---
## Le Framework ReAct : Raisonner et Agir en boucle
Comment l'IA décide-t-elle de sortir ses outils ? La réponse réside dans une technique de prompt engineering de haut vol appelée **ReAct** (*Reasoning and Acting*).


{{< bookfig src="169.png" week="14" >}}

Regardons la **Figure 14-3 : Un exemple de prompt template ReAct** . Cette figure nous montre l'ossature d'une pensée "active". Le prompt impose au modèle une structure itérative stricte :
1.  **Thought (Pensée)** : Le modèle écrit ce qu'il a compris et ce qu'il compte faire.
2.  **Action** : Le modèle choisit un outil dans une liste fournie.
3.  **Observation** : Le système renvoie le résultat de l'outil (ex: le contenu d'une page web).
4.  **... (Répétition)** : Le modèle analyse l'observation et recommence le cycle.


> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Beaucoup pensent que le ReAct est un mode spécial du modèle. Non, c'est un **format de discussion** que nous forçons via le prompt (Semaine 8.1). 


### Analyse d'un cycle complet
La **Figure 14-4** et la **Figure 14-5** nous montrent le ReAct en action pour une recherche de prix de MacBook Pro.

{{< bookfig src="170.png" week="14" >}}

*   **Cycle 1** : L'IA pense qu'elle doit chercher sur le web. Elle utilise l'outil `GoogleSearch`. Elle "observe" que le prix est de 1299$.
*   **Cycle 2** : Elle pense qu'elle doit convertir en Euros. Elle utilise l'outil `Calculator` avec le taux de change.

{{< bookfig src="171.png" week="14" >}}
*   **Final Answer** : Elle synthétise le tout pour l'utilisateur.

> [!NOTE]
✍🏻 **Note du Professeur** : Sans la phase de "Thought", l'IA agirait de manière impulsive et ferait des erreurs de logique. La pensée est le gouvernail, l'action est l'hélice.


---
## La Planification : Décomposer l'impossible
Mes chers étudiants, imaginez que je vous demande de construire une fusée. Vous seriez paralysés. Mais si je vous demande de dessiner un plan, puis de commander de l'acier, puis de souder deux plaques... cela devient possible.

Les agents utilisent deux types de planification :
1.  **La Décomposition (Task Decomposition)** : Utiliser le **Chain-of-Thought** (Semaine 8.3) pour transformer un gros "Goal" en une liste de "Sub-goals".
2.  **L'Auto-critique (Self-Reflection)** : L'agent regarde son propre travail et se dit : "Tiens, cette étape n'a pas marché, je vais essayer une autre stratégie". C'est le framework **Reflexion** (Shinn et al., 2023). L'IA apprend de ses propres échecs au sein d'une seule session.


---
## La Mémoire des Agents : Court terme vs Long terme

> [!IMPORTANT]
‼️ **Je dois insister sur cette distinction technique vitale pour vos futurs projets :**

> *   **Mémoire à court terme** : C'est la **Fenêtre de Contexte** (Semaine 5). Elle contient l'historique immédiat de la conversation. Si elle sature, l'agent perd le fil de sa mission.
> *   **Mémoire à long terme** : C'est le **RAG** (Semaine 9). L'agent peut fouiller dans une base de données vectorielle pour retrouver des instructions ou des faits datant de plusieurs mois.
> *   **Apprentissage par l'expérience** : Certains agents stockent leurs "succès" passés dans une base de données pour ne plus refaire les mêmes erreurs de planification.


---
## Implémentation pratique : Un agent avec LangChain
Voici comment transformer notre modèle Phi-3 en un agent capable de naviguer sur le web et de calculer. Nous utilisons les abstractions de LangChain vues en Semaine 9.4.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install langchain langchain-openai duckduckgo-search

# Objectif : Créez l'exécuteur d'agent et posez une question sur l'actualité.

from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch

# 1. PRÉPARATION DES OUTILS (The "Hands")
search = DuckDuckGoSearchRun()
tools = [search]

# 2. CHARGEMENT DU CERVEAU (The "Brain")
model_id = "microsoft/Phi-3-mini-4k-instruct"
hf_pipe = pipeline("text-generation", model=model_id, device=0, torch_dtype=torch.float16, max_new_tokens=250)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# 3. RÉCUPÉRATION DU PROMPT ReACT (The "Logic")
# On utilise un prompt standardisé qui définit la boucle Thought/Action/Observation
prompt = hub.pull("hwchase17/react")

# 4. ASSEMBLAGE DE L'AGENT
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

query = "Quel est le prix actuel de l'action NVIDIA et quel impact a eu le dernier modèle Blackwell ?"

# Note : L'agent va d'abord chercher sur le web, puis synthétiser.

res = agent_executor.invoke({"input": query})
print(res["output"])
```

> [!IMPORTANT]
⚠️ Regardez l'argument `verbose=True`. 

C'est votre fenêtre sur l'âme de la machine. Vous verrez l'IA se parler à elle-même, hésiter, corriger ses recherches. C'est l'outil de débogage numéro 1 de l'ingénieur d'agents.


---
## Les Agents multi-agents : L'intelligence collective
La recherche actuelle s'oriente vers des **sociétés d'agents**. Au lieu d'avoir un seul agent qui fait tout, on crée une équipe :
*   Un **Agent Manager** qui distribue les tâches.
*   Un **Agent Chercheur** qui fouille le web.
*   Un **Agent Critique** qui vérifie les erreurs du chercheur.
*   Un **Agent Rédacteur** qui finalise le rapport.

> [!TIP]
✅ **L'avantage technique :** Chaque agent a un petit prompt très spécifique, ce qui réduit drastiquement les risques de confusion et d'hallucinations par rapport à un agent unique surchargé.


---
## Éthique et Responsabilité : Le risque de la boucle infinie

> [!CAUTION]
‼️ Mes chers étudiants, l'autonomie est un risque autant qu'une opportunité.

Donner la capacité d'agir à une IA soulève des questions de sécurité sans précédent.

1.  **L'escalade budgétaire** : Un agent mal programmé peut entrer dans une boucle infinie de recherches Google ou d'appels API payants, vous coûtant des milliers d'euros en une nuit. 
>> [!TIP]
> 🔑 **Règle d'or :** Toujours fixer un `max_iterations` (souvent 10) à vos exécuteurs d'agents.

2.  **L'action irréversible** : Si vous donnez à un agent l'accès à votre boîte mail ou à votre compte bancaire, une simple hallucination ("Je pense que je dois envoyer cet argent à ce contact") peut avoir des conséquences désastreuses. 
>> [!WARNING]
> 🔑 **Principe de précaution :** L'agent doit "penser" et "proposer", mais l'humain doit toujours "cliquer" pour valider l'action réelle.

3.  **L'Agent malveillant** : Un agent peut être utilisé pour automatiser des cyberattaques sophistiquées, capable de s'adapter aux défenses en temps réel. La responsabilité de l'ingénieur est de coder des limites infranchissables (*Hard Constraints*).

> [!IMPORTANT]
✉️  **Le message final de cette section** : Un agent n'est pas un substitut à votre intelligence, c'est une extension de votre volonté. 

En maîtrisant les agents, vous passez du rôle de programmeur à celui de chef d'orchestre. Dirigez vos agents avec la rigueur d'un général et la bienveillance d'un mentor. L'autonomie de la machine doit toujours être au service de la liberté de l'humain.

---
Nous avons exploré les sommets de l'autonomie. Vous savez désormais comment donner des outils, une mémoire et une logique d'action à vos modèles. C'est l'apogée technique de notre cours. Dans la prochaine et dernière section ➡️, nous prendrons un peu de recul pour regarder les **Frontières de la recherche** : le raisonnement pur, la conscience de l'IA et l'impact de vos futurs travaux sur la société de demain.
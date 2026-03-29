---
title: "Laboratoire "
weight: 6
---


Bonjour à toutes et à tous ! Nous voici au dernier laboratoire technique de notre cursus. C'est un moment de synthèse, mais aussi d'audace. 

Aujourd'hui, nous n'allons pas seulement faire du code ; nous allons construire un **Agent**, une entité capable de décider et d'agir. 

> [!IMPORTANT]
📌 **Je dois insister :** l'IA autonome est la brique finale de votre arsenal. Elle demande une maîtrise parfaite du prompt, de l'outil et du raisonnement. 

Soyez les architectes vigilants de cette autonomie. Prêt·e·s pour le grand final ? C'est parti !

---
## 🔹 EXERCICE 1 : L'Art du Custom Tool (Outil Personnalisé)

**Objectif** : Dans la section 14.3, vous avez vu comment utiliser des outils préexistants (ex: recherche web). L'objectif ici est de créer votre propre fonction Python métier et de l'équiper sur un Agent ReAct.

```python
# --- QUESTION (STRUCTURE) ---
# Tâche : Utiliser le décorateur @tool de LangChain pour fournir une calculatrice de frais de port.
# !pip install -q langchain langchain-community

from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain import hub
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch

# Configuration du modèle "Cerveau" (TinyLlama)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
hf_pipe = pipeline("text-generation", model=model_id, device=0, torch_dtype=torch.float16, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# --- Votre code ici ---
```

<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- RÉPONSE ---
# 1. Définition de l'outil personnalisé (The "Hands")
@tool
def calculate_shipping_cost(weight_kg: float) -> str:
    """Calcule le coût de livraison en fonction du poids en kg. 
    Le tarif de base est de 5€, plus 2€ par kilogramme."""
    cost = 5.0 + (2.0 * float(weight_kg))
    return f"Le coût de livraison exact est de {cost}€."

tools = [calculate_shipping_cost]

# 2. Récupération de la logique ReAct (The "Logic")
prompt = hub.pull("hwchase17/react")

# 3. Création de l'agent et de son exécuteur
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, # Indispensable pour voir le raisonnement 'Thought'
    handle_parsing_errors=True,
    max_iterations=3 # Sécurité contre les boucles infinies
)

# 4. Test (Désactivé pour éviter de charger le serveur ici)
req = "Bonjour, quel sera le coût de livraison pour une commande de 3.5 kg ?"
agent_executor.invoke({"input": req})
print("Agent configuré avec succès ! Il sait désormais calculer des frais de port.")

```

**Explications détaillées** :
*   **Résultats attendus** : L'agent lit le prompt, extrait le chiffre "3.5", invoque l'outil `calculate_shipping_cost` et vous restitue la réponse "12.0€".
*   **Justification technique** : Remarquez l'importance absolue de la *docstring* (`"""Calcule le coût..."""`). Le LLM lit cette documentation pour comprendre à quoi sert l'outil et comment lui passer des paramètres. Plus votre description est claire, plus l'agent est intelligent.
*   **La puissance du concept** : C'est le pont ultime entre l'inférence probabiliste (LLM) et le code déterministe de votre entreprise (fonctions Python existantes).

</details>


---

## 🔹 EXERCICE 2 : Analyse critique : Limites et Directions

**Objectif** : Évaluer les capacités d'un modèle face à une énigme de raisonnement pur et proposer une solution d'architecture.

**Problème** : "Marie a 3 frères. Chaque frère a 2 sœurs. Combien de sœurs a Marie ?"

**Tâche** :
1.  Expliquez pourquoi un LLM en mode "Système 1" (réponse directe) risque de répondre "6" ou "2".
2.  Proposez une solution basée sur les concepts de la Semaine 8.3 ou 14.3.

**Réponse détaillée et Justification** :
1.  **Analyse de l'échec** : Le modèle fait une corrélation statistique rapide (3 frères x 2 sœurs = 6). Il ne construit pas de modèle mental de la famille.
2.  **Solution d'architecture** : 
    *   **Prompting** : Utiliser le **Chain-of-Thought** ("Let's draw the family tree step by step"). 
    *   **Agentique** : Utiliser un agent capable d'écrire un petit code Python pour simuler la famille et compter les membres. 

> [!NOTE]
🧠 **Justification** : Le raisonnement n'est pas une intuition, c'est une déduction. Si le cerveau statistique échoue, on utilise le code informatique (outil déterministe) pour garantir la vérité.

---

## 🔹 EXERCICE 3 : Conception d'un pipeline "Post-Transformer"

**Objectif** : Concevoir une architecture pour un cas d'usage de résumé de vidéos de 4 heures (très long contexte).

**Tâche** : Justifiez l'utilisation d'une architecture de type **Mamba (SSM)** par rapport à un **Transformer** standard pour ce projet.

**Explications détaillées** :
1.  **Le problème du Transformer** : Une vidéo de 4 heures transcrite représente des centaines de milliers de tokens. Avec une complexité $O(L^2)$, la VRAM nécessaire dépasserait les 80 Go d'un H100.
2.  **L'avantage de Mamba** : Sa complexité est **linéaire** ($O(L)$). La mémoire utilisée reste **constante** quelle que soit la durée de la vidéo.
3.  **Choix technique** : On utiliserait Mamba pour "lire" le flux et maintenir un état interne résumé, puis un petit bloc d'Attention (Transformer) à la fin pour générer la synthèse finale.

> [!NOTE]
🏆 **Verdict** : Mamba est le gagnant pour le "Long-range context".

---
**Mots-clés de la semaine** : Synthèse, Agents, ReAct, Autonomie, Mamba, State Space Models (SSM), Multi-agents, Planification, Mémoire long terme, Contexte infini.

**Pour la semaine prochaine** : nous poserons nos outils pour une session de révision intensive. Nous passerons en revue les points de blocage et nous nous préparerons à l'examen final. Félicitations pour ce parcours !
---
title: "1.4 Définition et applications des LLM"
weight: 5
---

## L'édifice de l'IA moderne : Quand la taille change la nature
Bonjour à toutes et à tous ! Nous arrivons à la dernière étape de notre première semaine. Nous avons vu le moteur (l'Attention) et les pièces mécaniques (le Transformer). Maintenant, prenons du recul pour admirer l'édifice tout entier. 

> [!IMPORTANT]
📌 **Je dois insister :** un "Large Language Model" n'est pas simplement un petit modèle qui a grandi. C'est une technologie où le changement d'échelle a provoqué l'émergence de capacités que personne n'avait prédites. 

Aujourd'hui, nous allons définir ce qu'est réellement un LLM, comment on "élève" ces géants, et surtout, comment ils transforment notre société. Respirez, car nous passons de la mathématique à la vision globale.

---
## Une définition mouvante : Qu'est-ce que "Large" ?
Le terme "Large" dans LLM est un horizon qui recule sans cesse. En 2018, le modèle BERT-base avec ses **110 millions de paramètres** était considéré comme une prouesse technologique "large". Aujourd'hui, nous manipulons des modèles comme Llama-3-70B (70 milliards) ou GPT-4 (qui dépasserait le millier de milliards).

> [!NOTE]
🔑 **La distinction fondamentale :** La "largesse" ne se mesure pas qu'au nombre de neurones artificiels. Elle se définit par trois piliers :
1.  **Le volume de données** : On parle de téraoctets de texte (tout Wikipédia, des millions de livres, tout le code de GitHub, une part immense du web).
2.  **La puissance de calcul** : Des milliers de GPU tournant pendant des mois.
3.  **L'émergence** : C'est le point le plus fascinant. À partir d'un certain seuil de taille, le modèle commence à savoir faire des choses pour lesquelles il n'a jamais été entraîné, comme résoudre des énigmes logiques ou coder.

---
## La saga GPT : De la curiosité au séisme mondial
Pour comprendre où nous en sommes, nous devons suivre l'évolution de la lignée la plus célèbre, détaillée ci-dessous.

### 1. L'aube : GPT-1 et l'intuition du décodeur (Figure 1-20)

{{< bookfig src="28.png" week="01" >}}

**Explication** : GPT-1, sorti en 2018, ne possédait que 117 millions de paramètres. La figure montre une architecture "Decoder-only" simple. L'innovation ? C'était la preuve que l'on pouvait entraîner un modèle sans étiquettes humaines, simplement en lui demandant de prédire le mot suivant sur 7000 livres. C'était le passage de "l'IA de laboratoire" à "l'IA apprenante". 

### 2. La rupture : GPT-2 et le danger de la fluidité (Figure 1-21)

{{< bookfig src="29.png" week="01" >}}

**Explication** : En 2019, OpenAI passe à 1,5 milliard de paramètres. La figure illustre un saut d'échelle massif.

> [!NOTE]
✍🏻 **Je dois insister :** GPT-2 a été le premier modèle capable d'écrire des articles de presse si convaincants qu'OpenAI a d'abord refusé de le publier, craignant une vague massive de "Fake News". 

C'est là que la société a réalisé que l'IA pouvait désormais mimer la prose humaine à la perfection.

### 3. L'ère des géants : GPT-3 et le Zero-shot (Figure 1-21 suite)
Toujours sur la **Figure 1-21**, on voit l'explosion vers **175 milliards de paramètres** en 2020. 

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** On pense souvent que GPT-3 est juste "plus fort". En réalité, il a introduit le **In-context learning**. 

On n'a plus besoin de ré-entraîner le modèle pour lui apprendre une tâche ; il suffit de lui donner deux ou trois exemples dans le prompt (Few-shot) ou même aucun (Zero-shot) pour qu'il comprenne.

### 4. La révolution sociétale : ChatGPT et GPT-4

{{< bookfig src="30.png" week="01" >}}

**Explication de la Figure 1-22** : Cette figure montre l'introduction de l' **Instruction Tuning**. On ne se contente plus de prédire le web ; on apprend au modèle à être un "assistant". 

{{< bookfig src="198.png" week="01" >}}

**Explication de la Figure 1-23** : On arrive à GPT-4, qui devient multimodal (il voit des images). C'est la fin du modèle purement textuel.

---
## Le paradigme de l'entraînement : La naissance d'un esprit numérique
Mes chers étudiants, voici le secret de fabrication que vous devez graver dans votre mémoire. Un LLM ne naît pas "intelligent", il le devient en deux étapes non-négociables.

### Étape 1 : Le Pré-entraînement (Pretraining) - L'éducation sauvage
C'est la phase "biblique" de l'IA. Le modèle lit tout ce qui est numérisé.
*   **Objectif** : Apprendre la structure du monde et du langage. 
*   **État final** : Le **Base Model** (ou Foundation Model). 

> [!IMPORTANT]
⚠️ Un Base Model est un savant fou. 

> Si vous lui demandez "Quelle est la capitale de la France ?", il pourrait vous répondre "Et quelle est la capitale de l'Espagne ?" car il a appris que les listes de questions se suivent souvent. Il ne sait pas encore qu'il doit vous servir.

### Étape 2 : Le Réglage Fin (Fine-tuning) - L'école de la courtoisie
C'est ici que VOUS intervenez en tant qu'ingénieurs. On prend le géant et on l'entraîne sur un dataset beaucoup plus petit (quelques milliers d'exemples) de dialogues parfaits.
*   **SFT (Supervised Fine-Tuning)** : On lui montre des paires "Question -> Réponse idéale".
*   **RLHF (Reinforcement Learning from Human Feedback)** : On demande à des humains de noter ses réponses. L'IA apprend ce que nous préférons : la clarté, la politesse et la vérité.

---
## L'explosion de 2023 : La libération des modèles

{{< bookfig src="32.png" week="01" >}}

**Explication de la Figure 1-24** : Cette figure est sans doute la plus importante pour votre future carrière. Elle montre qu'en 2023, le monopole d'OpenAI s'est effondré. 
*   On voit l'arrivée de **Llama** (Meta), **Mistral**, **Falcon**.

> [!TIP]
✉️ **Le message de la figure** : Nous sommes passés de modèles fermés (Proprietary) à des modèles ouverts (Open Models) que vous pouvez faire tourner sur votre propre ordinateur. **C'est la démocratisation de la puissance de calcul**!

---
## Applications pratiques : Le couteau suisse universel
Pourquoi les entreprises s'arrachent-elles ces modèles ? Parce qu'un seul modèle peut remplacer dix logiciels différents.

**Tableau 1-2 : Panorama des applications industrielles des LLM**

| Domaine | Application Concrète | Ce que le LLM apporte |
| :--- | :--- | :--- |
| **Programmation** | Copilot, génération de fonctions | Gain de productivité de 40% pour les développeurs. |
| **Relation Client** | Chatbots de support niveau 1 | Réponse instantanée 24h/24 sans frustration. |
| **Droit & Finance** | Résumé de contrats de 200 pages | Extraction instantanée des clauses de risque. |
| **Médecine** | Aide au diagnostic, synthèse de dossiers | Analyse croisée de milliers de publications. |
| **Marketing** | Copywriting, création de slogans | Génération de 50 variantes en 3 secondes. |

---
## Éthique et Responsabilité : Les ombres du géant

> [!CAUTION]
‼️ Je ne serais pas une bonne enseignante si je ne vous montrais que le côté brillant de la médaille. Ces modèles sont des miroirs déformants de notre humanité.

Nous devons faire face à quatre défis éthiques majeurs :

1.  **Hallucinations** : Le modèle privilégie la fluidité sur la vérité. S'il ne connaît pas la réponse, sa nature statistique le pousse à inventer une réponse crédible. 
>> [!IMPORTANT]
>📌 **Je dois insister :** Ne faites jamais confiance à un LLM pour un fait médical ou juridique sans une source vérifiable (RAG, que nous verrons en Semaine 9).

2.  **Biais et Équité** : Si le web est sexiste ou raciste, le LLM le sera. Aligner un modèle est un combat permanent contre les préjugés enfouis dans les données.

3.  **Transparence** : Comment le modèle a-t-il pris sa décision ? Personne ne sait lire dans les 175 milliards de paramètres de GPT-3. C'est le problème de la "boîte noire".

4.  **Propriété Intellectuelle** : À qui appartiennent les données d'entraînement ? Les procès actuels entre artistes et entreprises d'IA vont redéfinir le droit d'auteur pour le siècle à venir.

---
## Limited Resources are All You Need : L'IA pour tous
Une note d'espoir pour conclure : *On n'a pas besoin d'être milliardaire pour utiliser ces technologies.*

> [!TIP]
💡 **L'astuce de l'expert** : Grâce à la **Quantification** (réduire la précision des nombres) et au **PEFT** (modifier seulement 0,1% du modèle), vous pouvez adapter un modèle surpuissant sur une simple carte graphique T4 comme celle de notre laboratoire. L'intelligence est désormais un bien commun.

---
## Synthèse finale
> [!IMPORTANT]
✉️ **Le message à retenir** : Mes chers étudiants, vous avez maintenant les clés de la forteresse. Vous savez d'où vient l'IA (section 1.1), comment elle a appris à ne plus oublier (section 1.2), quel est son cœur atomique (section 1.3) et comment elle est éduquée pour nous servir (section 1.4). 

N'oubliez jamais : derrière la magie apparente des mots, il n'y a que de la statistique et de l'architecture. Mais la façon dont vous utiliserez ces statistiques déterminera le futur de notre lien au savoir. Soyez des ingénieurs rigoureux, mais soyez surtout des citoyens conscients.

---
Notre voyage théorique de la Semaine 1 s'achève ici. Reprenez votre souffle, car dans quelques instants, nous passons à la pratique en laboratoire. Préparez vos notebooks, nous allons découper nos premiers tokens !
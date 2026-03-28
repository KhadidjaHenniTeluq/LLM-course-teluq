---
title: "11.1 Full Fine-Tuning vs PEFT"
weight: 2
---

## La métamorphose du modèle : Du nouveau-né à l'expert
Imaginez un étudiant brillant qui a lu absolument tous les livres de la bibliothèque municipale. Il sait parler de tout, il connaît la grammaire et l'histoire, mais il n'a jamais vu un seul contrat d'assurance de sa vie. Cet étudiant, c'est votre **Foundation Model**. Pour en faire un juriste, vous n'allez pas lui demander de réapprendre à lire ou de relire toute la bibliothèque. Vous allez lui donner des cours spécialisés.

C'est ce que nous appelons le **Fine-Tuning**. C'est le processus qui permet de spécialiser un modèle de langage sur un domaine ou une tâche précise. Mais avant d'ouvrir vos notebooks, nous devons comprendre les étapes qui mènent à la naissance d'un LLM prêt pour la production.

Regardons attentivement la **Figure 11-1 : Apprentissage du modèle de langage par prédiction du mot suivant** . Cette illustration décrit la phase de **Pretraining** (Pré-entraînement). Ici, le modèle apprend seul, sur des données non étiquetées (unlabeled data). Son unique but est statistique : prédire la probabilité du mot suivant. 

{{< bookfig src="257.png" week="11" >}}

> [!NOTE]
🔑 **C'est le socle de base.** Durant cette phase, le modèle acquiert ce que l'on appelle une "compréhension du monde", mais il ne sait pas encore obéir à une consigne.

---
## L'avènement du Fine-Tuning Supervisé (SFT)
Une fois ce socle acquis, nous passons à la phase de **Supervised Fine-Tuning (SFT)**, illustrée par la **Figure 11-2 : Apprentissage pour suivre des instructions** .

{{< bookfig src="258.png" week="11" >}}

*   **Le concept** : Contrairement au pré-entraînement, le SFT utilise des données étiquetées par des humains (Instruction Data). 
*   **L'exemple de la figure** : On donne au modèle une tâche précise ("Tell me something about reinforcement learning"). Au lieu de simplement "compléter" le texte de manière aléatoire, le modèle apprend à produire une réponse utile et structurée.

> [!IMPORTANT]
🔑 **Je dois insister sur cette distinction :** Le pré-entraînement remplit la bibliothèque du cerveau de l'IA, le fine-tuning lui apprend à répondre aux questions des clients qui entrent dans cette bibliothèque.

---
## La méthode historique : Le Full Fine-Tuning
Pendant longtemps, la seule façon d'adapter un modèle était de pratiquer le **Full Fine-Tuning**.
*   **Le principe** : On prend le modèle (par exemple Llama-3-8B) et on ré-entraîne *l'intégralité* de ses 8 milliards de paramètres sur nos données spécifiques.
*   **L'avantage** : C'est la méthode la plus puissante. Le modèle peut modifier profondément ses représentations internes pour coller parfaitement à votre domaine.

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Le Full Fine-Tuning est devenu un luxe inaccessible pour la plupart des entreprises et des chercheurs. Pourquoi ?
1.  **Le coût mémoire** : Pour entraîner un modèle, vous ne stockez pas seulement ses poids (ex: 16 Go pour un modèle 8B en 16-bit). Vous devez stocker les gradients, les états de l'optimiseur (Adam) et les activations. Pour un modèle de 7B, il vous faut environ **120 à 160 Go de VRAM** pour un Full Fine-Tuning. Votre petite T4 sur Colab (16 Go) exploserait instantanément.
2.  **L'Oubli Catastrophique (Catastrophic Forgetting)** : En modifiant tous les poids trop brutalement, le modèle risque de devenir un expert en droit mais de perdre sa capacité à parler normalement ou à faire des calculs simples. C'est l'un des plus grands risques du réglage fin intégral.

Regardez la **Figure 11-3 : Comparaison entre Pré-entraînement et Fine-Tuning**. La figure montre que le pré-entraînement se fait sur des volumes massifs (Téraoctets) pour apprendre la langue, alors que le fine-tuning se fait sur des volumes réduits (Mégaoctets ou Gigaoctets) de données de haute qualité pour apprendre la tâche.

{{< bookfig src="262.png" week="11" >}}


---
## La révolution PEFT (Parameter-Efficient Fine-Tuning)
> [!TIP]
C'est ici que l'ingénierie moderne devient proprement géniale. 

Face à l'impossibilité de modifier des milliards de paramètres sur du matériel standard, les chercheurs ont inventé le **PEFT**.

> [!NOTE]
🔑 **L'intuition fondamentale :** Au lieu de changer tous les poids du modèle (ce qui revient à repeindre entièrement un gratte-ciel pour réparer une rayure sur un mur), nous allons garder le modèle original totalement **gelé** (frozen) et ne modifier (ou ajouter) qu'une minuscule fraction de paramètres (souvent moins de 1% !).

Parmi les techniques de PEFT, on trouve :
1.  **Les Adapters** : On insère de petites couches neuronales entre les couches du Transformer original.
2.  **Prompt Tuning** : On apprend des vecteurs spéciaux ("soft prompts") que l'on ajoute au début de la question de l'utilisateur.
3.  **LoRA (Low-Rank Adaptation)** : C'est la reine des techniques actuelles, que nous détaillerons en section 11.2.

---
## Pourquoi choisir le PEFT plutôt que le Full Fine-Tuning ?

> [!IMPORTANT]
⚠️ Mes chers étudiants, en tant qu'ingénieurs, vous devez être économes.
 
Le PEFT n'est pas un "choix par défaut pour les pauvres", c'est souvent le meilleur choix technique.

*   **Vitesse d'itération** : Entraîner 1% des paramètres prend beaucoup moins de temps.
*   **Stockage réduit** : Au lieu de sauvegarder un nouveau modèle de 30 Go pour chaque client, vous ne sauvegardez qu'un petit fichier de 50 Mo (l'adaptateur). À l'exécution, vous chargez le modèle de base une fois et vous "clipsez" l'adaptateur souhaité.
*   **Protection contre l'oubli** : Comme les poids originaux sont gelés, le modèle conserve toute sa "culture générale" et sa stabilité linguistique.

---
## Tableau 11-1 : Duel au sommet : Full FT vs PEFT

| Caractéristique | Full Fine-Tuning | PEFT (ex: LoRA) |
| :--- | :--- | :--- |
| **Paramètres entraînés** | 100% | < 1% (souvent 0.1%) |
| **Mémoire GPU requise** | Massive (ex: 160 Go pour 7B) | Faible (ex: 12-16 Go pour 7B) |
| **Risque d'oubli** | Très élevé | Très faible |
| **Coût de stockage** | Identique au modèle original | Minuscule (fichiers adaptateurs) |
| **Performance finale** | Optimale sur de très gros datasets | Souvent identique au Full FT sur petits/moyens datasets |

---
## Les trois étapes du cycle de vie d'un LLM
Pour conclure cette introduction aux stratégies, regardons la **Figure 11-4 : Les trois étapes de création d'un LLM de haute qualité** .

{{< bookfig src="259.png" week="11" >}}


1.  **Pretraining** : Passage d'un modèle non-entraîné à un **Base Model**. (Phase très coûteuse, réservée aux géants).
2.  **Fine-tuning (SFT)** : Passage du Base Model à un modèle **Instruction-tuned**. C'est là que vous intervenez le plus souvent en entreprise.
3.  **Preference Tuning (Alignment)** : Ajustement final via RLHF ou DPO (que nous verrons en Semaine 12) pour rendre le modèle poli et sûr.


> [!TIP]
🔑 **Mon message** : Vous voyez ce pipeline ? En tant qu'experts, vous n'avez pas besoin de refaire l'étape 1. Votre valeur ajoutée réside dans votre capacité à exécuter l'étape 2 (SFT) de manière chirurgicale. Et pour cela, le PEFT est votre scalpel.

---
## Éthique et Responsabilité : Les données sont des promesses

> [!CAUTION]
⚠️ Mes chers étudiants, le fine-tuning est le moment où vous injectez vos valeurs dans la machine.

Si vos données d'instruction contiennent des exemples de réponses biaisées, sexistes ou imprécises, le fine-tuning va "graver" ces comportements dans le modèle avec une force bien supérieure au pré-entraînement. 
1.  **La qualité prime sur la quantité** : Il vaut mieux 500 exemples parfaits écrits par des experts que 50 000 exemples médiocres ramassés sur le web.
2.  **Le risque de spécialisation** : Un modèle fine-tuné pour être un "vendeur agressif" peut devenir harcelant. Vous êtes responsables du ton et de l'éthique de la spécialisation.

---
Vous maîtrisez maintenant le paysage stratégique. Vous savez pourquoi le Full Fine-Tuning est un dinosaure en voie de disparition et pourquoi le PEFT est l'avenir de l'IA agile. Dans la prochaine section ➡️, nous allons plonger dans les mathématiques d'une technique qui a sauvé l'IA de la paralysie : nous allons décortiquer **LoRA**, la décomposition de matrices à bas rang.
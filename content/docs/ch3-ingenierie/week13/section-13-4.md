---
title: "13.4 Bonnes pratiques de déploiement"
weight: 5
---


## Le saut dans le vide : L'instant "Générer" en production

Bonjour à toutes et à tous ! Nous y sommes. Nous avons optimisé la vitesse (section 13.1), verrouillé la sécurité (section 13.2) et navigué dans les méandres de la loi (section 13.3). Mais posséder une formule 1 ne fait pas de vous un pilote de course si vous ne savez pas comment la gérer sur un circuit réel. 

> [!IMPORTANT]
📌 **Je dois insister :** le déploiement est le moment le plus critique de la vie d'un projet d'IA. C'est l'instant où votre code rencontre l'imprévisibilité de milliers d'êtres humains. 

Un déploiement réussi n'est pas un événement ponctuel, c'est un processus continu de vigilance, d'écoute et d'ajustement. Aujourd'hui, je vais vous donner les clés de la "salle de contrôle". Respirez, nous allons apprendre à piloter vos modèles dans la durée, avec sagesse et rigueur.

Passer d'un notebook Colab à une application utilisée par des clients réels demande un changement de paradigme. la performance d'un modèle en laboratoire (sur des benchmarks comme **MMLU** (*Massive Multitask Language Undertanding*) ou **GSM8k** (*Grade School Math 8K*)) ne garantit jamais son succès en production. Les utilisateurs vont poser des questions étranges, le serveur va chauffer, et les biais que nous pensions avoir éliminés vont refaire surface. C'est pour cela que nous devons construire une infrastructure de confiance autour du modèle.


---
## Le système nerveux du déploiement : Monitoring et Logging

> [!NOTE]
*On ne peut pas gérer ce que l'on ne peut pas mesurer.*

En production, vous avez besoin de "yeux" partout. Le monitoring des LLM se divise en trois couches essentielles.

### La couche technique (L'état de la machine)
Vous devez surveiller en temps réel la santé de vos GPU. 
*   **Usage de la VRAM** : Si votre cache KV (section 13.1) grandit trop vite, le serveur va planter.
*   **Latence de premier token (TTFT)** : C'est le temps que met l'utilisateur avant de voir la première lettre. S'il dépasse 2 secondes, l'expérience est perçue comme médiocre.
*   **Débit (Throughput)** : Combien de requêtes traitez-vous par minute ? Si ce chiffre chute alors que le trafic monte, votre file d'attente (batching) est mal configurée.

### La couche sémantique (Le contenu des réponses)
C'est ici que l'ingénierie des LLM devient unique. Vous devez monitorer la "dérive" (*drift*) de votre modèle.
*   **Dérive de sujet** : Si vous avez déployé un assistant pour la cuisine et que les utilisateurs commencent à lui poser des questions de politique, vos filtres de sécurité (section 13.2) doivent s'activer.
*   **Analyse de sentiment des retours** : Si le ton des utilisateurs devient agressif, c'est souvent le signe que l'IA répond de manière frustrante ou inutile. 

> [!IMPORTANT]
✍🏻 **Je dois insister :** écoutez le silence des utilisateurs qui partent, c'est votre plus grande alerte.

### La couche de sécurité (Les logs d'audit)

> [!WARNING]
⚠️ Le logging n'est pas une option, c'est une preuve.

Comme nous l'avons vu en section 13.3, le RGPD et l'AI Act imposent de pouvoir retracer une décision de l'IA. Vos logs doivent enregistrer :
1.  Le prompt complet envoyé (anonymisé).
2.  La version exacte du modèle et de l'adaptateur LoRA utilisé.
3.  Les hyperparamètres (température, top_p).
4.  Les documents récupérés par le RAG (section 9.1).
Sans ces quatre éléments, vous serez incapable de corriger un bug ou de répondre à une plainte légale.


---
## L'évolution continue : A/B Testing et Itération
> [!NOTE]
*Ne croyez jamais que votre modèle V1 est le meilleur.* 

En production, nous utilisons le **A/B Testing**, une méthode inspirée du marketing mais appliquée à la sémantique.

### Le duel des modèles
Imaginez que vous ayez une nouvelle version de votre adaptateur LoRA qui semble plus performante en test. Au lieu de remplacer l'ancien modèle, vous envoyez 10% des utilisateurs vers le nouveau (Modèle B) et 90% vers l'ancien (Modèle A). 
*   **Mesure de succès** : Quel modèle a le meilleur taux de satisfaction ? Lequel nécessite le moins de corrections humaines ?
*   **Shadow Deployment** : Une variante consiste à faire tourner le modèle B "dans l'ombre". Il génère des réponses pour chaque question, mais l'utilisateur ne voit que celles du modèle A. Vous comparez ensuite les réponses en interne pour valider la montée en version sans aucun risque pour le client. 

🔑 **C'est la méthode de sécurité maximale.**

### La boucle de feedback (RLHF continu)
Les données les plus précieuses sont celles que vos utilisateurs vous donnent gratuitement via le petit bouton "Pouce en l'air / Pouce en bas".

> [!NOTE]
✍🏻 **Note** : Ces clics sont vos futures données de préférence pour votre prochain entraînement DPO. 

Un déploiement responsable transforme chaque interaction en une leçon pour la version suivante. C'est ce qu'on appelle le **volant d'inertie de la donnée** (*Data Flywheel*).


---
## Le support utilisateur et l'interface de confiance

> [!CAUTION]
⚖️ L'IA est une prothèse cognitive, pas un substitut de responsabilité. 

La façon dont vous présentez l'IA à l'utilisateur change radicalement la perception de ses erreurs.

### La gestion des attentes
Un utilisateur qui croit parler à un humain sera furieux de découvrir une erreur factuelle. Un utilisateur qui sait qu'il interagit avec une "IA expérimentale" sera plus indulgent et vigilant.
*   **Disclaimers clairs** : Affichez toujours un message : "Cette IA peut halluciner, vérifiez les informations importantes."
*   **Bouton de signalement** : Facilitez la dénonciation des biais ou des erreurs. Un utilisateur qui se sent écouté est un utilisateur qui pardonne.

### Les citations : Le contrat de preuve
Comme nous l'avons appris en [**Semaine 9**]({{< relref "section-9-1.md" >}}#RAG-citation), un RAG doit citer ses sources.

> [!IMPORTANT]
📌 **Je dois insister :** En production, cliquez sur vos propres liens de sources ! Si l'IA cite la page 42 d'un PDF alors que l'info est en page 5, votre crédibilité s'effondre. 

Le support utilisateur commence par une interface qui permet à l'humain de vérifier la machine.

---
## Le filet de sécurité : Maintenance et Rollback

> [!WARNING]
*En informatique, tout ce qui peut mal tourner tournera mal un jour.*

Vous devez avoir un plan de secours.
### Le bouton "Panique" (Rollback)
Si après une mise à jour, votre IA commence soudainement à être impolie ou à donner des conseils financiers désastreux, vous devez pouvoir revenir à la version précédente en moins d'une minute. 
*   **Infrastructure immuable** : Ne modifiez jamais un modèle "en place". Déployez un nouveau conteneur et changez le routage. 
*   **Versioning des prompts** : 
>> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** On versionne le code, mais on oublie souvent de versionner les prompts. Un changement d'un seul mot dans votre prompt système (section 8.1) peut ruiner le comportement d'un modèle pourtant parfait.

### La maintenance des bases vectorielles
Votre base de connaissances (Semaine 6) n'est pas une pierre tombale. 
1.  **Mise à jour des faits** : Une information vraie en 2023 peut être fausse en 2024. Prévoyez un script qui rafraîchit vos embeddings régulièrement.
2.  **Nettoyage des doublons** : Des documents en triple dans votre base FAISS vont saturer la réponse de l'IA avec la même information, empêchant la diversité des points de vue (Semaine 8.2).


---
## La Checklist du déploiement responsable (10 piliers)
Pour conclure cette semaine, je vous demande d'imprimer virtuellement cette liste. C'est votre examen final de conscience professionnelle avant de mettre n'importe quel code en ligne.

1.  **Anonymisation** : Ai-je vérifié qu'aucune donnée personnelle (PII) n'est stockée dans mes logs ou mes bases vectorielles ?
2.  **Optimisation** : Mon KV cache et ma quantification (4-bit/8-bit) sont-ils configurés pour une latence minimale ?
3.  **Audit de Sécurité** : Mon système résiste-t-il aux 5 injections de prompt les plus courantes ?
4.  **Transparence** : L'interface indique-t-elle clairement que l'utilisateur parle à une IA ?
5.  **Ancrage (Grounding)** : Mon RAG fournit-il des citations vérifiables pour chaque fait important ?
6.  **Monitoring Éthique** : Ai-je un système d'alerte automatique si le score de toxicité des réponses augmente ?
7.  **Human-in-the-loop** : Pour les décisions à haut risque, un humain doit-il valider la sortie ?
8.  **Sobriété Numérique** : Ai-je choisi le plus petit modèle possible (ex: Phi-3 au lieu de GPT-4) capable de remplir la mission ?
9.  **Plan de Rollback** : Puis-je revenir à la version stable précédente en un clic ?
10. **Documentation (Model Card)** : Ma Model Card est-elle à jour avec les limites connues du système ?


---
## Conclusion de la semaine
> [!TIP]
✉️ **Le message final** : Mes chers étudiants, vous avez maintenant terminé le cycle complet du déploiement. Vous n'êtes plus seulement des "codeurs de modèles", vous êtes des architectes de systèmes intelligents et responsables. 

N'oubliez jamais : un déploiement n'est pas la fin d'un projet, c'est le début d'une conversation avec le monde. Soyez attentifs aux murmures de vos utilisateurs, soyez impitoyables avec vos propres biais et soyez fiers de la rigueur que vous apportez à cette technologie. 

L'IA de demain ne sera pas jugée sur sa puissance brute, mais sur sa fiabilité et son intégrité. Vous avez les outils pour construire cette IA de confiance.

---
Nous avons terminé cette immense semaine ! C'était le dernier grand chapitre technique de notre voyage. Vous avez appris à transformer le génie statistique en un serviteur industriel et éthique. La semaine prochaine, nous prendrons de la hauteur. Nous ferons la synthèse de tout ce que nous avons appris et nous regarderons vers l'horizon : **les agents autonomes**, **les nouvelles architectures** comme Mamba, et **l'avenir de la recherche**. Mais avant cela, place au laboratoire final de production 🧪➡️!
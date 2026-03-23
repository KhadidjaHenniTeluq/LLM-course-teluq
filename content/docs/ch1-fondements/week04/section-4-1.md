---
title: "4.1 La famille BERT"
weight: 2
---

{{< katex />}}

## L'intuition de l'encodeur : Lire entre les lignes

Imaginez que vous lisiez une phrase. Pour en comprendre le sens, vous ne vous contentez pas de regarder les mots qui précèdent. Votre regard fait des va-et-vient, n'est-ce pas ? C'est exactement ce que fait BERT (*Bidirectional Encoder Representations from Transformers*). Contrairement à la famille GPT que nous avons croisée et qui est "aveugle" aux mots futurs, BERT est **bidirectionnel**.

Comme vous pouvez l'observer sur la **Figure 4-1 : Architecture BERT base**, un modèle BERT-base est une pile de 12 blocs "encodeurs". 

{{< bookfig src="25.png" week="04" >}}

{{% hint info %}}
🔑 **Je dois insister sur cette distinction :** ici, il n'y a pas de mécanisme de masquage de l'attention comme dans les décodeurs. Chaque mot (token) peut "voir" tous les autres mots de la phrase, qu'ils soient à sa gauche ou à sa droite. C'est cette vision globale qui permet une compréhension contextuelle si riche.
{{% /hint %}}

## Le token [CLS] : Le porte-parole de la phrase
Dans un modèle de représentation, nous ne voulons pas générer la suite du texte, nous voulons un "résumé mathématique" de ce que le texte raconte. Pour cela, BERT introduit une convention géniale : le token **[CLS]** (pour *Classification*).

> [!CAUTION]
> ⚠️ **Attention : erreur fréquente ici !** Le token `[CLS]` n'est pas un mot magique. C'est simplement un token spécial que l'on place systématiquement au tout début de chaque entrée. Parce que BERT utilise la *Self-Attention* totale, ce token `[CLS]` va absorber des informations de *tous* les autres tokens de la phrase. 

> [!NOTE]
🔑 **La règle d'or :** À la sortie de la 12ème couche, le vecteur correspondant au token `[CLS]` est utilisé comme la représentation compressée de toute la séquence. C'est ce vecteur que nous brancherons sur un classifieur pour décider si une critique est positive ou négative.

## Apprendre par les trous : Le Masked Language Modeling (MLM)
Comment entraîne-t-on un modèle à "comprendre" sans lui donner de labels ? On utilise un jeu d'enfant : le texte à trous. C'est le **Masked Language Modeling**, illustré dans les **Figures 4-2 et 4-3**.

{{< bookfig src="26.png" week="04" >}}
{{< bookfig src="27.png" week="04" >}}


Le processus est fascinant :
1.  On prend une phrase normale (ex: "Le chat boit du lait").
2.  On cache aléatoirement 15% des mots avec un token spécial **[MASK]**.
3.  On demande au modèle de deviner ce qu'il y avait sous le masque.

Pour réussir, BERT est forcé d'apprendre la grammaire ("Le" suggère un nom masculin), la syntaxe, et surtout la sémantique ("boit" et "lait" sont fortement liés au mot caché "chat").

## L'évolution de la lignée : De RoBERTa à DeBERTa
Depuis la sortie de BERT par Google en 2018, la famille s'est agrandie. Regardez la **Figure 4-4 : Timeline des modèles BERT-like**. Chaque nouveau-venu a apporté une innovation majeure.

{{< bookfig src="91.png" week="04" >}}

### 1. RoBERTa (Facebook AI)
RoBERTa est la preuve que "plus c'est long, meilleur c'est". Les chercheurs ont repris l'architecture de BERT mais l'ont entraînée sur beaucoup plus de données, pendant plus longtemps, et en supprimant une tâche d'entraînement secondaire jugée inutile (la *Next Sentence Prediction*). 
> [!NOTE]
🔑 **Notez bien :** RoBERTa est souvent le choix par excellence pour la classification aujourd'hui car ses représentations sont plus robustes.

### 2. DistilBERT (Hugging Face)
« Tout le monde n'a pas un supercalculateur dans son garage ! » DistilBERT utilise une technique appelée "distillation de connaissances". On prend un gros BERT (le professeur) et on entraîne un petit BERT (l'élève) à imiter ses réponses. 
*   **Résultat** : 40% plus petit, 60% plus rapide, tout en conservant 97% des performances. C'est le modèle idéal pour les applications mobiles ou les serveurs à faibles ressources.

### 3. ALBERT et DeBERTa
*   **ALBERT** : Utilise des astuces mathématiques pour réduire le nombre de paramètres sans perdre en puissance.
*   **DeBERTa** : Introduit l'attention "désentrelacée" (*disentangled attention*). Il sépare le contenu du mot de sa position relative. C'est actuellement l'un des modèles les plus performants sur les benchmarks de compréhension de lecture.

## Tableau comparatif de la famille BERT

| Modèle | Innovation Clé | Cas d'usage idéal |
| :--- | :--- | :--- |
| **BERT** | Premier modèle bidirectionnel massif | Baseline historique, recherche générale |
| **RoBERTa** | Entraînement optimisé, plus de données | Classification haute performance |
| **DistilBERT** | Distillation (plus léger/rapide) | Inférence en temps réel, Edge computing |
| **DeBERTa** | Attention désentrelacée | Tâches de raisonnement complexe |


## Éthique et Transparence : Le miroir bidirectionnel
> [!WARNING]
⚠️ **Mes chers étudiants, soyez vigilants!** 
Parce que BERT regarde la phrase dans les deux sens, il est extrêmement sensible au contexte. Mais ce contexte contient nos biais. Si BERT voit 100 000 fois "L'infirmière est entrée dans la pièce" et "Le médecin est entré dans la pièce", l'embedding du mot `[MASK]` dans "Le [MASK] est entré..." sera statistiquement biaisé vers le genre masculin ou féminin selon la profession.

> [!IMPORTANT]
🔑 **Je dois insister :** Ces modèles de représentation ne sont pas des arbitres de vérité. Ce sont des miroirs statistiques de nos écrits. Lorsque vous utilisez BERT pour filtrer des CV ou analyser des sentiments, vous devez auditer les biais que le modèle pourrait avoir "appris" durant son *MLM*. 

---
Vous avez maintenant les clés de la famille BERT. Vous comprenez que leur force réside dans leur capacité à "figer" le sens d'un texte dans un vecteur riche et bidirectionnel. Dans la section suivante, nous verrons comment transformer ces connaissances théoriques en stratégies concrètes d'utilisation : faut-il tout réentraîner ou simplement "geler" le modèle ?

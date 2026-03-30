---
title: "3.1 Le mécanisme d'attention : Mathématiques détaillées"
weight: 2
---

{{< katex />}}

## La fin de la lecture linéaire : L'intuition de l'omniprésence
Imaginez que vous soyez dans une soirée très bruyante. Pour comprendre votre interlocuteur, votre cerveau ignore 90 % des sons ambiants et se concentre sur les fréquences de sa voix. C'est l'attention sélective. En NLP, nous avons longtemps forcé les machines à lire comme des écoliers, un mot après l'autre (RNN). 

> [!IMPORTANT]
‼️ **Je dois insister :** le Transformer a aboli cette dictature du temps. Il ne lit pas de gauche à droite; il regarde la phrase comme une image globale. 

Regardons la **Figure 3-1 : Cadrage simplifié de l'attention** . 

{{< bookfig src="68.png" week="03" >}}

Cette illustration nous présente une séquence d'entrée où un mot (noté par la flèche rouge) est en train d'être traité. Ce mot ne se contente pas de regarder son voisin ; il envoie des "sondes" vers toutes les autres positions de la séquence. 
*   **L'idée clé** : Chaque mot de la phrase reçoit un "budget" d'attention de 100 % qu'il doit répartir entre tous les autres mots, y compris lui-même. 
*   **Le résultat** : Un mot isolé (embedding statique) s'enrichit de l'information de ses voisins pour devenir un vecteur contextuel unique.

---
## Le Trio Magique : Query, Key et Value

> [!TIP]
✍🏻 Mes chers étudiants, voici le concept le plus crucial de votre formation. Si vous comprenez le triplet Query, Key et Value, vous comprenez l'intelligence artificielle moderne. 

Le Transformer ne compare pas les vecteurs de mots directement. Il projette chaque mot dans trois espaces fonctionnels différents.

Comme l'illustre la **Figure 3-2 : Matrices de projection**, le modèle possède trois matrices de poids apprises durant l'entraînement : $W_Q, W_K$ et $W_V$. Lorsqu'un mot entre dans la couche d'attention :

{{< bookfig src="71.png" week="03" >}}

1.  **Query (La Requête - Q)** : C'est ce que le mot cherche. Si le mot est "il", sa requête demande : "Où est mon sujet masculin dans cette phrase ?".
2.  **Key (La Clé - K)** : C'est l'étiquette du mot. Un mot comme "livre" possède une clé qui dit : "Je suis un objet inanimé masculin".
3.  **Value (La Valeur - V)** : C'est l'information sémantique pure que le mot contient.

> [!TIP]
💡 **Mon analogie** : Imaginez que vous cherchiez un tutoriel de cuisine sur YouTube. 

> Votre barre de recherche est la **Query**. Le titre de la vidéo sur le serveur est la **Key**. Le contenu de la vidéo (ce que vous allez apprendre) est la **Value**. L'attention est l'algorithme qui fait correspondre votre recherche au titre le plus proche.

---
## Le calcul matriciel étape par étape
Nous décomposons cette chorégraphie mathématique avec une précision chirurgicale.

### Étape 1 : Le calcul des scores de pertinence (Figure 3-3)

{{< bookfig src="72.png" week="03" >}}

**ℹ️ Explication** : Pour chaque mot, on multiplie sa **Query** par les **Keys** de tous les autres mots. Mathématiquement, c'est un produit scalaire (Dot Product). 
*   Si les vecteurs Q et K pointent dans la même direction, le score est élevé.
*   Si ils sont orthogonaux, le score est nul.
Cette opération crée une grille de scores indiquant à quel point chaque mot est "intéressé" par les autres.

### Étape 2 : Le "Scaling" et le Softmax (Figure 3-4)

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Si l'on s'arrête aux scores bruts, les nombres peuvent devenir immenses, ce qui fait "exploser" les gradients lors de l'entraînement.

> [!TIP]
✅ **La solution mathématique** : On divise les scores par la racine carrée de la dimension des clés ($\sqrt{d_k}$). C'est le **Scaled Dot-Product Attention**. 


{{< bookfig src="73.png" week="03" >}}

Ensuite, comme le montre la **Figure 3-4** , on applique une fonction **Softmax**. 
*   **L'effet visuel** : Les scores sont transformés en probabilités entre 0 et 1. La somme totale pour chaque mot est égale à 100 %. On voit alors apparaître une "carte d'attention" où certains liens s'allument (forte probabilité) et d'autres s'éteignent.


### Étape 3 : La pondération des valeurs (Figure 3-5)

{{< bookfig src="74.png" week="03" >}}

**ℹ️ Explication** : Enfin, on multiplie ces probabilités par les **Values**. 
*   Si le mot "il" a 90 % d'attention sur "livre", alors 90 % du vecteur final de "il" sera composé de l'information sémantique de "livre". 
*   Le résultat est un nouveau vecteur, le **Contextual Embedding**, qui a "aspiré" le sens de son environnement.


---
## Multi-Head Attention : Les cerveaux parallèles
🤔 *Pourquoi se contenter d'un seul regard sur une phrase ?* 

Un mot peut avoir plusieurs rôles : il a un rôle grammatical, un rôle sémantique et un rôle émotionnel. 

C'est ce qu'illustre la **Figure 3-6 : Intuition des têtes d'attention**. 

{{< bookfig src="70.png" week="03" >}}

Au lieu d'avoir un seul trio Q, K, V, nous en créons plusieurs en parallèle (généralement **8**, **12** ou même **96** têtes). 
*   Une tête peut se spécialiser dans la détection des verbes.
*   Une autre dans la résolution des pronoms (coréférence).
*   Une autre dans l'analyse du ton (ironie).

> [!IMPORTANT]
‼️ **Je dois insister :** La Multi-Head Attention est ce qui donne au Transformer sa nuance. 

À la fin, on fusionne (concatène) les résultats de toutes les têtes pour obtenir une vision riche et multidimensionnelle de la phrase.

---
## Exemple numérique : Le bac à sable des matrices
Pour bien fixer l'idée, imaginons une séquence de deux tokens : "Chat" et "Dort".
Supposons que nos vecteurs de Query et Key soient simplifiés à 2 dimensions.

1.  **Matrices Q et K** :
    *   $Q_{chat} = [1, 0]$, $K_{chat} = [1, 0]$
    *   $Q_{dort} = [0, 1]$, $K_{dort} = [0, 1]$
2.  **Calcul du score** (Produit scalaire $Q \cdot K^T$) :
    *   Score "Chat" vers "Chat" : $1\times1 + 0\times0 = 1$
    *   Score "Chat" vers "Dort" : $1\times0 + 0\times1 = 0$
3.  **Softmax** :
    *   Le mot "Chat" porte 100 % de son attention sur lui-même (car score=1 vs 0). 
    *   Si les vecteurs étaient plus proches (ex: "Chat" et "Félin"), les scores seraient partagés (ex: 60 % / 40 %).

> [!NOTE]
✍🏻 **Note** : Dans un vrai LLM comme GPT-4, ces calculs se font sur des vecteurs de dimension **4096** ou plus. 

> La complexité est telle que seule la puissance des GPU (section 1.2) permet de résoudre ces milliards de multiplications par seconde.

---
## L'attention comme moteur de la parallélisation
⁉️ *Pourquoi avons-nous abandonné les RNN ?* 

Parce que dans le calcul $Q \cdot K^T$, nous pouvons calculer TOUS les scores de TOUS les mots en une seule opération matricielle géante.

> [!NOTE]
🚀 **La rupture technologique** : On ne fait plus la queue. Le GPU traite la phrase entière comme un bloc de pixels. C'est ce qui a permis de multiplier par 1000 la vitesse d'entraînement et d'ingérer l'intégralité du web.

---
## Éthique et Transparence : Le biais de l'attention

> [!CAUTION]
⚖️ Mes chers étudiants, l'attention n'est pas neutre.

Les matrices $W_Q, W_K, W_V$ sont apprises sur des données humaines. 
1.  **Le renforcement des stéréotypes** : Si, dans les données d'entraînement, le mot "Infirmière" porte systématiquement son attention sur des pronoms féminins, le modèle va figer cette association. L'attention devient alors un mécanisme de reproduction des préjugés. 
2.  **L'opacité du raisonnement** : Visualiser l'attention (Exercice 1 du laboratoire) nous donne une illusion de compréhension. Mais attention : une tête d'attention qui regarde une virgule peut le faire pour des raisons de syntaxe pure, et non pour le sens. 

**Ne prêtez pas d'intentions humaines à une multiplication matricielle!**

---
## Synthèse
Pour maîtriser cette section, vous devez être capables de réciter la formule de l'attention de *Vaswani et al. (Attention is all you need!)* :

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
*   $QK^T$ : Qui regarde qui ? (Scores)
*   $\sqrt{d_k}$ : On calme les nombres (Scaling).
*   $softmax$ : On transforme en pourcentages (Normalisation).
*   $V$ : On extrait le sens (Information).

> [!TIP]
✉️ **Mon message** : L'attention est l'acte par lequel le modèle crée du contexte. 

> Sans elle, les mots sont des îles. Avec elle, ils forment un continent de pensée. C'est la brique la plus puissante jamais inventée en informatique linguistique.

---
Vous avez maintenant dompté le lion ! Vous comprenez le mécanisme de l'attention. Mais un problème subsiste : si on traite tout en même temps, comment le modèle sait-il que le mot "Le" est avant le mot "Chat" ? Dans la section suivante ➡️, nous allons découvrir la boussole du Transformer : l'**Encodage Positionnel**.
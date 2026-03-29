---
title: "1.1 Évolution du Traitement du Langage Naturel (NLP)"
weight: 2
---

## La perspective historique : Une accélération exponentielle
Mes chers étudiants, regardez le monde autour de vous. Aujourd'hui, votre téléphone traduit des panneaux en temps réel et votre éditeur de texte finit vos phrases. Mais cela n'a pas toujours été aussi fluide. 

Pour débuter, observons la **Figure 1-1 : Timeline historique du NLP** . Cette illustration nous montre que le domaine a connu trois grandes ères. 

{{< bookfig src="5.png" week="01" >}}

1.  **L'ère symbolique (1950-1990)** : On tentait de coder manuellement des règles de grammaire. C'était l'époque des "systèmes experts". C'était rigide et incapable de gérer l'ironie ou l'évolution naturelle du langage.
2.  **L'ère statistique (1990-2010)** : On a commencé à compter. Si le mot "argent" apparaît souvent avec "banque", alors il y a une probabilité de lien. C'est l'époque du *Machine Learning* classique.
3.  **L'ère neuronale (2012-Aujourd'hui)** : C'est l'explosion du *Deep Learning*. On ne compte plus seulement, on apprend des représentations mathématiques multidimensionnelles. 

> [!NOTE]
💡 **Notez bien cette intuition :** En 2023, nous avons atteint un point de bascule où les modèles génératifs (ChatGPT, Claude, Llama) ont fusionné toutes ces connaissances pour devenir des assistants universels.

---
## Les missions de l'IA de langage
Avant de coder, demandons-nous : que voulons-nous que la machine fasse ? La **Figure 1-2 : Tâches typiques du Language AI** nous présente les quatre piliers fondamentaux que nous allons explorer tout au long du semestre :

<a id="fig-1-2"></a>
{{< bookfig src="6.png" week="01" >}}

*   **La Génération de texte** : Produire du contenu fluide (emails, poèmes, code).
*   **Les Embeddings** : Transformer du sens en coordonnées GPS mathématiques (essentiel pour la recherche sémantique).
*   **La Classification** : Ranger des textes dans des cases (Spam/Non-spam, Positif/Négatif).
*   **L'Extraction** : Sortir des informations précises d'un texte (noms de lieux, dates, prix). 

> [!WARNING]
⚠️ **Attention :** Un LLM moderne fait tout cela à la fois, mais historiquement, nous utilisions un modèle différent pour chaque tâche !

---
## La méthode de la "Sacoche de mots" (Bag-of-Words) : L'intuition du compte
Pour qu'un ordinateur traite du texte, il faut transformer les lettres en chiffres. La méthode la plus ancienne et la plus célèbre est le **Bag-of-Words (BoW)**. 

Imaginez que vous preniez une phrase, que vous découpiez chaque mot, et que vous les jetiez tous dans un sac en ignorant totalement leur ordre.

ce processus est détaillé via trois figures capitales :

**1. La Tokenisation (Figure 1-3)** : 
Le premier pas est de découper la chaîne de caractères. Dans l'exemple "That is a cute dog", on sépare chaque mot sur les espaces. Chaque morceau est un **token**.

{{< bookfig src="7.png" week="01" >}}

> [!NOTE]
✍🏻 **Je dois insister :** Pour un modèle BoW, "chien" et "chiens" sont deux tokens totalement différents. La machine ne sait pas encore qu'ils parlent du même animal.

**2. La Construction du Vocabulaire (Figure 1-4)** : 
On prend tous les mots uniques de toutes nos phrases. Si on a deux phrases : "That is a cute dog" et "My cat is cute", notre vocabulaire devient : `[that, is, a, cute, dog, my, cat]`. C'est notre dictionnaire de référence.

{{< bookfig src="8.png" week="01" >}}

**3. La Vectorisation (Figure 1-5)** : 
C'est ici que le texte devient un vecteur (une liste de nombres). Pour la phrase "My cat is cute", on regarde notre vocabulaire :
*   Le mot `that` est présent ? Non (0).
*   Le mot `is` est présent ? Oui (1).
*   Le mot `cute` est présent ? Oui (1).
*   ... et ainsi de suite.
On obtient un vecteur : `[0, 1, 0, 1, 0, 1, 1]`. 

{{< bookfig src="9.png" week="01" >}}


> [!WARNING]
⚠️ Regardez bien la faille de ce système.

> Si je vous donne les mots "mange", "le", "chat", "la", "souris", pouvez-vous savoir si c'est le chat qui mange la souris ou l'inverse ? Non. Le vecteur est identique. **On a perdu la syntaxe.**

---
## Le problème de la polysémie : L'exemple "bank"
Imaginez un instant que vous cherchiez "bank" sur Google.

Dans l'approche BoW ou même avec les premiers modèles statistiques, le mot "bank" n'a qu'une seule existence numérique. 
1. "I sat on the river **bank**." (Rive de rivière)
2. "I went to the **bank** to deposit money." (Institution financière)

> [!IMPORTANT]
📌 **Je dois insister :** Dans ces modèles anciens, le vecteur du mot "bank" est une moyenne statistique de tous ses sens. 

> C'est comme essayer de définir une couleur qui serait un mélange de bleu et de rouge : vous obtenez du violet, mais vous avez perdu la pureté des deux couleurs d'origine. C'est la limite des **représentations non contextuelles**.

---
## La transition vers les Embeddings Denses (Word2Vec)
En 2013, la recherche a basculé. Au lieu d'avoir des vecteurs "creux" (plein de zéros), on a inventé les **embeddings denses**.

**L'intuition de Word2Vec** :

<a id="fig-1-6"></a>

*   **Figure 1-6** : On utilise un petit réseau de neurones. Ce n'est pas encore un LLM, mais c'est son ancêtre direct. Chaque mot est relié à d'autres par des "poids" numériques.

{{< bookfig src="10.png" week="01" >}}
<a id="fig-1-7"></a>


*   **Figure 1-7** : Le modèle s'entraîne à deviner si deux mots sont voisins. Si "Chat" et "Miaule" sont souvent voisins, leurs vecteurs vont se rapprocher géométriquement.

{{< bookfig src="11.png" week="01" >}}

<a id="fig-1-8"></a>

*   **Figure 1-8** : On découvre que les dimensions du vecteur capturent des propriétés. Une dimension pourrait représenter le genre (masculin/féminin), une autre la royauté, une autre l'aspect animal.

{{< bookfig src="12.png" week="01" >}}

<a id="fig-1-9"></a>

*   **Figure 1-9** : Si on projette ces vecteurs en 2D, on voit que "Chat" et "Chien" sont proches, alors que "Banane" est très loin.

{{< bookfig src="13.png" week="01" >}}

> [!TIP]
🔑 **Le miracle mathématique :** 
`Vecteur(Roi) - Vecteur(Homme) + Vecteur(Femme) = Vecteur(Reine)`
Le langage est devenu une géométrie. On peut calculer le sens.

---
<a id="tab-1-1"></a>

## Tableau 1-1 : Approches Symboliques vs Neuronales

| Caractéristique | Approche Symbolique (BoW / TF-IDF) | Approche Neuronale (Embeddings / LLM) |
| :--- | :--- | :--- |
| **Philosophie** | Compter les occurrences | Apprendre les relations |
| **Type de vecteur** | **Creux (Sparse)** : immense taille, majoritairement des zéros | **Dense** : taille fixe (ex: 768), nombres réels partout |
| **Contexte** | Ignoré (Sacoche de mots) | Capturé (Voisinage sémantique) |
| **Synonymes** | "Achat" et "Acquisition" sont 100% différents | "Achat" et "Acquisition" sont très proches dans l'espace |
| **Polysémie** | Échec total | Gérée par le contexte (Transformers) |

---

<a id="tf-idf"></a>

## L'évolution vers TF-IDF : Un premier pas vers la pertinence
Avant d'arriver au tout-neuronal, nous avons utilisé le **TF-IDF** (*Term Frequency-Inverse Document Frequency*). 
*   **TF** : Si un mot apparaît souvent dans mon document, il est important.
*   **IDF** : Si ce mot apparaît dans TOUS les documents de la bibliothèque (comme "le" ou "de"), il ne sert à rien pour différencier les sujets. On réduit son poids.

C'était une amélioration majeure pour la recherche documentaire, mais comme le BoW, cela restait une méthode de "comptage" incapable de comprendre que "voiture" et "automobile" désignent le même objet.

---
## Éthique et Responsabilité : Les racines du biais

> [!CAUTION]
⚖️ Mes chers étudiants, soyez vigilants. 

> Dès cette section, vous devez comprendre une chose : les embeddings neuronaux (Word2Vec) apprennent du monde tel qu'il est écrit, pas tel qu'il devrait être.

Si le modèle apprend sur des textes où "infirmière" est toujours associé aux femmes et "médecin" aux hommes, sa géométrie vectorielle va **figer** ce préjugé. 

> [!IMPORTANT]
‼️ **Je dois insister :** Le biais n'est pas un bug informatique, c'est un reflet statistique de nos propres écrits. En tant qu'experts, votre rôle est de savoir que ces vecteurs portent en eux les cicatrices des préjugés humains. 


---
Nous avons vu comment nous sommes passés de la simple statistique de comptage (BoW), qui traitait les mots comme des étiquettes isolées, à la géométrie sémantique (Word2Vec), qui traite les mots comme des points dans un espace de concepts. C'est une avancée immense, mais il manquait encore une chose : la capacité de traiter l'ordre des mots et la structure des phrases sur de longues distances. C'est ce défi qui a mené aux architectures séquentielles que nous verrons en section suivante ➡️.

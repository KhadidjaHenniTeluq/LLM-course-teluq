---
title: "2.4 Plongements lexicaux (Embeddings)"
weight: 5
---

## Bienvenue au cœur de la galaxie sémantique

« Bonjour à toutes et à tous ! Nous arrivons enfin au moment que je préfère. Si la tokenisation que nous avons vue en section 2.1 est l'acte de découper le langage, les **embeddings** sont l'acte de lui donner une âme mathématique. 

{{% hint info %}}
🔑 **Je dois insister :** c'est ici que réside la magie véritable de l'IA moderne. Sans les embeddings, un ordinateur ne ferait que manipuler des étiquettes numérotées. Avec les embeddings, il commence à "ressentir" la proximité entre les concepts. Imaginez que chaque mot de notre dictionnaire soit une étoile dans une galaxie immense. Les embeddings sont les coordonnées GPS précises qui permettent de savoir quelle étoile brille à côté d'une autre.
{{% /hint %}}

## Qu'est-ce qu'un Embedding ? L'intuition géométrique

Oubliez les définitions arides. Un embedding est une **représentation vectorielle dense**. 
*   **Vectorielle** : Une liste de nombres (ex: [0.1, -0.5, 0.8...]).
*   **Dense** : Contrairement au Bag-of-Words (section 1.1), il n'y a presque pas de zéros. Chaque nombre porte une information.

Comme vous pouvez le voir sur la **Figure 2-6**, chaque dimension du vecteur peut être imaginée comme une "propriété" abstraite. 

{{< bookfig src="12.png" week="02" >}}

**Analogie** : Imaginez un vecteur à 3 dimensions pour décrire des fruits : [Sucré, Rouge, Gros]. 
*   Une "Pomme" pourrait être `[0.9, 0.8, 0.4]`.
*   Une "Banane" pourrait être `[0.7, 0.1, 0.5]`.
*   Une "Pastèque" pourrait être `[0.6, 0.1, 0.9]`.

Dans un LLM, nous n'utilisons pas 3 dimensions, mais souvent **768** ou **1024**, voire plus. 

{{% hint warning %}}
**Attention : erreur fréquente ici !** Ces dimensions ne correspondent pas à des concepts humains clairs comme "couleur" ou "poids". Ce sont des caractéristiques apprises par le modèle au fil de ses lectures, souvent trop abstraites pour nous, mais d'une précision redoutable pour lui.
{{% /hint %}}

## Le premier pilier : Les Embeddings Statiques (Word2Vec)

Avant les LLM, nous utilisions des embeddings dits **statiques**. Le plus célèbre est **Word2Vec** (2013). La **Figure 2-7** montre comment le modèle stocke ces vecteurs : c'est une simple table de correspondance (Lookup Table). Chaque token a son vecteur unique, une fois pour toutes.

{{< bookfig src="43.png" week="02" >}}

### L'algorithme Skip-gram et le Negative Sampling

Comment la machine apprend-elle ces vecteurs ? Par l'observation de ses voisins. C'est l'intuition du **Skip-gram** illustrée en **Figure 2-8** : on prend un mot cible (ex: "sat") et on demande au modèle de prédire les mots qui l'entourent ("The", "cat", "on", "the").

{{< bookfig src="50.png" week="02" >}}

{{% hint info %}}
🔑 **Note technique sur le Negative Sampling** : Pour apprendre efficacement, le modèle a besoin de contre-exemples. Si je lui montre seulement que "chat" va avec "mange", il pourrait devenir "paresseux" et répondre "oui" à tout. On lui montre donc des paires absurdes (ex: "chat" + "ordinateur") et on lui dit : "Ce ne sont pas des voisins". C'est ce contraste qui sculpte la précision du vecteur. 
{{% /hint %}}

{{< bookfig src="47.png" week="02" >}}
{{< bookfig src="48.png" week="02" >}}
{{< bookfig src="49.png" week="02" >}}
{{< bookfig src="51.png" week="02" >}}
{{< bookfig src="52.png" week="02" >}}

### La limite fatidique : La Polysémie

Le problème des modèles comme Word2Vec ou GloVe, c'est qu'ils sont incapables de gérer les mots à plusieurs sens. 

{{% hint info %}}
🔑 **C'est une distinction non-négociable :** Dans un modèle statique, le mot "avocat" n'a qu'un seul vecteur. Si vous parlez de justice, le vecteur est le même que si vous parlez de guacamole. Le modèle fait une "moyenne" maladroite des sens, ce qui brouille sa compréhension.
{{% /hint %}}

## Le second pilier : Les Embeddings Contextuels (La révolution BERT)

C'est ici qu'interviennent les LLM modernes. Regardez attentivement les **Figures 2-14 et 2-15**. Contrairement à Word2Vec, un modèle comme BERT ou DeBERTa ne se contente pas de chercher le vecteur dans une table. Il fait passer le mot à travers ses couches de Transformers (l'attention, vue en section 1.3).

{{< bookfig src="44.png" week="02" >}}
{{< bookfig src="45.png" week="02" >}}

**Le résultat ?** Le mot "bank" n'aura pas le même vecteur s'il est suivi de "money" ou de "river". Le vecteur est **calculé dynamiquement** en fonction de l'environnement. C'est ce qu'on appelle la **contextualisation**. Un mot devient un caméléon : il change de couleur vectorielle selon le support sur lequel il se pose.

## La Géométrie du Sens : Similarité Cosinus

Comment savoir si deux mots sont proches ? On utilise la **Similarité Cosinus**. Comme le montre la **Figure 2-16**, on ne regarde pas la longueur des vecteurs, mais l'angle entre eux dans cet espace à haute dimension. 
*   Angle proche de 0° : Les mots sont synonymes ou très liés.
*   Angle de 90° : Les mots n'ont aucun rapport.
*   Angle de 180° : Les mots sont opposés (rare en langage naturel).

{{< bookfig src="101.png" week="02" >}}

{{% hint info %}}
🔑 **Je dois insister :** Cette propriété géométrique est la base de tous les moteurs de recherche modernes. On ne cherche plus des mots-clés, on cherche des vecteurs proches.
{{% /hint %}}

## Laboratoire de code : Créer des Embeddings avec Sentence-Transformers

Mettons cela en pratique. Nous allons utiliser un modèle léger et performant : `all-MiniLM-L6-v2`. Ce modèle transforme une phrase entière en un seul vecteur de 384 dimensions.

```python
# Installation : pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

# 1. Chargement du modèle (Optimisé pour Colab T4)
# Ce modèle est petit mais redoutable pour la similarité sémantique.
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Le chat dort sur le tapis.",
    "Un félin se repose sur la moquette.",
    "Le cours de l'action Apple est en hausse.",
    "J'aime manger des pommes bien rouges."
]

# 2. Encodage : Transformation en vecteurs
embeddings = model.encode(sentences)

print(f"Forme de la matrice d'embeddings : {embeddings.shape}") 
# (4, 384) -> 4 phrases, chacune représentée par 384 nombres.

# 3. Calcul de similarité entre la phrase 0 et la phrase 1
sim_chat = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarité entre 'chat' et 'félin' : {sim_chat.item():.4f}")

# 4. Comparaison avec la phrase 2 (finance)
sim_finance = util.cos_sim(embeddings[0], embeddings[2])
print(f"Similarité entre 'chat' et 'Apple stock' : {sim_finance.item():.4f}")
```
<!-- TODO: add colab link -->

{{% hint warning %}}
Observez les scores. La similarité entre la phrase sur le chat et celle sur le félin sera très élevée (proche de 0.8 ou 0.9), même si elles ne partagent *aucun* mot commun ! C'est la preuve que le modèle a compris le concept derrière les symboles.
{{% /hint %}}

## Applications Pratiques des Embeddings

Pourquoi passer autant de temps sur ce concept ? Parce qu'il est partout :
1.  **Recherche Sémantique** : Trouver un document même si la requête utilise des synonymes.
2.  **Clustering (Semaine 7)** : Regrouper automatiquement des milliers d'emails par thématique.
3.  **Systèmes de Recommandation** : Si vous aimez la chanson A, le système cherche la chanson dont l'embedding est le plus proche de A (voir Figure 2-17).

{{< bookfig src="53.png" week="02" >}}

4.  **Détection d'Anomalies** : Un texte dont le vecteur est très éloigné de tous les autres est probablement un spam ou une erreur.

## Éthique et Biais : La face cachée des vecteurs

{{% hint danger %}}
« Mes chers étudiants, écoutez-moi bien. » 
Parce que les embeddings apprennent à partir de nos textes, ils figent nos préjugés dans le marbre mathématique. 
*   **Stéréotypes** : Si le mot "technologie" est statistiquement plus associé aux hommes dans les textes du web, l'embedding de "technologie" sera géométriquement plus proche de "homme". 
*   **Conséquence** : Un algorithme de recrutement basé sur ces embeddings pourrait rejeter des CV de femmes simplement parce que leur profil est "mathématiquement" moins proche du vecteur "ingénieur".

🔑 **C'est votre responsabilité :** Avant d'utiliser un modèle d'embedding, vérifiez toujours sa provenance et testez ses biais avec des paires de mots sensibles. L'IA ne doit pas être un amplificateur d'injustices.
{{% /hint %}}

## Synthèse de la Semaine 2

« Nous avons parcouru un chemin immense aujourd'hui ! »
1.  Nous avons appris à découper le texte en **tokens** (atomes).
2.  Nous avons vu comment ces tokens sont traduits en **embeddings denses** (coordonnées).
3.  Nous avons compris la différence entre un vecteur **statique** (mort) et un vecteur **contextuel** (vivant et changeant).

« Vous tenez entre vos mains les clés de la compréhension des LLM. La semaine prochaine, nous monterons encore d'un cran : nous entrerons dans la salle des machines du Transformer pour voir exactement comment les têtes d'attention manipulent ces vecteurs pour créer de l'intelligence. »

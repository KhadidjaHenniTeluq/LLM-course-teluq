---
title: "2.4 Plongements lexicaux (Embeddings)"
weight: 5
---

## Donner une chair aux nombres : La naissance du vecteur de sens
Bonjour à toutes et à tous ! Nous arrivons au point culminant de cette semaine. Dans la section précédente, nous avons appris à découper le langage en petits index numériques. Nous savons transformer le mot "Cœur" en un numéro de dictionnaire, par exemple le `1254`. Mais posons-nous une question fondamentale : qu'est-ce que le chiffre `1254` sait de l'amour, de l'anatomie ou du courage ? Rien. Pour un ordinateur, `1254` est juste une étiquette, aussi vide de sens que le chiffre `1255`. 

> [!IMPORTANT]
📌 **Je dois insister :** pour que l'IA comprenne vraiment, nous devons passer du "numéro d'ordre" à la "position géographique 📍". 

Aujourd'hui, nous allons découvrir les **Embeddings**, ou plongements lexicaux. Nous allons apprendre comment chaque mot reçoit une "adresse" dans un espace à plusieurs centaines de dimensions, où la proximité physique reflète enfin la proximité de sens. 

Respirez, nous allons donner une âme mathématique aux jetons !

---
## 1. Le concept d'Embedding : Une carte du monde sémantique
Un embedding est une représentation d'un token sous la forme d'un vecteur dense de nombres réels. Au lieu d'un simple index, chaque mot est défini par une liste de coordonnées (généralement 768 ou 1024 dimensions dans les modèles modernes). 

Regardons la **Figure 2-6 : Embeddings associés au vocabulaire** .

{{< bookfig src="43.png" week="02" >}}

**Explication** : Cette illustration nous montre le lien physique entre le tokeniseur et le modèle. 
1.  Le tokeniseur fournit un ID (ex: 0, 1, ..., 50 257). 
2.  Le modèle possède une immense matrice de poids appelée "Embedding Matrix". 
3.  L'ID sert de numéro de ligne : si le token est `1`, le modèle va chercher la deuxième ligne de sa matrice. Cette ligne contient le vecteur (l'embedding) associé à ce mot. 

> [!TIP]
💡 **Mon intuition :** Imaginez que le langage soit une galaxie. Chaque mot est une étoile. 

> Les embeddings sont les coordonnées GPS (Latitude, Longitude, Altitude...) qui permettent de dire que l'étoile "Planète" est plus proche de l'étoile "Terre" que de l'étoile "Sandwich".

---
## 2. L'héritage de Word2Vec : La révolution des embeddings statiques
Pour comprendre comment ces vecteurs sont nés, nous devons revenir en 2013 avec l'algorithme **Word2Vec**. nous proposons de revisiter ses fondements à travers les figures de la semaine 1.

### La structure neuronale ( [Figure 1-6 - Semaine 1]({{< relref "section-1-1.md" >}}#fig-1-6))

**ℹ️ Explication** : Cette figure montre que les embeddings ne sont pas programmés par des humains, ils sont **appris**. 

On utilise un réseau de neurones très simple avec une couche cachée. 

> [!TIP]
💡 **La découverte majeure** : les poids de cette couche cachée, une fois l'entraînement fini, deviennent les embeddings eux-mêmes. Le sens est un sous-produit de l'apprentissage statistique.


### L'apprentissage par voisinage ([Figure 1-7 - Semaine 1]({{< relref "section-1-1.md" >}}#fig-1-7))

**ℹ️ Explication** : Comment apprend-on le sens ? Par le contexte ! La figure illustre une tâche de prédiction : "Le mot A et le mot B sont-ils voisins ?". 
*   On montre au modèle des milliers de phrases. 
*   Si "Chat" et "Miaule" apparaissent souvent ensemble, le modèle va ajuster leurs vecteurs pour qu'ils pointent dans la même direction. 

> [!TIP]
> *   💡 **L'intuition technique :** C'est ce qu'on appelle l'apprentissage contrastif. On rapproche les mots qui se fréquentent et on éloigne ceux qui n'ont rien à voir.


### La décomposition des propriétés ([Figure 1-8 - Semaine 1]({{< relref "section-1-1.md" >}}#fig-1-8))

**ℹ️ Explication** : C'est ici que la magie opère. La figure montre que chaque dimension du vecteur finit par capturer une propriété abstraite. 
*   Une colonne pourrait représenter l'aspect "Animal".
*   Une autre l'aspect "Royal".
*   Une autre le "Genre" (masculin/féminin).
Ainsi, le mot "Roi" aura une valeur élevée en "Royal" et "Homme", tandis que "Reine" sera élevée en "Royal" et "Femme". 
> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Dans la réalité, ces dimensions ne sont pas nommées ainsi. Elles sont des abstractions mathématiques que nous interprétons après coup.

### La géométrie du sens ([Figure 1-9 - Semaine 1]({{< relref "section-1-1.md" >}}#fig-1-9))

**ℹ️ Explication** : Si l'on réduit ces centaines de dimensions à seulement 2 (pour pouvoir les dessiner), on observe des grappes (clusters). "Chat", "Chien" et "Chiot" forment un petit village sémantique. "Banane" et "Pomme" en forment un autre, très loin du premier. 

> [!IMPORTANT]
‼️ **Je dois insister :** La similarité entre deux pensées humaines est devenue une simple question de **distance cosinus** entre deux points.


---
## 3. Les limites des embeddings statiques

**Pourquoi ne pas s'être arrêté à Word2Vec ? demanderez-vous.** 
> 👉 La réponse est simple -> Parce que Word2Vec souffre d'un défaut fatal : il est **statique**. 

Prenons un exemple : le mot "avocat".
1. "J'ai mangé un **avocat** mûr."
2. "Mon **avocat** a plaidé ma cause."

Dans Word2Vec, le token `avocat` n'a qu'un seul vecteur. C'est une moyenne confuse entre un fruit et un juriste. Le modèle est incapable de changer sa vision du mot selon la phrase/contexte. 

> [!NOTE]
🧱 **C'est le mur de la polysémie.** Pour le franchir, il nous fallait l'architecture Transformer.

---
## 4. L'avènement des Embeddings Contextuels
C'est ici que nous entrons dans l'ère des LLM modernes. 

Regardez la **Figure 2-7** .

{{< bookfig src="44.png" week="02" >}}

**ℹ️ Explication** : Contrairement aux modèles statiques, un LLM (comme BERT ou Phi-3) traite la phrase entière avant de produire le vecteur final d'un mot. 
*   Le modèle regarde les mots "alentour". 
*   Si le mot "mûr" est présent, il va "mélanger" le vecteur d' `avocat` avec des informations de nourriture. 
*   Le résultat est un **vecteur dynamique** qui n'existe que pour cette phrase précise.

Regardez la **Figure 2-8** .

{{< bookfig src="45.png" week="02" >}}

**ℹ️ Explication** : Cette figure détaille le flux. L'entrée est un embedding statique (une base brute), mais après être passé par les couches d'Attention (Semaine 3), il ressort transformé en embedding contextuel. 

> [!IMPORTANT]
✍🏻 **Notez bien cette distinction :** l'embedding de base est le "potentiel" du mot, l'embedding contextuel est sa "réalité" dans une phrase donnée.


---
## 5. Création et extraction : Le LLM comme extracteur de sens
Aujourd'hui, nous utilisons souvent les LLM non pas pour générer du texte, mais pour générer des vecteurs de haute qualité pour la recherche sémantique (Semaine 6) ou le clustering (Semaine 7).

### Laboratoire de code : Extraire des embeddings contextuels
Voici comment utiliser un modèle de pointe (DeBERTa ou MPNet) pour transformer vos phrases en vecteurs sur Google Colab.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Chargement d'un modèle optimisé pour les phrases
model = SentenceTransformer("all-mpnet-base-v2")

# 2. Nos phrases à transformer en vecteurs
sentences = [
    "The financial bank is closed today.",
    "I am walking along the river bank.",
    "The institution is out of money."
]

# 3. GÉNÉRATION DES EMBEDDINGS (Inférence)
embeddings = model.encode(sentences)

# 4. COMPARAISON (Similarité Cosinus)
# On compare la phrase 1 (Banque finance) avec la phrase 2 (Rive) 
# et avec la phrase 3 (Institution/Argent)
sim_1_2 = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
sim_1_3 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

print(f"Similarité 'Finance' vs 'Rive' : {sim_1_2:.4f}")
print(f"Similarité 'Finance' vs 'Institution' : {sim_1_3:.4f}")
```

> [!IMPORTANT]
‼️ Observez les résultats. Vous verrez que la phrase 1 est beaucoup plus "proche" mathématiquement de la phrase 3, alors qu'elles ne partagent aucun mot important (sauf le mot "banque" sous-entendu). 

> C'est la preuve que les embeddings contextuels ont capturé le **concept** et non seulement les lettres.

---
## 6. Applications pratiques : Au-delà du NLP
Les embeddings ne servent pas qu'à traiter du texte. nous montrons une application fascinante : les **Systèmes de recommandation**.

**Le cas Spotify/Netflix** :
Imaginez que nous traitions une "playlist" de musique comme une "phrase", et chaque "chanson" comme un "mot".
*   Si des millions d'utilisateurs écoutent la chanson A juste après la chanson B, l'IA va créer des embeddings proches pour A et B.
*   📊 **Le résultat** : Quand vous écoutez un artiste, Spotify regarde quels sont les vecteurs les plus proches dans l'espace multidimensionnel pour vous proposer votre prochaine découverte. 

C'est le même algorithme Word2Vec appliqué aux comportements humains !

---
## 7. Éthique et Responsabilité : Le biais est une distance

> [!CAUTION]
‼️ Mes chers étudiants, l'espace vectoriel n'est pas un paradis mathématique neutre. 

Comme nous l'avons vu, les embeddings sont le reflet fidèle de nos préjugés. 
1.  **Stéréotypes de genre** : Dans de nombreux modèles, le vecteur "Secrétaire" est géométriquement plus proche de "Femme" et "Ingénieur" plus proche de "Homme". 
2.  **Biais culturels** : Les modèles entraînés sur le web occidental peuvent associer des sentiments négatifs à des prénoms ou des cultures qu'ils connaissent mal, simplement par manque de diversité dans les données. 
💥 **Conséquence technique :** Si vous utilisez ces embeddings pour filtrer des CV ou accorder des prêts bancaires, votre IA sera injuste par construction géométrique.

> [!TIP]
✅️ **Mon conseil** : Avant d'utiliser un modèle d'embedding en production, visualisez vos clusters ! Si vous voyez que des groupes de population sont isolés ou injustement étiquetés par la machine, c'est que vos données d'entraînement sont polluées. L'IA responsable commence par l'audit de sa géométrie.


---
## Synthèse de la semaine
Nous avons parcouru un chemin immense cette semaine. Nous avons appris que :
*   Le texte doit être découpé intelligemment (Tokenisation) pour être digestible.
*   Chaque morceau de texte devient un point dans un espace géant (Embeddings).
*   La force des LLM réside dans le fait que ces points sont **mobiles** et s'adaptent au contexte de la phrase.

> [!TIP]
✉️ **Mon message** : Vous avez maintenant les briques (tokens) et le ciment (embeddings). 

> Vous comprenez comment la machine transforme le verbe en vecteur. C'est une étape de géant. Mais une question demeure : comment ces vecteurs "discutent-ils" entre eux au sein du modèle pour créer une pensée cohérente ? C'est le secret de l'**Architecture Transformer** que nous allons ouvrir ensemble la semaine prochaine. Félicitations pour votre persévérance !
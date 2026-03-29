---
title: "1.2 Limites des architectures séquentielles : RNN et LSTM"
weight: 3
---

{{< katex />}}

## Le règne de la récurrence : Quand l'IA apprend à lire de gauche à droite
Bonjour à toutes et à tous ! Je suis ravie de vous retrouver. Dans notre section précédente (1.1), nous avons découvert comment transformer des mots en "adresses mathématiques" dans un espace vectoriel. C'était une avancée majeure, mais restons lucides : une liste de mots n'est pas une phrase. Le langage est une mélodie, une séquence où l'ordre des notes change tout le sens. 

> [!IMPORTANT]
📌 **Je dois insister :** pendant près de vingt ans, le monde de l'IA a été dominé par une idée fixe : pour comprendre le langage, la machine doit le traiter exactement comme nous, mot après mot, de gauche à droite. C'est ce que nous appelons l'ère des **Réseaux de Neurones Récurrents (RNN)**. 

Mais comme nous allons le voir, cette imitation de la lecture humaine a fini par devenir une prison technologique. Respirez, nous allons analyser pourquoi ces géants aux pieds d'argile ont dû céder la place.

---
## L'intuition du RNN : La mémoire de travail
Un RNN fonctionne sur un principe de boucle. Imaginez que vous lisiez un livre. Pour comprendre la page 10, vous avez besoin de vous souvenir de ce qui s'est passé à la page 9. 
*   Le modèle reçoit un mot ($x_t$).
*   Il possède un "état caché" ($h_t$), qui est sa mémoire interne.
*   À chaque nouveau mot, il mélange l'information du mot actuel avec sa mémoire du passé pour mettre à jour sa compréhension globale.

C'est une structure magnifique sur le papier, car elle respecte la nature temporelle du langage. Mais en pratique, elle s'est heurtée à deux murs infranchissables : **le goulot d'étranglement sémantique** et **la mort du signal (le gradient)**.

---
## L'architecture Encodeur-Décodeur
Pour des tâches comme la traduction, nous avons utilisé une structure en deux blocs, illustrée par la **Figure 1-10 : Architecture RNN encoder-decoder** .

{{< bookfig src="15.png" week="01" >}}

**Explication de la Figure 1-10** : Cette illustration est capitale. Elle montre deux cerveaux distincts.
1.  **L'Encodeur (à gauche)** : Son rôle est de "digérer" la phrase source (ex: "I love llamas"). Il traite "I", puis "love", puis "llamas". À chaque étape, sa mémoire interne s'enrichit.
2.  **Le Vecteur de Contexte (Le centre)** : C'est le point critique. Une fois que l'encodeur a fini de lire, il doit résumer TOUT le sens de la phrase dans un seul et unique vecteur final.
3.  **Le Décodeur (à droite)** : Il reçoit ce vecteur et tente de reconstruire la phrase dans une autre langue (ex: "Ik hou van lama's").

> [!TIP]
💡 **Notez bien cette intuition :** L'encodeur est comme un traducteur qui écoute une phrase de 5 minutes, prend une seule petite note sur un post-it, et donne ce post-it au décodeur pour qu'il réécrive le discours entier. 

---
## Le Goulot d'étranglement
C'est ici que l'architecture montre ses limites physiques. Regardez la **Figure 1-11 : Context embedding dans RNN** . 

{{< bookfig src="17.png" week="01" >}}

**Explication de la Figure 1-11** : La figure montre visuellement que la quantité d'information que l'on peut faire passer entre l'encodeur et le décodeur est fixe. 
*   Si la phrase fait 3 mots ("I love you"), le vecteur de contexte est à l'aise.
*   Si la phrase fait 50 mots, avec des propositions subordonnées complexes, le vecteur de contexte sature. 

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent qu'il suffit d'augmenter la taille du vecteur. Mais en augmentant la taille, on multiplie les paramètres et le modèle devient impossible à entraîner. 

> C'est ce qu'on appelle le **Bottleneck Problem** (le goulot d'étranglement). L'information est littéralement écrasée et perdue avant d'atteindre le décodeur.

---
## Le processus Autorégressif
Une fois que le décodeur commence à parler, il suit une logique particulière. La **Figure 1-12 : Processus autoregressive** nous montre que l'IA ne génère pas la phrase d'un coup.

{{< bookfig src="16.png" week="01" >}}

**Explication de la Figure 1-12** : 
*   Étape 1 : Le modèle prédit "Ik". 
*   Étape 2 : Il prend "Ik" comme nouvelle entrée pour prédire "hou". 
*   Étape 3 : Il prend "Ik hou" pour prédire "van".

> [!IMPORTANT]
🔑 **C'est un concept non-négociable :** presque tous les LLM, même les plus modernes, sont encore **autorégressifs**. 

> Ils sont prisonniers de cette boucle où la sortie précédente devient l'entrée suivante. Le problème des RNN est que cette boucle est trop dépendante de la qualité du premier vecteur de contexte.

---
## La Disparition du Gradient : Pourquoi l'IA oublie les débuts de phrase
Mes chers étudiants, imaginez que vous fassiez une partie de téléphone arabe (Chinese Whispers) avec 100 personnes. À la fin de la chaîne, le message original est déformé. Dans un RNN, c'est la même chose.

Lors de l'entraînement, nous calculons une erreur (la différence entre ce que l'IA a dit et la vérité). Cette erreur doit "remonter" le temps pour dire aux neurones du début de la phrase : "Hé, vous avez mal interprété le sujet !". 
*   **Vanishing Gradient** (Disparition) : À force de remonter les étapes, le signal mathématique devient si petit qu'il s'évapore. Les neurones du début de la phrase n'apprennent jamais rien. Le modèle oublie le début du texte.
*   **Exploding Gradient** (Explosion) : À l'inverse, le signal peut devenir infini et faire "planter" les calculs de la machine.

---
## La solution partielle : LSTM (Long Short-Term Memory)
En 1997, Hochreiter et Schmidhuber ont inventé le **LSTM** pour tenter de sauver les RNN. Imaginez que dans chaque neurone, nous ajoutions une "autoroute de l'information" protégée par des portes.
*   **La porte d'oubli** : Elle décide quelle information du passé est devenue inutile (ex: changer de paragraphe).
*   **La porte d'entrée** : Elle décide quelle nouvelle information est digne d'être mémorisée.
*   **La porte de sortie** : Elle filtre ce que l'on montre au reste du réseau.

> [!NOTE]
✍🏻 **Je dois insister :** Les LSTM ont permis de passer de 10 mots de mémoire à environ 100 ou 200 mots. 

> C'était un progrès immense, mais pour lire un livre ou comprendre un contrat juridique, c'était encore bien trop peu. L'architecture restait désespérément **séquentielle**.

---
## Pourquoi la récurrence empêche le "Scaling" ?
C'est le point de vue de l'ingénieur de production. Comme chaque mot a besoin du résultat du mot précédent pour être calculé, on ne peut pas utiliser la pleine puissance des cartes graphiques (GPU).
*   Les GPU adorent faire des milliers de calculs **en même temps** (parallélisation).
*   Les RNN obligent le GPU à attendre : "J'ai fini le mot 1, donne-moi le mot 2...". 
C'est pour cela que nous ne pouvions pas entraîner de modèles sur l'intégralité d'Internet avec des RNN. C'était tout simplement trop lent.

---
## Laboratoire de code : Structure d'un RNN en PyTorch
Voici une implémentation simplifiée pour que vous puissiez "voir" la dépendance séquentielle. Notez bien le passage de l'état caché (`hidden`).

```python
import torch
import torch.nn as nn

# Un RNN simple pour comprendre la séquence
# Testé sur Colab T4
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # La cellule RNN qui mélange Entrée + Passé
        # batch_first=True permet de traiter (Batch, Sequence, Features)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # Couche de sortie pour prédire le mot suivant
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialisation du premier état caché à zéro
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # Passage dans le RNN : il traite toute la séquence
        # Mais en interne, il fait une boucle mot par mot !
        out, hn = self.rnn(x, h0)
        
        # On ne garde que le résultat du dernier mot pour la prédiction
        # C'est notre fameux "Vecteur de Contexte"
        context_vector = out[:, -1, :]
        return self.fc(context_vector)

```

> [!IMPORTANT]
⚠️ Observez la ligne `out[:, -1, :]`. Nous jetons littéralement tous les calculs des mots précédents pour ne garder que le dernier. Vous comprenez maintenant pourquoi l'information se perd !

---
## Éthique et Biais : Le biais de primauté

> [!CAUTION]
⚖️ Mes chers étudiants, même l'architecture dicte nos préjugés.

Dans un RNN, les mots du début de la phrase ont moins d'influence sur la fin que les mots récents (à cause de la disparition du gradient). Si vous entraînez un modèle de justice sur des dossiers, et que le RNN "oublie" le contexte initial de l'affaire pour ne se concentrer que sur les derniers mots techniques, vous créez une IA injuste par amnésie. 

> [!IMPORTANT]
🔑 **La responsabilité de l'ingénieur est de garantir une attention équitable à toute la donnée.** 

---
## Synthèse de la section
Nous avons vu comment les RNN et LSTM ont tenté de capturer la mélodie du langage en traitant les mots un par un. Nous avons compris leurs trois péchés originels :
1.  **Le Goulot d'étranglement** : Tout compresser dans un seul point.
2.  **L'Oubli** : La perte du signal au fil du temps.
3.  **La Lenteur** : L'incapacité à calculer en parallèle.

> [!TIP]
✉️ **Mon message** : Imaginez la frustration des chercheurs en 2016 ! Ils avaient des données, ils avaient des GPU puissants, mais leurs modèles étaient bloqués par cette structure séquentielle. 

> C'est dans ce climat de blocage qu'est née une idée radicale : et si nous arrêtions de lire dans l'ordre ? Et si nous donnions au modèle un moyen de "téléporter" son attention n'importe où, instantanément ? C'est la naissance des Transformers.
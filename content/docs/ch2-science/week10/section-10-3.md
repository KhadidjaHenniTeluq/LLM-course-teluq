---
title: "10.3 Architecture avancée : BLIP-2"
weight: 4
---


## Le traducteur universel : Pourquoi nous avons besoin d'un pont
Bonjour à toutes et à tous ! Nous arrivons maintenant au cœur battant de la multimodalité moderne. Dans la section 10.2, nous avons appris comment un Vision Transformer (ViT) "découpe" le monde en patches pour le transformer en nombres. C'est une prouesse, mais restons lucides : ces nombres ne sont pas encore du langage. Ils sont une suite de coordonnées visuelles brutes. 
> [!NOTE]
🔑 **Je dois insister :** si vous essayez de brancher directement un ViT sur un LLM comme Llama-3, vous obtiendrez un silence total. Pourquoi ? Parce qu'ils parlent deux langues mathématiques radicalement différentes. 

C'est ce que nous appelons le **Modality Gap** (l'écart de modalité). Aujourd'hui, nous allons étudier l'une des architectures les plus géniales pour combler ce fossé : **BLIP-2**. Respirez, nous allons découvrir comment construire un interprète capable de traduire les pixels en concepts pour que notre IA puisse enfin discuter avec nous d'une photo.

L'enjeu de BLIP-2 (*Bootstrapping Language-Image Pre-training*) est de taille. Entraîner un modèle multimodal géant à partir de zéro coûte des millions 💵 et nécessite des mois de calcul. L'idée révolutionnaire des chercheurs de Salesforce a été de dire : "Et si nous gardions nos meilleurs modèles actuels (le ViT pour voir et le LLM pour parler) totalement **gelés**, et que nous n'entraînions qu'un petit traducteur intelligent au milieu ?". 

Comme vous pouvez le voir sur la **Figure 10-12 : Le Q-Former comme pont** , l'architecture repose sur cette brique centrale appelée le **Q-Former**. Cette figure nous montre que le ViT à gauche et le LLM à droite ne changent pas ; seul le Q-Former est "entraînable". C'est un gain d'efficacité colossal. 

{{< bookfig src="213.png" week="10" >}}

---
## L'ingénierie du Q-Former : Interroger le visuel

> [!IMPORTANT]
🔑 **C'est le concept le plus sophistiqué de ce chapitre :** Le Q-Former (*Querying Transformer*) n'est pas un simple adaptateur linéaire. C'est un Transformer à part entière qui joue le rôle d'un "enquêteur". 

Regardez attentivement la **Figure 10-13 : Processus d'entraînement en deux étapes** . Pour comprendre le Q-Former, imaginez qu'il possède une petite collection de "Jetons de questions" (Learnable Queries), généralement 32 tokens. Ces jetons ne correspondent à aucun mot humain au départ. Leur mission est d'aller "frapper à la porte" du Vision Transformer pour lui demander : "Qu'y a-t-il d'intéressant dans cette image pour un modèle de langage ?". 
*   **Self-Attention** : Les 32 jetons discutent entre eux pour ne pas poser la même question.
*   **Cross-Attention** : Les jetons vont piocher des informations dans les patches du ViT (vus en 10.2).
*   **Résultat** : Ils ressortent avec un résumé compressé de l'image, parfaitement digeste pour un LLM. 

{{< bookfig src="214.png" week="10" >}}

> [!TIP]
🔑 **Mon intuition :** Le Q-Former est comme un journaliste qui regarde une scène de crime (l'image) et qui n'en rapporte que les 32 indices les plus importants pour que le rédacteur en chef (le LLM) puisse écrire son article. Il élimine le bruit (les pixels inutiles) pour ne garder que la sémantique.

---
## Étape 1 : Apprendre à comprendre (Representation Learning)
L'entraînement de ce "pont" se fait en deux phases cruciales. Comme l'illustre la **Figure 10-14 : Étape 1 de l'entraînement de BLIP-2** , nous commençons par apprendre au Q-Former à aligner le visuel et le textuel *sans le LLM*.

{{< bookfig src="215.png" week="10" >}}

Le modèle s'exerce sur trois tâches simultanées pour devenir un expert en sémantique :
1.  **Image-Text Contrastive Learning** : Comme pour CLIP (section 10.1), on force le résumé du Q-Former à être mathématiquement proche de la légende de l'image.
2.  **Image-Text Matching** : On demande au modèle : "Est-ce que cette légende décrit vraiment cette photo ?". C'est un test de vérification binaire (Oui/Non).
3.  **Image-grounded Text Generation** : On force le Q-Former à prédire les mots de la légende à partir de ses jetons visuels. 

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** On ne cherche pas encore à "discuter" avec l'image. On cherche simplement à ce que les 32 jetons du Q-Former capturent l'essence de ce qui est écrit dans la légende. 

---
## Étape 2 : Apprendre à parler (Generative Learning)
Une fois que le Q-Former a compris comment extraire le sens, il faut le connecter à la "bouche" du système : le LLM. C'est ce que montre la **Figure 10-15 : Connexion au LLM** .

{{< bookfig src="216.png" week="10" >}}

> [!NOTE]
🔑 **La prouesse technique :** On ajoute une simple **couche de projection linéaire** à la sortie du Q-Former. Son rôle est de transformer les 32 vecteurs visuels pour qu'ils fassent la même taille que les vecteurs de mots du LLM. 
*   Le LLM reçoit alors ces 32 jetons comme s'il s'agissait du début d'une phrase. 
*   Pour le LLM, c'est comme si nous lui chuchotions : "Voici le contexte visuel que tu dois utiliser pour ta réponse".
*   On entraîne alors cette couche de projection pour que le LLM génère la réponse parfaite à une question sur l'image.

Regardez la **Figure 10-16 : Architecture BLIP-2 complète** . Elle résume la symphonie : l'image entre dans le ViT, le Q-Former l'interroge, le résultat est projeté vers le LLM, et le LLM répond. Tout cela en ne modifiant que le "traducteur" central.

{{< bookfig src="217.png" week="10" >}}

---
## Cas d'usage : Visual Question Answering (VQA)
L'application la plus spectaculaire de BLIP-2 est le **VQA**. Contrairement au simple *captioning* (légendage), le VQA permet une interaction dynamique. 

Comme le montre l'exemple de la **Figure 10-17** , vous pouvez demander : "De quelle couleur est la voiture ?" ou "Pourquoi cette image est-elle drôle ?". Le modèle utilise ses jetons visuels pour aller chercher l'information spécifique demandée dans la question. 

> [!TIP]
🔑 **C'est le début de l'IA capable de raisonnement visuel.** 

{{< bookfig src="212.png" week="10" >}}

---
## Laboratoire de code : Discussion avec une image (VQA)
Voyons comment mettre en œuvre BLIP-2 sur votre GPU T4. Nous allons utiliser la version "mini" basée sur OPT-2.7B pour qu'elle tienne confortablement dans votre mémoire VRAM.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install transformers accelerate bitsandbytes pillow requests

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import requests
import torch

# 1. CHARGEMENT DU TRADUCTEUR ET DU GÉNÉRATEUR
# On utilise la version OPT-2.7B qui est un bon équilibre pour le T4
model_id = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_id)
# Utilisation de la demi-précision (float16) pour économiser la mémoire
model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# 2. PRÉPARATION DE L'IMAGE
url = "./images/car.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 3. LE PROMPT MULTIMODAL
# On prépare une question sur l'image
question = "Question: What is unique about the car in this photo? Answer:"

# 4. LE PONT Q-FORMER EN ACTION
# Le processeur prépare l'image pour le ViT et le texte pour le Q-Former
inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)

# 5. GÉNÉRATION
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    # Décodage pour obtenir la réponse humaine
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(f"Image chargée : {url}")
print(f"IA : {answer}")

```

> [!IMPORTANT]
⚠️ Si vous recevez une erreur de type `OutOfMemory`, c'est que vous avez oublié le `torch_dtype=torch.float16`. 
> [!IMPORTANT]
🔑 **Je dois insister :** en multimodal, la mémoire est votre ressource la plus précieuse. Le chargement en 16-bit divise par deux le poids du modèle sur votre carte graphique.

---
## Éthique et Responsabilité : Les hallucinations visuelles

> [!CAUTION]
⚠️ Mes chers étudiants, l'IA peut parfois "voir" ce qu'elle a envie de voir.

Parce que BLIP-2 s'appuie sur un LLM puissant pour générer du texte, il souffre du même problème que nous avons vu en Semaine 5 : l'imagination statistique.
1.  **Le biais de confirmation** : Si vous demandez "Est-ce qu'il y a un chat dans cette cuisine ?" sur une photo vide, le modèle, influencé par la probabilité statistique que les chats soient dans les cuisines, pourrait répondre "Oui" avec assurance. 
2.  **L'illusion de la description** : la **Figure 10-18 : Test de Rorschach avec BLIP-2** montre que si l'image est ambiguë, le modèle plaque ses propres schémas mentaux appris durant son entraînement. Il ne "voit" pas l'image, il essaie de lui donner un sens qui ressemble aux millions d'autres images qu'il a déjà analysées. 

{{< bookfig src="218.png" week="10" >}}


> [!TIP]
🔑 **Mon message** : Un système multimodal est une chaîne de confiance. 

> Si le ViT rate un détail, le Q-Former ne pourra pas le traduire, et le LLM inventera la suite. Ne prenez jamais la réponse d'un modèle comme BLIP-2 pour une preuve légale ou médicale sans une double vérification humaine. L'IA est une aide à la perception, pas un substitut à l'observation.

---
Vous maîtrisez maintenant l'architecture de pointe de l'IA multimodale. Vous savez comment un pont peut relier le regard (ViT) à la parole (LLM). C'est une étape majeure. Dans la dernière section de cette semaine ➡️, nous conclurons en explorant les limites de ces modèles et les applications industrielles qui transforment déjà notre quotidien.
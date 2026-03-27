---
title: "10.4 Applications pratiques et limites"
weight: 5
---

## L'IA face au monde physique : De la contemplation à l'action
Bonjour à toutes et à tous ! Nous voici arrivés au sommet du 2ème chapitre (Sciences des LLMs). Nous avons compris comment l'alignement (section 10.1) et l'architecture (section 10.2 et 10.3) permettent à une machine de "voir". 

Mais maintenant, posons-nous la question qui fâche : à quoi cela sert-il vraiment dans la "vraie vie" ? Et surtout, pourquoi ne pouvons-nous pas encore confier nos vies à ces yeux artificiels ? 

> [!IMPORTANT]
🔑 **Je dois insister :** en multimodalité, l'erreur n'est plus seulement textuelle, elle devient perceptive.

> Une IA qui confond un champignon comestible avec un vénéneux sur une photo n'est pas seulement "imprécise", elle est dangereuse. 

Aujourd'hui, nous allons explorer le champ des possibles tout en gardant nos pieds (et nos yeux) bien ancrés sur terre.

---
## Cas d'usage 1 : Le légendage automatique (Image Captioning)
La première application, et sans doute la plus répandue, est l' **Image Captioning**. Comme nous l'avons vu en section 10.3 avec la **Figure 10-17 : Exemple de modèle multimodal** , le but est de transformer un signal visuel en une description textuelle fluide.

{{< bookfig src="212.png" week="10" >}}

*   **Le contenu de la Figure 10-17** : Cette illustration nous montre une interaction en deux temps. Dans le premier bloc, l'utilisateur présente une photo d'une voiture de sport sur une route au coucher du soleil. Le modèle BLIP-2 répond : "A sports car driving on the road at sunset". 

> [!NOTE]
🔑 **Notez bien cette intuition :** ce n'est pas une simple détection d'objets (étiquettes "voiture", "soleil"). C'est une synthèse narrative qui lie les objets par des actions ("driving") et un contexte temporel ("sunset").

*   **Applications industrielles** : 
    *   **E-commerce** : Générer automatiquement des fiches produits à partir de photos (ex: "Chaussure de course bleue avec semelle blanche"). 
    *   **Accessibilité** : Décrire le monde en temps réel pour les personnes malvoyantes. 
    *   **Gestion de contenu** : Indexer des millions d'images dans une base de données pour pouvoir les retrouver par une recherche textuelle (section 6.1).

---
## Cas d'usage 2 : Le dialogue visuel et le VQA
Le niveau supérieur est le **Visual Question Answering (VQA)**. Ici, le modèle ne se contente pas de décrire, il analyse.

Regardez la suite de la **Figure 10-17** . L'utilisateur pose une question de suivi : "What would it take to drive such a car?" (Que faudrait-il pour conduire une telle voiture ?). Le modèle répond avec une pointe d'humour et de réalisme : "A lot of money and time". 

> [!IMPORTANT]
🔑 **Je dois insister sur cette prouesse :** le modèle a dû extraire le concept sémantique de "voiture de luxe" de l'image, le lier à ses connaissances sur le coût de la vie (mémoire interne du LLM) et formuler une réponse cohérente. C'est ce qu'on appelle la **génération conditionnée par l'image**.

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** On pourrait croire que le modèle "raisonne" comme un humain.

> En réalité, il utilise le Q-Former (section 10.3) pour transformer les pixels de la voiture de sport en un vecteur proche du concept "richesse" dans son espace sémantique. C'est une association statistique de haut vol.

---
## Limite majeure 1 : Les Hallucinations visuelles et le test de Rorschach
C'est le point le plus critique de cette section. Regardez attentivement la **Figure 10-18 : Test de Rorschach avec BLIP-2**.

{{< bookfig src="218.png" week="10" >}}

*   **Explication de la Figure 10-18** : Les auteurs ont présenté une tache d'encre symétrique et ambiguë au modèle. BLIP-2 répond avec une assurance totale : "A black and white ink drawing of a bat" (Un dessin à l'encre noir et blanc d'une chauve-souris). 
*   **Le problème** : Une tache de Rorschach n'est "rien". C'est une forme abstraite. Mais le modèle, entraîné à toujours donner la suite la plus probable, "hallucine" une interprétation. 

> [!CAUTION]
⚠️ Mes chers étudiants, comprenez bien ceci 👇! 

> Si vous présentez une image de mauvaise qualité ou une situation jamais vue à un modèle multimodal, il ne vous dira pas "Je ne sais pas". Il va tenter de projeter ses propres biais d'entraînement sur l'image. C'est la **Pérotomanie de l'IA** : la conviction d'avoir raison sur une perception fausse.

---
## Limite majeure 2 : La "Cécité" aux détails fins
Comme nous l'avons appris en section 10.2, les images sont découpées en patches (carrés de 16x16).

> [!IMPORTANT]
🔑 **Conséquence technique :** Si une information cruciale est plus petite qu'un patch (par exemple, une petite fissure sur une pièce industrielle ou un petit signe clinique sur une radio), le modèle risque de l'ignorer totalement. Le processus de compression du Q-Former, bien qu'efficace, sacrifie les détails au profit de la sémantique globale. 

> [!WARNING]
⚠️ **Avertissement** : N'utilisez jamais un modèle comme BLIP-2 pour du contrôle qualité de haute précision ou de la lecture de micro-caractères sans une architecture spécifique.

---
## Limite majeure 3 : Le biais de corrélation
Si une image montre une cuisine, le modèle "s'attend" statistiquement à y voir une femme ou des ustensiles. 
*   *Exemple* : Si on lui montre un homme faisant la cuisine dans un environnement sombre, le modèle pourrait légender l'image comme "Une femme préparant le dîner" simplement parce que la probabilité statistique du mot "femme" est plus forte dans ce contexte visuel. 

> [!CAUTION]
‼️ **C'est le danger du miroir déformant :** l'IA ne voit pas ce qui est là, elle voit ce qui est *probable* d'être là d'après ses millions d'images d'entraînement.

---
## Laboratoire de code : Création d'une interface interactive
Pour conclure cette semaine, je veux que vous sachiez comment créer un outil que vous pourrez montrer à vos collègues. Nous allons utiliser `ipywidgets` pour créer un chatbot multimodal miniature.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install ipywidgets pillow transformers requests torch

import ipywidgets as widgets
from IPython.display import display, HTML
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# 1. INITIALISATION (Comme en section 10.3)
model_id = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# 2. CRÉATION DE L'INTERFACE
# Créer un champ texte et un bouton pour interroger une image chargée
output_area = widgets.Output()
input_text = widgets.Text(placeholder="Posez une question sur l'image...")
button = widgets.Button(description="Demander à l'IA")

def on_button_clicked(b):
    with output_area:
        output_area.clear_output()
        # On utilise une image de test par défaut
        url = "http://images.cocodataset.org/val2017/000000039769.jpg" # 2 chats
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        display(image.resize((300, 200)))
        
        # Préparation du prompt VQA
        prompt = f"Question: {input_text.value} Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
        
        # Inférence
        out = model.generate(**inputs, max_new_tokens=30)
        answer = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        
        print(f"\nAssistant : {answer}")

button.on_click(on_button_clicked)
display(widgets.VBox([input_text, button, output_area]))
```

---
## Éthique et Société : Les nouveaux risques de la vision artificielle
> [!CAUTION]
⚠️ Mes chers étudiants, le pouvoir de "voir" s'accompagne d'une responsabilité de surveillance. 

1.  **Vie privée** : Un modèle multimodal peut identifier des lieux, des plaques d'immatriculation ou des visages sans que l'utilisateur en soit conscient. 
2.  **Manipulation (Deepfakes)** : La multimodalité facilite la création de contenus trompeurs. Si une IA peut parfaitement décrire une fausse image, elle donne de la crédibilité à un mensonge. 
3.  **Responsabilité juridique** : Si une IA multimodale commet une erreur d'analyse visuelle dans une voiture autonome, qui est responsable ? Le créateur du ViT ? L'ingénieur qui a fait le fine-tuning du LLM ? Le fournisseur de données ?

> [!TIP]
🔑 **Mon message** : Nous avons ouvert les yeux de l'IA, mais n'oublions pas d'ouvrir les nôtres. La multimodalité est une étape vers une IA plus humaine, plus proche de notre façon de percevoir. Mais elle reste une machine de probabilités. Utilisez-la pour amplifier votre regard, pas pour le remplacer.

---
Nous avons terminé cette immense semaine ! Vous savez désormais comment aligner le texte et l'image, comment transformer des pixels en séquences et comment construire un traducteur entre ces deux mondes. 

La semaine prochaine ➡️, nous reviendrons à la racine de la performance : comment adapter ces modèles géants à VOS besoins spécifiques. Bienvenue dans le monde du **Fine-tuning supervisé** et de la méthode **LoRA**. Mais d'abord, place au laboratoire final !

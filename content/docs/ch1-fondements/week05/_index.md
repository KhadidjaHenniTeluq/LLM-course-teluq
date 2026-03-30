---
title: "Semaine 5 : Modèles génératifs (decoder-only)"
weight: 5
bookCollapseSection : true
---


# 📝 Semaine 5 : Modèles génératifs (decoder-only)

Bonjour à toutes et à tous ! Je suis ravie de vous retrouver pour cette étape que beaucoup d'entre vous attendaient avec impatience. Jusqu'ici, nous avons appris à découper le texte et à le faire comprendre par la machine. Aujourd'hui, nous basculons du côté de la création. 

> [!IMPORTANT]
✍🏻 **Je dois insister :** nous allons étudier les modèles qui "parlent", ceux qui ont déclenché la révolution que nous vivons. Mais attention, derrière la fluidité de la famille **GPT**, il n'y a pas de conscience, seulement une prodigieuse maîtrise des probabilités.

Préparez-vous à apprendre l'art de murmurer à l'oreille des géants !

---

## 📖 Description générale
Cette semaine marque notre entrée dans le monde des modèles **Decoder-only**. Nous allons explorer la généalogie de la famille **GPT** (*Generative Pre-trained Transformer*), de la preuve de concept GPT-1 à l'omniscience de GPT-4. Nous décortiquerons la mécanique **autorégressive** : comment un modèle prédit le prochain mot, encore et encore, pour construire un récit. Nous ferons une distinction capitale entre les **Base Models** (savants bruts) et les **Chat/Instruct Models** (assistants alignés). Enfin, nous apprendrons à piloter ces modèles grâce aux paramètres de génération (**Température**, **Top-P**) pour équilibrer précision et créativité.

---
## 🧠 Pré-requis importants
Pour maîtriser cette semaine, vous devez avoir solidement acquis :
1.  **L'architecture Transformer (Semaine 3)** : Particulièrement le concept de "Self-Attention" (que nous utiliserons ici en mode masqué).
2.  **La Tokenisation (Semaine 2)** : Comprendre que le modèle ne prédit pas des lettres, mais des index de dictionnaire.
3.  **Probabilités de base** : Comprendre ce qu'est une distribution (quel mot a le plus de chances de sortir).

**Ressources pour réviser les pré-requis :**
*   🎥 **CodeEmporium (YouTube)** : [Transformers Explained](https://www.youtube.com/watch?v=TQQlZhbC5ps) .
*   💻 **Hugging Face Course** : [How do Transformers work?](https://huggingface.co/learn/nlp-course/chapter1/1) .

---

## 📚 Ressources utiles pour les concepts de la semaine

### 🌐 Articles et Blogs de référence
*   **Jay Alammar** : [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) – Le guide visuel indispensable pour comprendre comment le décodeur génère du texte.
*   **Hugging Face Blog** : [How to generate text: using different decoding methods](https://huggingface.co/blog/how-to-generate) – Une explication technique parfaite de la température, du Top-K et du Top-P.
*   **PromptingGuide.ai** : [Introduction au Prompt Engineering](https://www.promptingguide.ai/fr) – Pour comprendre comment parler aux modèles Instruct.

### 📺 Vidéos recommandées
*   🎥 **Andrej Karpathy** : [Let's build GPT: from scratch, in code.](https://www.youtube.com/watch?v=kCc8FmEb1nY) – Une vidéo légendaire pour comprendre le "Next Token Prediction".

### 🛠️ Outils pratiques
*   **Hugging Face Documentation** : [Text Generation Strategies](https://huggingface.co/docs/transformers/v4.39.0/en/main_classes/text_generation) – La référence pour coder vos paramètres.

---

> [!NOTE]
‼️ **Mon conseil** : Un modèle génératif est comme un miroir de ses données. 

> ⚠️ il est capable de simuler la logique, mais il peut aussi "halluciner" des faits avec une assurance totale. Ne confondez jamais la fluidité du style avec la véracité de l'information. En tant qu'experts, votre rôle est de rester les gardiens de la réalité !

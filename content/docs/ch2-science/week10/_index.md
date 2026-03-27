---
title: "Semaine 10 : LLM multimodaux"
weight: 5
---

# 📝 Semaine 10 : LLM multimodaux

Bonjour à toutes et à tous ! Je vous présente mes excuses pour ces contretemps techniques ; en IA comme en pédagogie, la vérification est la mère de la sûreté. 


🔑 **Je dois insister :** cette semaine est celle de la convergence. Jusqu'ici, nos modèles étaient "aveugles", enfermés dans le texte. Aujourd'hui, nous leur donnons la vue. Imaginez un modèle capable non seulement de lire une ordonnance, mais aussi d'analyser l'imagerie médicale correspondante. Nous allons briser la prison du langage pur pour entrer dans l'ère de la perception globale. Préparez-vous, car c'est ici que l'IA devient véritablement impressionnante !

---

## 📖 Description générale
Cette semaine est dédiée aux **Modèles de Langage Multimodaux** (*Multimodal Large Language Models*). Nous allons explorer comment les architectures Transformers s'adaptent pour traiter simultanément du texte et des images. Nous étudierons **CLIP**, le pionnier de l'alignement texte-image, et le **Vision Transformer (ViT)**, qui traite les images comme des séquences de mots. Enfin, nous maîtriserons **BLIP-2**, une architecture ingénieuse qui utilise un "Q-Former" pour faire le pont entre un encodeur visuel et un LLM classique, permettant de discuter littéralement avec une image.

---

## 🧠 Pré-requis importants
Pour cette immersion, vous devez avoir révisé :
1.  **L'Attention Spatiale** : Comprendre que la self-attention (Semaine 3) peut s'appliquer aux pixels comme aux mots.
2.  **L'Espace Vectoriel Commun** : L'idée que le vecteur du mot "Chien" et le vecteur d'une photo de chien doivent être proches mathématiquement.

**Ressources pour réviser les pré-requis (Vérifiées et fonctionnelles) :**
*   🎥 **Computerphile** : [Vision Transformer (ViT) - How it works](https://www.youtube.com/watch?v=TrdevFK_am4) (Excellente intuition sur le découpage en patches).
*   💻 **Hugging Face Documentation** : [Concepts de Computer Vision](https://huggingface.co/docs/transformers/tasks/image_classification) (Pour comprendre le prétraitement des images).

---

## 📚 Ressources utiles pour les concepts de la semaine (Vérifiées le 22/05/2024)

### 🌐 Articles et Blogs de référence
*   **Lilian Weng (OpenAI)** : [Vision-Language Pre-training](https://lilianweng.github.io/posts/2023-06-24-vlm/) – Le guide le plus complet et rigoureux sur les modèles multimodaux (Vérifié : Fonctionnel).
*   **OpenAI Index** : [CLIP: Connecting Text and Images](https://openai.com/index/clip/) – L'article original expliquant l'apprentissage contrastif (Vérifié : Fonctionnel).
*   **Google Research Blog** : [Transformers for Image Recognition at Scale](https://blog.research.google/2020/12/transformers-for-image-recognition-at.html) – L'introduction officielle du Vision Transformer (ViT) (Vérifié : Fonctionnel).
*   **Hugging Face Blog** : [BLIP-2: Scalable Pre-training for VLM](https://huggingface.co/blog/blip-2) – Analyse technique de l'architecture Q-Former et son intégration (Vérifié : Fonctionnel).
*   **Jay Alammar** : [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) – (Voir la section sur le "Text Encoder" pour comprendre comment CLIP aligne le sens) (Vérifié : Fonctionnel).

### 🛠️ Tutoriels et Outils
*   **Hugging Face Documentation** : [Using CLIP for Zero-Shot Image Classification](https://huggingface.co/docs/transformers/model_doc/clip) – Guide pratique pour classer des images sans entraînement (Vérifié : Fonctionnel).
*   **OpenCLIP GitHub** : [Models and Usage](https://github.com/mlfoundations/open_clip) – La bibliothèque de référence pour les modèles CLIP open-source (Vérifié : Fonctionnel).

---

🔑 **Mon conseil** : Dans les sections à venir, gardez en tête l'analogie du "Traducteur" : le Q-Former de BLIP-2 est comme un interprète assis entre un photographe (ViT) et un écrivain (LLM). Son rôle est de transformer des signaux visuels en concepts que l'écrivain peut comprendre. 

⚠️ **Attention :** un mauvais alignement conduit à des hallucinations visuelles !
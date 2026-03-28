---
title: "Semaine 11 : Fine-tuning supervisé de modèles génératifs"
weight: 1
bookCollapseSection : true
---

# 📝 Semaine 11 : Fine-tuning supervisé de modèles génératifs

Bonjour à toutes et à tous ! Préparez-vous bien, car nous entrons aujourd'hui dans la cour des grands de l'ingénierie des LLM. Si nous nous sommes contentés jusqu'à présent d'utiliser des modèles déjà pré-entraînés, une question cruciale se pose : comment faire lorsque vous devez développer une expertise dans un domaine qu'aucune IA n'a encore exploré ? 

> [!IMPORTANT]
🔑 **Je dois insister :** ne vous laissez pas intimider par la taille des modèles. 

Aujourd'hui, je vais vous apprendre à rééduquer un géant de 7 milliards de paramètres avec les ressources d'un simple ordinateur portable. Bienvenue dans l'ère du **Fine-tuning efficace** !

---
## 📖 Description générale
Cette semaine est consacrée à l'adaptation des modèles de langage génératifs via le **Supervised Fine-Tuning (SFT)**. Nous allons comparer le Fine-Tuning complet et le **PEFT** (*Parameter-Efficient Fine-Tuning*). 

Nous plongerons dans la méthode **LoRA**, qui permet de modifier seulement une infime fraction des poids d'un modèle, et nous découvrirons la **Quantification (QLoRA)** pour réduire drastiquement l'empreinte mémoire. Enfin, nous apprendrons à utiliser le framework **TRL** et son **SFTTrainer** pour transformer un modèle de base en un assistant d'instruction performant.

---
## 🧠 Pré-requis importants
Pour maîtriser le fine-tuning supervisé, vous devez avoir révisé :
1.  **Le Fine-tuning de BERT (Semaine 4)** : Pour comprendre la logique de base de la mise à jour des poids.
2.  **Paramètres de génération (Semaine 5)** : Car un modèle mal réglé après fine-tuning reste imprévisible.
3.  **Bases du Gradient Descent** : L'intuition mathématique de la correction d'erreur.

**Ressources pour réviser les pré-requis :**
*   💻 **Hugging Face Course** : [Fine-tuning a pretrained model](https://huggingface.co/learn/nlp-course/chapter3/1) .
*   📚 **PyTorch Tutorials** : [What is torch.nn?](https://pytorch.org/tutorials/beginner/nn_tutorial.html) – Pour l'intuition sur les couches et les paramètres.


---
## 📚 Ressources utiles pour les concepts de la semaine

### 🌐 Articles et Blogs de référence
*   **Hugging Face Blog** : [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/blog/peft) – Ce blog explique comment adapter de grands modèles avec des ressources limitées. Il présente notamment LoRA et les préfixes virtuels.
*   **ArXiv (Article LoRA)** : [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685) – L'article scientifique fondateur pour ceux qui veulent comprendre la décomposition matricielle mathématique.
*   **Hugging Face Blog** : [Making LLMs even more accessible (QLoRA)](https://huggingface.co/blog/4bit-transformers-bitsandbytes) – Un guide technique sur la quantification 4-bit. Il montre comment charger des modèles massifs sur un seul GPU.
*   **Lightning AI** : [LoRA from scratch](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch) – Une implémentation pas à pas pour comprendre ce qui se passe réellement dans les couches de poids.

### 🛠️ Documentations et Outils
*   **TRL (Transformer Reinforcement Learning)** : [SFTTrainer Documentation](https://huggingface.co/docs/trl/sft_trainer) – Le manuel d'utilisation de l'outil que nous utiliserons pour nos laboratoires.
*   **BitsAndBytes GitHub** : [Quantization Basics](https://github.com/bitsandbytes-foundation/bitsandbytes) – Pour comprendre les types de données NF4 et l'optimisation GPU.

### Vidéos recommandées
*   **Krish Naik**: [Finetuning LLM](https://www.youtube.com/playlist?list=PLZoTAELRMXVN9VbAx5I2VvloTtYmlApe3) - Une playliste couvrant tous les concepts que vous allez voir cette semaine.
*   **Ai Engineer** : [RFT, DPO, SFT: Fine-tuning avec OpenAI — Ilan Bigio, OpenAI](https://www.youtube.com/watch?v=JfaLQqfXqPA) - Atelier complet couvrant toutes les formes de fine-tuning et de prompt engineering (SFT, DPO, RFT), l'optimisation de prompts et l'architecture d'agents.

---
> [!NOTE]
🔑 **Note** : Dans cette semaine, nous allons parler de "rang" et de "matrices". 

> [!WARNING]
⚠️ **Attention :** ne vous perdez pas dans les équations. Retenez l'image : LoRA est comme un post-it intelligent que l'on colle sur un dictionnaire géant pour y ajouter de nouvelles définitions sans réécrire tout le livre !

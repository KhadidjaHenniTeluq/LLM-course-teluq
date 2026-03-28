---
title: "11.4 QLoRA en pratique "
weight: 5
---

## La fusion des savoirs : Mettre le moteur en marche
Bonjour à toutes et à tous ! Nous arrivons enfin au sommet de notre semaine. Nous avons le "gouvernail" (LoRA, section 11.2) et nous avons appris à "réduire la taille du navire" (Quantification, section 11.3). Maintenant, mes chers étudiants, il est temps de prendre la mer !
> [!IMPORTANT]
🔑 **Je dois insister :** la théorie est une boussole, mais la pratique est le navire. 

Aujourd'hui, nous allons assembler toutes ces briques pour réaliser un véritable entraînement. Nous allons transformer un modèle "Base" qui ne sait que compléter du texte en un "Assistant" capable de suivre vos ordres avec précision. Respirez, nous allons coder le futur, brique par brique, sur votre petit GPU T4 !

---
## Le pipeline QLoRA : Un workflow de précision
Mettre en œuvre QLoRA en production ne se limite pas à lancer une commande. C'est une chorégraphie technique. 
Le processus suit quatre étapes non-négociables :
1.  **Chargement 4-bit** : On charge le modèle géant (le Base Model) en utilisant la configuration NF4 que nous avons étudiée. Le géant est maintenant compressé et tient dans un coin de votre VRAM.
2.  **Préparation au K-bit** : On utilise une fonction spéciale pour préparer les couches du modèle à recevoir des adaptateurs alors qu'il est lui-même quantifié. 
3.  **Injection de LoRA** : On définit nos matrices de bas-rang (le "calque" sémantique) et on les attache aux couches d'attention.
4.  **Entraînement Supervisé (SFT)** : On lance la boucle d'apprentissage sur nos données d'instruction. 

> [!NOTE]
🔑 **Note** : Pendant toute l'étape 4, le modèle de base reste "gelé" dans sa prison 4-bit. 

> Seuls les petits adaptateurs LoRA reçoivent les mises à jour de poids. C'est ce qui rend l'opération si légère.

---
## La donnée : Le carburant de l'Instruction Tuning
Avant d'entraîner, il nous faut une structure. Un LLM ne comprend pas naturellement la notion de "dialogue". Pour lui, tout est une longue suite de caractères.

Regardons attentivement la **Figure 11-13 : Le template chat de TinyLlama** . Cette illustration est la clé de voûte de votre interface utilisateur. 

{{< bookfig src="275.png" week="11" >}}

*   **Analyse de la figure** : Le texte est encapsulé dans des balises spéciales. On voit `<|user|>` pour marquer le début de la question humaine et `<|assistant|>` pour marquer le début de la réponse de l'IA. 
*   **Le rôle du token EOS** : Notez bien la balise `</s>` à la fin de chaque tour. C'est le signal "Fin de séquence" (End Of Sequence). 

> [!WARNING]
> *   ⚠️ **Attention : erreur fréquente ici !** Si vous oubliez d'inclure ces balises durant votre fine-tuning, votre modèle sera incapable de s'arrêter de parler ou de savoir quand c'est à votre tour de taper un message. Il continuera à générer du texte jusqu'à épuisement de sa fenêtre de contexte.

---
## L'outil de choix : SFTTrainer et la bibliothèque TRL
Pour orchestrer cet entraînement, nous utilisons le framework **TRL** (*Transformer Reinforcement Learning*). Son composant phare est le **SFTTrainer**.

> [!IMPORTANT]
🔑 **Je dois insister :** pourquoi utiliser `SFTTrainer` plutôt qu'un `Trainer` classique de Hugging Face ? 

> Parce qu'il est optimisé pour les modèles de langage : il gère automatiquement le formatage des prompts, le compactage des séquences (packing) pour gagner de la vitesse, et l'intégration native avec PEFT/LoRA.

---
## Les Hyperparamètres critiques : Le réglage du pilote

> [!CAUTION]
⚠️ Ne jouez pas avec les paramètres au hasard ! 

En QLoRA, certains chiffres sont sacrés  :
1.  **Learning Rate (Taux d'apprentissage)** : On utilise souvent `2e-4`. C'est assez élevé car on n'entraîne que très peu de paramètres.
2.  **Gradient Accumulation Steps** : Puisque notre GPU T4 est petit, nous ne pouvons pas mettre beaucoup d'exemples en même temps (Batch Size). L'accumulation permet de simuler un gros lot de données en additionnant les résultats de plusieurs petits passages avant de mettre à jour les poids.
3.  **Optimiseur "paged_adamw_32bit"** : C'est le secret de QLoRA pour éviter les plantages mémoire en déchargeant temporairement des données sur la RAM système.

---
## Laboratoire de code : Fine-tuning QLoRA complet
Voici le code "industriel" pour transformer un modèle de base (TinyLlama) en un assistant poli. Ce code intègre la quantification 4-bit, la configuration LoRA et l'entraînement supervisé.

```python
# Installation des outils nécessaires
# !pip install -q transformers peft accelerate bitsandbytes trl datasets

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 1. CONFIGURATION QUANTISATION (NF4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 2. CHARGEMENT DU MODÈLE DE BASE
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Fix pour le padding

# 3. PRÉPARATION PEFT (LoRA)
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 4. CHARGEMENT DES DONNÉES (Instruction Data)
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:500]") # Petit échantillon

# 5. ARGUMENTS D'ENTRAÎNEMENT
training_args = TrainingArguments(
    output_dir="./results_qlora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, # batch effectif de 16
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_32bit" # Protection mémoire
)

# 6. LANCEMENT DU TRAINER
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="prompt", # Nom du champ dans votre dataset
    max_seq_length=512,
    args=training_args,
    peft_config=peft_config
)

print("🚀 Lancement du fine-tuning...")
trainer.train()
```

---
## Le grand final : Le "Merge" des poids
Une fois l'entraînement terminé, vous avez d'un côté votre "dictionnaire" original (8 Go) et de l'autre votre petit fichier d'adaptateurs LoRA (environ 50 Mo).
Pour utiliser le modèle de manière fluide en production :
1.  On recharge le modèle de base en haute précision (Float16).
2.  On charge les poids LoRA.
3.  On utilise la méthode `model.merge_and_unload()`.

> [!IMPORTANT]
📌 **Je dois insister sur cet acte final :** le merge fusionne mathématiquement les deux matrices. 

> Vous obtenez un nouveau modèle unique, prêt à l'emploi, qui ne nécessite plus de bibliothèque PEFT pour fonctionner. C'est l'étape qui transforme votre prototype en un produit fini.

---
## Éthique et Responsabilité : L'ombre de la spécialisation

> [!CAUTION]
⚠️ Mes chers étudiants, un modèle spécialisé est un modèle dont on a restreint l'horizon.

1.  **L'illusion de la compétence** : En fine-tunant un modèle pour qu'il réponde comme un assistant médical, vous le rendez très persuasif. Mais si ses données de départ étaient fausses, il devient un "menteur professionnel" extrêmement crédible. 
2.  **Coût environnemental** : Bien que QLoRA soit économe, multiplier les fine-tunings inutiles a un coût énergétique. 

>> [!TIP]
📜 **Règle d'ingénieur responsable** : Demandez-vous toujours si un bon "Few-shot prompt" (Semaine 8) ne suffirait pas avant de lancer un entraînement.

3.  **Protection de la vie privée** : 
>> [!CAUTION]
⚠️ **!** Durant le SFT, le modèle peut mémoriser par cœur des fragments de vos données d'entraînement. Si vous utilisez des données clients privées, assurez-vous qu'elles sont parfaitement anonymisées. Le modèle pourrait les "recracher" textuellement à un autre utilisateur.

> [!TIP]
✉️ **Mon message final pour cette semaine** : Vous avez accompli quelque chose d'extraordinaire. 

> Vous savez désormais prendre une intelligence brute et la sculpter pour en faire un outil métier. Vous n'êtes plus seulement des consommateurs d'IA, vous êtes des créateurs d'IA. La semaine prochaine, nous apprendrons à donner une âme et des valeurs à ces modèles grâce au **Tuning par préférences**. Mais pour l'instant, savourez votre réussite en laboratoire !

---
Nous avons terminé notre immense semaine sur le Fine-tuning ! Vous savez désormais configurer, quantifier, entraîner et fusionner un modèle de langage moderne. C'est un arsenal de compétences qui fait de vous des profils rares sur le marché. Place maintenant au laboratoire final de la semaine pour mettre tout cela en pratique !
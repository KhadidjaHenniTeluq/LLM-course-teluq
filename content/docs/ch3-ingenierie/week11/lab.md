---
title: "Laboratoire "
weight: 6
---

Bonjour à toutes et à tous ! Nous y sommes : le moment de vérité où le code rencontre la puissance de calcul. Dans ce laboratoire, vous allez orchestrer la métamorphose d'un modèle de langage. 
> [!IMPORTANT]
📌 **Je dois insister :** le fine-tuning est un art de la précision. Un mauvais réglage de la quantification ou un rang LoRA inadapté, et votre modèle pourrait perdre toute sa cohérence. 

Nous allons utiliser **TinyLlama**, un modèle robuste et léger, parfait pour notre GPU T4. Ne vous contentez pas de regarder les barres de progression d'entraînement : essayez de comprendre comment chaque ligne de configuration économise vos précieux gigaoctets de VRAM. Prêt·e·s à forger votre propre assistant ? C'est parti !

---
## 🔹 EXERCICE 1 : Configuration QLoRA pour Colab

**Objectif** : Configurer les briques technologiques (BitsAndBytes et LoRA) pour charger TinyLlama sur un GPU T4.

**Code (Testé sur Colab T4)** :
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- STRUCTURE DE BASE ---
# Tâche : Définissez une config 4-bit (NF4) et une config LoRA (r=8, alpha=16)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- CODE DE LA RÉPONSE ---
# 1. Configuration BitsAndBytes pour la quantification 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 # Précision de calcul pour le T4
)

# 2. Chargement du modèle de base
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# 3. Préparation au training K-bit et configuration LoRA
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"], # On cible les couches d'attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Vérification
model.print_trainable_parameters()
```

**Explications détaillées** :
*   **Résultats attendus** : Le message doit indiquer que moins de 1% des paramètres sont entraînables (environ 0.10% à 0.50%).
*   **Justification** : `nf4` est utilisé car il est plus précis que le `fp4` classique pour les poids neuronaux. On cible `q_proj` et `v_proj` car ce sont les matrices où l'attention sémantique est la plus forte.

</details>


---
## 🔹 EXERCICE 2 : Préparation du SFTTrainer

**Objectif** : Mettre en place la boucle d'entraînement supervisé (SFT) avec la bibliothèque TRL.

**Code (Testé sur Colab T4)** :
```python
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- STRUCTURE DE BASE ---
# Tâche : Initialisez le SFTTrainer pour une époque sur un échantillon de données.
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:100]")
```

<details>
<summary><b>Voir la réponse</b></summary>

```python

# --- CODE DE LA RÉPONSE ---
# 1. Arguments d'entraînement optimisés pour le T4
training_args = TrainingArguments(
    output_dir="./tinyllama_sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4, # Simule un batch de 8
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    fp16=True, # Indispensable sur T4
    optim="paged_adamw_32bit", # Protection contre le crash VRAM
    save_strategy="no"
)

# 2. Initialisation du SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="prompt", # Nom du champ texte dans ultrachat
    max_seq_length=512,
    args=training_args
)

# Simulation du lancement (ne pas exécuter si vous n'avez pas de GPU actif)
trainer.train()
print("Trainer configuré et prêt pour l'entraînement !")

```

**Explications détaillées** :
*   **Résultats attendus** : Un objet `trainer` prêt qui, lors de l'appel à `.train()`, verrait sa perte (loss) diminuer progressivement.
*   **Justification** : L'usage de `gradient_accumulation_steps` est vital -> il permet d'entraîner le modèle avec moins de VRAM en n'actualisant les poids que toutes les 4 petites étapes.

</details>


---
## 🔹 EXERCICE 3 : Inférence et Merge des poids

**Objectif** : Fusionner l'adaptateur LoRA avec le modèle de base pour créer un fichier modèle final autonome.

**Code (Testé sur Colab T4)** :
```python
# --- STRUCTURE DE BASE ---
# Tâche : Fusionnez l'adaptateur entraîné avec le modèle original.
# Note : Cette étape demande de charger le modèle en Float16 (pas en 4-bit) !
```

<details>
<summary><b>Voir la réponse</b></summary>

```python
# --- CODE DE LA RÉPONSE ---
from peft import PeftModel

# 1. Recharger le modèle de base en Float16 (Haute précision)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 2. Charger l'adaptateur LoRA par-dessus (on suppose qu'il a été sauvé dans './results')
model_to_merge = PeftModel.from_pretrained(base_model, "./tinyllama_sft")

# 3. Fusion mathématique (Merge)
merged_model = model_to_merge.merge_and_unload()

# 4. Sauvegarde du modèle final autonome
merged_model.save_pretrained("./final_assistant_model")

print("Modèle fusionné avec succès. Prêt pour le déploiement sans PEFT !")

```

**Explications détaillées** :
*   **Attentes** : Le modèle final doit pouvoir être chargé avec un simple `AutoModelForCausalLM.from_pretrained()` sans avoir besoin de la bibliothèque `peft`.
*   **Justification** : `merge_and_unload()` additionne physiquement les matrices LoRA aux matrices originales. 

> [!WARNING]
⚠️ **Avertissement** : On ne peut pas "merger" proprement un modèle 4-bit, c'est pourquoi on repasse en Float16 pour cette étape finale.

</details>


---
**Mots-clés de la semaine** : Fine-tuning, SFT, PEFT, LoRA, Rang (Rank), Alpha, QLoRA, NF4, Double Quantization, Paged Optimizer, Merge.

**En prévision de la semaine suivante** : Nous allons apprendre à donner une "conscience sociale" à notre modèle. Comment s'assurer qu'il reste poli et utile ? Bienvenue dans le monde de l'**Alignement par préférences (RLHF & DPO)**.
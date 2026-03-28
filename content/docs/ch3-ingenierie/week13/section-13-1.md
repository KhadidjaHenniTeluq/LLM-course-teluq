---
title: "13.1 Mécanismes d'inférence"
weight: 2
---

## Le mur de la latence : Pourquoi l'inférence est un défi
Imaginez que vous posiez une question à un expert et qu'il mette 45 secondes à prononcer chaque mot. Vous perdriez patience instantanément, n'est-ce pas ?

En production, l'ennemi numéro un est la **latence**. Un utilisateur s'attend à une réponse fluide, presque instantanée. Or, comme nous l'avons vu en Semaine 5, la génération de texte est un processus **autorégressif** : le modèle doit recalculer l'intégralité de ses probabilités pour chaque nouveau token produit. 

> [!IMPORTANT]
📌 **Je dois insister sur cette distinction :** l'entraînement est une phase de calcul intensif ponctuelle, mais l'inférence est un coût récurrent. 

Si votre modèle est lent, il coûte cher en électricité et fait fuir vos utilisateurs. Aujourd'hui, nous allons voir comment "tricher" intelligemment avec les mathématiques pour rendre l'inférence foudroyante. 

---
## Le sauveur de la vitesse : Le KV Cache
C'est sans doute l'optimisation la plus importante de l'architecture Transformer pour la génération. Regardons attentivement la **Figure 13-1 : KV cache pour l'accélération** .

{{< bookfig src="63.png" week="13" >}}

**Explication de la Figure 13-1** : Cette illustration nous montre le secret de la fluidité des chatbots. 
*   **Le problème sans cache** : À chaque fois que le modèle génère le token numéro 51, il doit relire les 50 tokens précédents pour calculer l'attention. C'est un gaspillage immense, car les représentations des 50 premiers mots n'ont pas changé ! C'est comme si, pour chaque nouveau mot d'une phrase, vous deviez relire tout le livre depuis le début.
*   **La solution avec cache** : On stocke les vecteurs **Keys (K)** et **Values (V)** de tous les tokens déjà traités dans la mémoire du GPU. 
*   **L'effet visuel dans la figure** : On voit que seule la "dernière colonne" de calcul est active. Le reste est simplement récupéré dans la mémoire "cache". 

> [!TIP]
⚡ **L'impact technique :** Le KV Cache transforme une complexité quadratique ($O(n^2)$) en une complexité linéaire ($O(n)$) par rapport à la longueur de la séquence générée.

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Le cache KV consomme beaucoup de VRAM. Pour un modèle Llama-3-70B, le cache peut occuper plusieurs gigaoctets à lui seul. C'est le compromis classique de l'informatique : **on sacrifie de la mémoire pour gagner du temps**.

---
## La quantification pour l'inférence (Post-Training Quantization)
Nous avons vu la quantification pour l'entraînement (QLoRA) en Semaine 11. Mais pour le déploiement, nous utilisons la **PTQ** (*Post-Training Quantization*). L'objectif n'est plus d'apprendre, mais de faire tenir le modèle final sur le plus petit serveur possible.

Regardez la **Figure 13-2 : Représentation des bits** . Elle nous rappelle que passer de 16 bits à 4 bits divise la taille du modèle par 4.

{{< bookfig src="271.png" week="13" >}}

⭐ **Les 3 formats rois du déploiement :**
1.  **GGUF (Llama.cpp)** : C'est le format universel pour l'inférence sur CPU et GPU. Il permet de "déborder" sur la RAM système si la VRAM est pleine. C'est l'outil idéal pour le déploiement local ou sur serveurs modestes.
2.  **AWQ (Activation-aware Weight Quantization)** : Un format plus récent et plus précis pour les GPU NVIDIA. Il protège les poids les plus importants pour l'intelligence du modèle, garantissant presque aucune perte de qualité en 4-bit.
3.  **EXL2** : Ultra-optimisé pour les cartes graphiques grand public, permettant des vitesses de génération dépassant les 100 tokens par seconde.

---
## Le débit vs la latence : Batching et Inférence continue

> [!IMPORTANT]
📌 En tant qu'ingénieurs, vous devez jongler avec deux chiffres contradictoires.
*   **La Latence** : Le temps pour qu'un utilisateur reçoive son premier mot.
*   **Le Débit (Throughput)** : Le nombre total de mots que votre serveur peut générer par seconde pour 100 utilisateurs simultanés.

Pour optimiser le débit, nous utilisons le **Continuous Batching**. Au lieu d'attendre qu'une phrase soit finie pour en commencer une autre, le serveur insère de nouvelles requêtes dès qu'un token est généré pour un autre utilisateur. C'est une gestion de flux tendu qui maximise l'usage du GPU à 100%.

---
## Hardware et optimisation : Choisir sa monture
Le choix du matériel dépend de votre budget et de votre besoin de vitesse.
*   **Inférence sur GPU (NVIDIA A100/H100)** : Vitesse maximale, supporte des centaines d'utilisateurs. Coût élevé.
*   **Inférence sur Consumer GPU (RTX 4090/3060)** : Très rapide pour un usage interne ou petite échelle.
*   **Inférence sur CPU (Serveurs classiques)** : Possible grâce à la quantification GGUF, mais beaucoup plus lent (souvent 2-5 tokens/sec). Idéal pour les tâches de fond non interactives (ex: résumé de mails la nuit).

---
## Laboratoire de code : Inférence optimisée avec GGUF
Nous allons utiliser `llama-cpp-python` pour charger un modèle TinyLlama quantifié. C'est l'implémentation la plus robuste pour faire tourner des modèles sur le matériel limité d'une instance Colab gratuite.

```python
# Testé sur Colab T4 16GB VRAM
# !pip install llama-cpp-python 

from llama_cpp import Llama
import time

# 1. CHARGEMENT OPTIMISÉ
# On télécharge un modèle au format GGUF (quantifié en 4-bit ou 8-bit)
model_path = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf" # Exemple de nom de fichier

# Initialisation du moteur d'inférence
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,        # -1 signifie : "Mettre TOUTES les couches sur le GPU T4"
    n_ctx=2048,             # Taille de la fenêtre de contexte
    n_batch=512,            # Taille des lots pour le traitement initial
    verbose=False
)

# 2. MESURE DE LA PERFORMANCE
prompt = "Q: What are the three pillars of a responsible LLM deployment? A:"

start_time = time.time()
output = llm(
    prompt, 
    max_tokens=100, 
    stop=["Q:", "\n"], 
    echo=False,
    temperature=0.2 # On reste factuel pour la prod
)
end_time = time.time()

# 3. CALCUL DU DÉBIT (TOKENS PAR SECONDE)
text_output = output["choices"][0]["text"]
tokens_generated = output["usage"]["completion_tokens"]
tps = tokens_generated / (end_time - start_time)

print(f"Réponse : {text_output}")
print(f"--- STATISTIQUES D'INFÉRENCE ---")
print(f"Vitesse : {tps:.2f} tokens/sec")
print(f"Temps total : {end_time - start_time:.2f} secondes")
```

> [!NOTE]
✍🏻 **Note** : Observez la vitesse. Sur une T4, un modèle 1B quantifié en 8-bit devrait dépasser les 50 tokens par seconde. C'est bien plus rapide que la lecture humaine ! C'est ce niveau de performance que vous devez viser en production.

---
## Éthique et Responsabilité : L'IA économe

> [!CAUTION]
⚖️ Mes chers étudiants, l'optimisation n'est pas qu'une question d'argent, c'est une question d'écologie numérique.

Chaque appel à un modèle non optimisé consomme une énergie inutile. 
1.  **L'impact carbone de l'inférence** : Si votre application devient virale et sert des millions de requêtes, une optimisation du KV cache ou de la quantification réduit l'empreinte carbone de votre entreprise de manière colossale.
2.  **Démocratisation** : Optimiser l'inférence permet de faire tourner l'IA sur des terminaux moins chers, rendant la technologie accessible aux populations qui n'ont pas accès aux derniers Mac M3 ou aux serveurs Cloud coûteux. **C'est un pilier de l'équité numérique.**

> [!TIP]
✉️ **Mon message** : Maîtriser l'inférence, c'est transformer une curiosité mathématique en un service public ou industriel. 

> Un modèle lent est un modèle mort. Un modèle optimisé est un modèle qui se fond dans la vie de l'utilisateur. Soyez les ingénieurs qui rendent l'IA invisible à force d'efficacité.

---
Vous maîtrisez maintenant la mécanique de la vitesse. Vous savez comment compresser l'intelligence et comment gérer la mémoire du GPU comme des professionnels. Dans la prochaine section ➡️, nous allons aborder le versant sombre mais nécessaire : la **Sécurité**. Nous allons apprendre à protéger votre IA contre les attaques et les biais, car une IA rapide qui se fait pirater est un danger public.
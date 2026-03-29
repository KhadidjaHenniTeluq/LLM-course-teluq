---
title: "2.3 Propriétés des tokeniseurs modernes"
weight: 4
---

## Au-delà du découpage : L'ingénierie des réglages
Bonjour à toutes et à tous ! Nous avons comparé les regards des géants dans la section précédente. Vous avez vu que GPT-4 ne "voit" pas la même chose que BERT ou StarCoder. Mais comment ces différences voient-elles le jour ? 

> [!IMPORTANT]
📌 **Je dois insister :** un tokeniseur ne naît pas "bon" ou "mauvais" par magie ; il est le produit de choix techniques délibérés que nous appelons les **propriétés du tokeniseur**.

Aujourd'hui, nous allons apprendre à régler les curseurs de cette machine de précision. Comprendre ces propriétés, c'est comprendre pourquoi votre modèle sera rapide, précis, ou au contraire, totalement perdu face à une langue étrangère. Respirez, nous allons entrer dans les coulisses de la configuration.

Nous détaillerons les trois piliers qui définissent l'identité d'un tokeniseur : la taille du vocabulaire, la gestion des tokens spéciaux et le traitement de la casse (capitalisation). 

Analysons ces paramètres avec la rigueur de l'ingénieur et l'intuition du linguiste.

---
## 1. La taille du vocabulaire : Le dilemme de la résolution
La propriété la plus visible d'un tokeniseur est la taille de son dictionnaire interne, souvent notée `vocab_size`. 

> [!NOTE]
**💡 Mon analogie** : Imaginez que la taille du vocabulaire soit la résolution d'un capteur photo. 

*   Un **petit vocabulaire** (ex: 30 000 tokens pour BERT) est comme une photo basse résolution. On voit les formes globales, mais pour les détails (les mots rares), on est obligé de "zoomer" et de découper le mot en plein de petits carrés (sous-mots). 
*   Un **grand vocabulaire** (ex: 100 000+ tokens pour GPT-4) est une photo haute définition. Le modèle peut identifier des concepts complexes d'un seul coup d'œil. 

> [!WARNING]
💥 **L'impact technique caché :** Pourquoi ne pas créer un dictionnaire de 10 millions de mots alors ? À cause du coût en mémoire vive (VRAM). 

> Rappelez-vous cette équation fondamentale : 
`Taille de la couche d'embedding = taille du vocabulaire × dimension du modèle`. 
Si vous avez un vocabulaire de 100 000 tokens et que chaque vecteur (embedding) fait 1024 nombres, vous consommez déjà plus de 100 millions de paramètres **avant même d'avoir commencé le premier bloc Transformer**. C'est un arbitrage constant entre la richesse de la compréhension et la légèreté du modèle.


---
## 2. Les Tokens Spéciaux : La signalisation du langage

> [!WARNING]
⚠️ **Attention : erreur fréquente ici !** Beaucoup d'étudiants pensent qu'un tokeniseur ne contient que des mots ou des morceaux de mots. 

C'est faux. Pour qu'un Transformer fonctionne, il a besoin de "panneaux de signalisation". Ce sont les **Special Tokens**.

Chaque modèle possède sa propre syntaxe de contrôle :
*   **[CLS] (Classification)** : Le token de tête utilisé par BERT pour résumer une phrase.
*   **[SEP] (Separator)** : Indique au modèle que l'on passe d'une phrase A à une phrase B.
*   **[PAD] (Padding)** : 
>> [!IMPORTANT]
🫀 **C'est une propriété vitale.** Les GPU adorent les grilles de taille fixe. Si vous avez une phrase de 5 mots et une de 10 mots, le tokeniseur ajoute des tokens `[PAD]` à la plus courte pour que les deux fassent la même longueur mathématique.
*   **[MASK]** : Utilisé pour le pré-entraînement (MLM - Masked Language Modeling) qu'on verra plus tard.
*   **BOS/EOS (Beginning/End Of Sentence)** : Pour les modèles génératifs comme GPT, ces tokens indiquent : "C'est ici que l'histoire commence" et "J'ai fini de parler, c'est à toi".

**Discussion sur les tokens de domaine (Galactica) :** Comme nous l'avons vu, certains modèles ajoutent des tokens comme `<work>` pour signaler un espace de brouillon. 

>> [!NOTE]
✍🏻 **Je dois insister :** en tant que développeurs, vous pouvez ajouter vos propres tokens spéciaux si vous créez une IA pour un métier précis (ex: un token `[GENE]` pour isoler des séquences biologiques).

---
## 3. La Normalisation et la Capitalisation
Avant même de découper le texte en tokens, le tokeniseur applique une phase de **Normalisation**. 

### Le cas de la Casse (Cased vs Uncased)
Nous avons vu avec BERT que le choix est binaire :
*   **Uncased** : On transforme tout en minuscules. C'est une forme de simplification statistique. On apprend au modèle que "Le", "le" et "LE" sont le même concept. Cela réduit le besoin de données, mais cela rend le modèle "aveugle" à l'emphase (le cri en majuscules) ou aux noms propres.
*   **Cased** : On préserve la casse. C'est plus difficile à apprendre, mais c'est indispensable pour le code (où `Variable` est différent de `variable`) ou pour la traduction de haute qualité.

### Le nettoyage de surface
Certains tokeniseurs appliquent des règles de "stripping" (suppression des espaces inutiles) ou de normalisation Unicode (ex: transformer le caractère "é" composé en un "e" plus un accent "´"). 

> [!IMPORTANT]
‼️ Si votre tokeniseur supprime les accents sans vous le dire, votre modèle sera incapable de distinguer "pêcher" (le fruit ou le poisson) et "pécher" (la faute morale). 

> **Vérifiez toujours la fonction de normalisation de votre outil !**

---
## 4. Le Domaine des données : L'ADN du Tokeniseur
C'est sans doute la propriété la plus sous-estimée. Un tokeniseur est **entraîné**. Il n'est pas programmé avec des règles de dictionnaire ; il a "appris" quelles étaient les suites de caractères les plus fréquentes sur un jeu de données précis.

*   **Le biais de domaine** : Si vous prenez un tokeniseur entraîné sur des tweets et que vous lui donnez un contrat notarié de 1850, il va "hacher" chaque mot complexe en 10 tokens minuscules. 
*   **Pourquoi est-ce grave ?** Plus vous avez de tokens pour une même idée, plus le modèle risque de perdre le fil (phénomène de fragmentation sémantique). 

> [!TIP]
🔑 **La règle d'or** : Le tokeniseur et le modèle doivent avoir été "élevés" sur les mêmes données.

**Discussion sur l'indentation (StarCoder2)** :
Revenons sur le sujet de l'indentation du code. Pour un texte naturel, l'espace n'est qu'un séparateur. Pour le code, l'espace est une **information de contrôle**. Un tokeniseur moderne pour le code doit posséder la propriété de préserver les blocs d'espaces. Si le tokeniseur fusionne trois espaces en un seul, le modèle ne "comprendra" jamais l'arborescence d'une fonction Python ou un autre langage sensible à l'indentation comme *Haskell* et les langages de configuration comme *YAML*.


---
## 5. Bonnes pratiques de sélection : Comment choisir ?
*Imaginez que vous deviez déployer un assistant de diagnostic médical multilingue. Quel compromis feriez-vous entre la taille du vocabulaire et la précision des terminologies latines ?*

Suivez ces trois critères non-négociables :

1.  **L'efficacité du dictionnaire** : Faites un test simple. Prenez 100 phrases de votre domaine. Comptez le nombre total de tokens générés par deux tokeniseurs différents. Celui qui produit le moins de tokens est généralement le plus "intelligent" pour votre usage, car il possède des concepts entiers dans son dictionnaire.
2.  **La gestion des caractères inconnus** : Si vous travaillez sur des données scientifiques ou multilingues, évitez les modèles qui sortent trop souvent des `[UNK]`. Privilégiez le **Byte-level BPE**.
3.  **La compatibilité avec le LLM** : 
>> [!CAUTION]
⚠️ **Point crucial !** On ne change jamais le tokeniseur d'un modèle déjà entraîné. Si vous utilisez Llama-3, vous DEVEZ utiliser le tokeniseur de Llama-3. Les poids du modèle ont été calibrés sur ces index précis.

---
## Laboratoire de réflexion : Le "Token Tax" (La taxe du token)

> [!CAUTION]
⚖️ Mes chers étudiants, les propriétés techniques ont des conséquences humaines. 

Parce que les tokeniseurs sont majoritairement configurés pour l'anglais, les langues "complexes" (comme l'arabe, le coréen ou le finnois) subissent ce que les chercheurs appellent la **Token Tax**. 
*   Pour une même idée, un utilisateur anglais paiera 10 tokens. 
*   Un utilisateur parlant une langue moins représentée paiera 30 tokens pour la même phrase.

> [!IMPORTANT]
❗ **Conséquence :** Cela signifie que les populations des pays du Sud ont des IA moins performantes (fenêtre de contexte plus petite) et plus chères (facturation au token). 

En tant qu'experts, votre responsabilité est de privilégier les modèles *"Truly Multilingual"* qui ont des vocabulaires larges et équitables.

---
## Synthèse
Nous avons exploré les entrailles des tokeniseurs. Nous avons compris que :
*   La **taille du dictionnaire** définit la résolution de la compréhension.
*   Les **tokens spéciaux** sont l'ossature du dialogue.
*   La **normalisation** peut être une alliée ou une ennemie de la précision.
*   Le **domaine d'entraînement** dicte la performance réelle sur le terrain.

> [!TIP]
🔑 **Mon message** : Le tokeniseur est la porte d'entrée de l'intelligence. 

> S'il est mal configuré, le cerveau de l'IA vivra dans le brouillard. Soyez méticuleux dans le choix de vos paramètres, car ils sont le premier filtre de la vérité de vos données.

---
Vous maîtrisez maintenant l'art du découpage et les secrets de la configuration. Mais une question demeure : une fois que nous avons nos numéros de tokens, comment la machine fait-elle pour savoir que le token `cat` (chat) est sémantiquement proche du token `kitten` (chaton) ? Il nous manque la "chair" des mots. Rendez-vous dans la section suivante ➡️ pour découvrir les **Embeddings**, la géométrie du sens.
# 🧠 Funktionsweise - Wie die Minecraft-KI arbeitet

## Überblick

Diese KI ist ein **Conditional Transformer GPT** (Generative Pre-trained Transformer), der speziell für die Generierung von Minecraft-Strukturen aus Text-Beschreibungen trainiert wurde.

**Ein Satz-Zusammenfassung:**  
Die KI liest deinen Text-Prompt, versteht die Beschreibung durch einen Text-Encoder, und generiert dann Block-für-Block eine passende 16×16×16 Minecraft-Struktur durch Cross-Attention zwischen Text und Struktur-Tokens.

---

## 🏗️ System-Architektur

### Komponenten-Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│  USER INPUT: "a big medieval house with oak wood"           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  TEXT TOKENIZER                                             │
│  • Zerlegt Text in Wörter: ["a","big","medieval","house"...]│
│  • Konvertiert zu Token-IDs: [5, 42, 78, 34, ...]          │
│  • Vokabular: 202 Wörter aus Training-Dateinamen           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  TEXT ENCODER (4-Layer Transformer)                         │
│  • Versteht semantische Bedeutung des Prompts              │
│  • Erzeugt Text-Embeddings (256-dimensional)               │
│  • Output: Kontext-Vektor für jeden Token                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  CONDITIONAL GPT MODEL (12-Layer Transformer)               │
│  • Generiert Blöcke autoregressiv (einer nach dem anderen) │
│  • Jeder Block berücksichtigt:                             │
│    1. Alle bisher generierten Blöcke (Self-Attention)      │
│    2. Die Text-Beschreibung (Cross-Attention)              │
│  • 512-dimensional, 8 Attention-Heads                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: 16×16×16 Minecraft-Struktur                        │
│  • 4096 Blöcke aus 121 möglichen Block-Typen               │
│  • Gespeichert als .schem Datei                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔤 1. Text-Verarbeitung

### 1.1 Text Tokenizer

**Aufgabe:** Konvertiert Text-Prompt in numerische Token.

**Beispiel:**
```
Input:  "a big medieval house with oak wood"
Tokens: [5, 42, 78, 34, 89, 15, 91]
        ↓  ↓   ↓   ↓    ↓   ↓   ↓
        a big med house with oak wood
```

**Vokabular-Aufbau:**
- Extrahiert aus 2415 Trainings-Dateinamen
- Beispiel-Dateiname: `a_big_abandoned_barn_built_out_of_spruce_wood_0001.schem`
- Underscores → Leerzeichen, Zahlen entfernt
- Finale Vokabular-Größe: **202 einzigartige Wörter**
- Top-Wörter: with (9777×), and (6175×), wood (2783×), interior (2371×)

**Spezial-Tokens:**
- `<PAD>` (ID: 0) - Auffüllen auf feste Länge
- `<UNK>` (ID: 1) - Unbekannte Wörter

**Max. Länge:** 128 Tokens (genug für komplexe Beschreibungen)

### 1.2 Text Encoder

**Architektur:**
- 4 Transformer-Schichten
- 256-dimensionale Embeddings
- 8 Attention-Heads
- Positionelle Encodings

**Funktion:**
Der Text-Encoder versteht die **semantische Bedeutung** der Wörter im Kontext:

```
"house with oak" 
    ↓
[Embedding-Vektor der versteht: 
 - "house" ist das Haupt-Objekt
 - "oak" ist ein Material
 - "with" beschreibt eine Eigenschaft]
```

**Output:** 
- Text-Embeddings: (batch_size, 128, 256)
- Text-Mask: Markiert welche Tokens echt sind (nicht Padding)

---

## 🧱 2. Block-Verarbeitung

### 2.1 Block Vocabulary

**121 verschiedene Minecraft-Blöcke + 5 Spezial-Tokens:**

**Kategorien:**
- **Natural (34)**: stone, dirt, grass_block, oak_log, spruce_log, sand, ...
- **Building (21)**: stone_bricks, oak_planks, glass, cobblestone, ...
- **Wool/Concrete (32)**: white_wool, red_concrete, blue_terracotta, ...
- **Functional (29)**: oak_door, torch, chest, crafting_table, ...
- **Special (5)**: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `air`

**Warum diese Blöcke?**
Extrahiert aus den 2415 Trainings-Strukturen - das sind die am häufigsten verwendeten Blöcke.

### 2.2 Struktur-Format

**16×16×16 = 4096 Blöcke**

**Speicherformat (Y-Z-X Order):**
```python
# Y = Höhe (0-15, unten nach oben)
# Z = Tiefe (0-15, vorne nach hinten)  
# X = Breite (0-15, links nach rechts)

position = y * 16 * 16 + z * 16 + x
```

**Warum Y-Z-X?**
- Preserviert vertikale Zusammenhänge (Schichten)
- Besser für Transformer (nahe Blöcke = nahe Positionen)
- Natürliche Bau-Reihenfolge (von unten nach oben)

---

## 🤖 3. Das Conditional GPT Modell

### 3.1 Model-Architektur

**Größe:**
- **Parameter:** ~50-60 Millionen
- **Schichten:** 12 Transformer-Blocks
- **Dimensionen:** 512 (d_model)
- **Attention-Heads:** 8
- **Feed-Forward:** 2048
- **Dropout:** 0.1

**Zwei Hauptkomponenten:**

#### A) Self-Attention (Was wurde schon gebaut?)
```
Jeder neue Block schaut auf alle bisherigen Blöcke:
"Aha, die letzten 50 Blöcke waren oak_planks...
 ich sollte das Muster fortsetzen"
```

#### B) Cross-Attention (Was sagt der Text-Prompt?)
```
Der Block schaut auf den Text-Prompt:
"Der Prompt sagt 'medieval house with oak wood'...
 ich sollte oak-bezogene Blöcke verwenden"
```

### 3.2 Conditional Transformer Block

**Jeder der 12 Blöcke macht:**

```
1. SELF-ATTENTION
   Input: Bisherige Blöcke
   Output: "Kontext der Struktur-Geschichte"
   
2. CROSS-ATTENTION  
   Input: Struktur-Kontext + Text-Embeddings
   Output: "Struktur-Kontext konditioniert auf Text"
   
3. FEED-FORWARD
   Input: Konditionierter Kontext
   Output: Finale Block-Representation
```

**In Formeln:**
```python
# Self-Attention: Verstehe die Struktur bisher
x = x + SelfAttention(x)
x = LayerNorm(x)

# Cross-Attention: Kombiniere mit Text-Information
x = x + CrossAttention(query=x, key=text, value=text)
x = LayerNorm(x)

# Feed-Forward: Finales Processing
x = x + FeedForward(x)
x = LayerNorm(x)
```

### 3.3 Autoregressive Generierung

**Block-für-Block Generierung:**

```
Schritt 1: <BOS> → "stone" (Basis-Block)
Schritt 2: <BOS> stone → "stone" (weiter Basis)
Schritt 3: <BOS> stone stone → "cobblestone"
...
Schritt 4096: <BOS> ... → "air" → <EOS>
```

**Warum autogressiv?**
- Jeder Block beeinflusst die nächsten
- Natürliche Struktur-Entwicklung
- Konsistente Muster

**Sampling-Strategien:**

1. **Greedy (nicht verwendet):**
   ```python
   next_block = argmax(probabilities)
   # → Immer gleiche Struktur
   ```

2. **Temperature Sampling:**
   ```python
   logits = logits / temperature
   # temperature=1.0: Original-Verteilung
   # temperature=0.7: Konservativer (weniger Chaos)
   # temperature=1.3: Kreativer (mehr Variation)
   ```

3. **Top-K Sampling:**
   ```python
   # Nur die top-K wahrscheinlichsten Blöcke behalten
   top_k = 50  # Wähle aus den 50 besten
   ```

4. **Top-P (Nucleus) Sampling:**
   ```python
   # Wähle Blöcke bis kumulative Prob. >= p
   top_p = 0.95  # Top 95% der Wahrscheinlichkeit
   ```

---

## 📊 4. Training-Prozess

### 4.1 Dataset

**Training-Daten:**
- **2415 .schem Dateien** mit Beschreibungen im Dateinamen
- **Split:** 80% Train (1932) / 10% Val (242) / 10% Test (241)
- **Augmentierung:** Rotationen (90°, 180°, 270°), Flips

**Data Loading:**
```python
for each .schem file:
    1. Parse filename → text description
    2. Load .schem → 16x16x16 blocks
    3. Tokenize text → text_ids
    4. Flatten blocks → block_sequence
    5. Create training pair:
       Input:  (text_ids, <BOS> + blocks[:-1])
       Target: (blocks + <EOS>)
```

### 4.2 Training Loop

**Jede Epoche:**
```
Für jeden Batch (16 Strukturen):
    1. Encode Text-Prompts
    2. Forward-Pass durch Modell
    3. Berechne Loss (Cross-Entropy)
    4. Backward-Pass (Gradienten)
    5. Update Weights (Adam Optimizer)
```

**Loss-Funktion:**
```python
# Cross-Entropy Loss
loss = CrossEntropyLoss(
    predictions,  # (batch, 4096, 121)
    targets,      # (batch, 4096)
    ignore_index=PAD_TOKEN
)
```

**Optimizer:**
- **AdamW** (Adam mit Weight Decay)
- Learning Rate: 0.0001
- Weight Decay: 0.01
- Gradient Clipping: 1.0 (verhindert Exploding Gradients)

**Learning Rate Schedule:**
- **Cosine Annealing**: LR sinkt sanft über Zeit
- Start: 0.0001 → Ende: ~0.00001

### 4.3 Validation

**Jede Epoche nach Training:**
```python
validation_loss = evaluate(model, val_dataset)

if validation_loss < best_loss:
    save_checkpoint("conditional_model_best.pt")
    best_loss = validation_loss
```

**Warum Validation?**
- Verhindert Overfitting
- Frühes Stoppen möglich
- Best Model ≠ Final Model

---

## 💾 5. Checkpoint-System

### 5.1 Was wird gespeichert?

```python
checkpoint = {
    'epoch': 42,
    'model_state_dict': {...},  # Alle Weights
    'optimizer_state_dict': {...},  # Optimizer-State
    'loss': 2.34,
}
```

### 5.2 Checkpoint-Typen

1. **conditional_model_best.pt**
   - Niedrigster Validierungs-Loss
   - Meist die beste Wahl für Generation

2. **conditional_model_final.pt**
   - Nach allen Epochen
   - Manchmal overfitted

3. **conditional_model_epoch_N.pt**
   - Alle 10 Epochen
   - Zum Experimentieren / Recovery

---

## 🎯 6. Generation-Prozess

### 6.1 Von Prompt zu Struktur

```
1. USER INPUT
   "a big medieval house with oak wood"
   
2. TEXT TOKENIZATION
   → [5, 42, 78, 34, 89, 15, 91, 0, 0, ...]  (128 tokens)
   
3. TEXT ENCODING
   → Text-Embeddings (128, 256)
   
4. AUTOREGRESSIVE GENERATION
   Start: [<BOS>]
   Loop 4096 mal:
       a) Predict nächsten Block
       b) Sample von Probability-Distribution
       c) Append zu Sequence
   End: [<BOS>, stone, stone, oak_planks, ..., air, <EOS>]
   
5. RESHAPE & SAVE
   → 16×16×16 Array
   → .schem Datei
```

### 6.2 Generation-Parameter

**Temperature (0.5 - 2.0):**
```
Low (0.7):  Konservativ, nah am Training
            ├─ stone (70%)
            ├─ cobblestone (20%)
            └─ oak_planks (10%)

Medium (1.0): Ausgewogen
              ├─ stone (40%)
              ├─ cobblestone (30%)
              └─ oak_planks (30%)

High (1.3):  Kreativ, experimentell
             ├─ stone (25%)
             ├─ cobblestone (25%)
             ├─ oak_planks (25%)
             └─ spruce_planks (25%)
```

**Top-K (0 = disabled, 20-100):**
```
top_k = 50
→ Wähle nur aus den 50 wahrscheinlichsten Blöcken
→ Verhindert total unrealistische Blöcke
```

**Top-P (0 = disabled, 0.9-0.99):**
```
top_p = 0.95
→ Wähle Blöcke bis 95% kumulative Wahrscheinlichkeit
→ Dynamische Auswahl (mehr bei unsicheren Stellen)
```

---

## 📈 7. Metriken & Bewertung

### 7.1 Training-Metriken

**Loss (Cross-Entropy):**
- Misst wie gut das Modell den nächsten Block vorhersagt
- Niedriger = besser
- Typisch: Start ~4.5 → Ende ~2.0-2.5

**Perplexity:**
```python
perplexity = exp(loss)
```
- Interpretierbar als "Verwirrung" des Modells
- Niedriger = besser

### 7.2 Qualitäts-Bewertung

**Schwierig automatisch zu bewerten!** Aber:

1. **Block-Diversity:**
   ```python
   unique_blocks / total_blocks
   # Zu niedrig: Langweilig (nur stone)
   # Zu hoch: Chaotisch (random Blöcke)
   ```

2. **Struktur-Kohärenz:**
   - Verbundene Wände?
   - Dach über Räumen?
   - Sinnvolle Material-Kombinationen?

3. **Prompt-Adherence:**
   - "with oak wood" → Viel oak_planks?
   - "stone base" → Untere Schichten = stone?

4. **Menschliche Bewertung:**
   - Sieht es aus wie ein Gebäude?
   - Ist es dem Prompt ähnlich?
   - Würdest du es in Minecraft bauen?

---

## 🔬 8. Technische Details

### 8.1 Attention-Mechanismus

**Self-Attention (Vereinfacht):**
```python
# Für jeden Block:
for i in range(current_position):
    # Wie wichtig ist Block i für die Vorhersage?
    attention_weight = softmax(
        query[current] @ key[i]
    )
    
    # Gewichtete Summe aller vorherigen Blöcke
    context += attention_weight * value[i]
```

**Cross-Attention:**
```python
# Für jeden Block:
for text_token in text_embeddings:
    # Wie relevant ist dieses Text-Token?
    attention_weight = softmax(
        query[current_block] @ key[text_token]
    )
    
    # Integriere Text-Information
    context += attention_weight * value[text_token]
```

### 8.2 Positional Encoding

**Warum?**
Transformer hat keine inhärente Positions-Information.

**Wie?**
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- pos = Position im Sequence
- i = Dimensions-Index
- Einzigartig für jede Position
- Ermöglicht Interpolation für ungesehene Längen

### 8.3 Layer Normalization

**Stabilisiert Training:**
```python
normalized = (x - mean(x)) / sqrt(var(x) + epsilon)
output = gamma * normalized + beta
```

- Verhindert Exploding/Vanishing Activations
- Ermöglicht tiefere Netzwerke
- Schnelleres Konvergieren

### 8.4 Dropout

**Regularisierung:**
```python
# Training: Zufällig 10% der Neuronen deaktivieren
if training:
    x = x * (random() > 0.1)
```

- Verhindert Overfitting
- Zwingt Netzwerk zu robusteren Features
- Nur während Training, nicht bei Generation

---

## ⚡ 9. Performance-Optimierungen

### 9.1 Batch Processing

**Parallel verarbeiten:**
```
Batch-Size 16:
→ 16 Strukturen gleichzeitig
→ GPU-Auslastung ↑
→ Training schneller
```

**Trade-off:**
- Größer: Schneller, aber mehr Memory
- Kleiner: Langsamer, aber weniger Memory

### 9.2 Mixed Precision Training

**FP16 statt FP32:**
```python
# Halbiert Memory-Verwendung
# ~2x schneller Training
# Minimal Loss in Genauigkeit
```

(Aktuell nicht implementiert, aber möglich)

### 9.3 Gradient Accumulation

**Für große Batch-Sizes:**
```python
# Simuliert Batch-Size 64 mit nur 16:
for i in range(4):
    loss = forward(batch[i])
    loss.backward()  # Akkumuliert Gradienten
    
optimizer.step()  # Update nach 4 Batches
```

### 9.4 Data Loading

**Parallel & Pinned Memory:**
```python
DataLoader(
    num_workers=4,      # 4 Threads laden Daten
    pin_memory=True,    # Schnellerer GPU-Transfer
    prefetch_factor=2   # Lädt im Voraus
)
```

---

## 🧪 10. Limitierungen & Future Work

### 10.1 Aktuelle Limitierungen

1. **Fixe Größe:** Nur 16×16×16
   - Größere Strukturen nicht möglich
   - Kleine Details schwierig

2. **Limitiertes Vokabular:** 202 Text-Wörter
   - Neue Konzepte nicht verstanden
   - Beschreibungen müssen ähnlich zu Training sein

3. **Keine Garantien:**
   - Kann "unmögliche" Strukturen erzeugen
   - Schwebende Blöcke möglich
   - Physik nicht berücksichtigt

4. **Lange Generierung:** ~30 Sekunden pro Struktur
   - Autoregressive = langsam
   - 4096 sequential steps

5. **Training-Daten Bias:**
   - Nur mittelalterliche/Fantasy-Stile
   - Keine modernen Gebäude
   - Material-Kombinationen aus Training

### 10.2 Mögliche Verbesserungen

**Größere Strukturen:**
```python
# 32×32×32 oder 64×64×64
# Benötigt: Mehr Memory, längeres Training
```

**Hierarchical Generation:**
```python
# 1. Grobe Struktur (8×8×8)
# 2. Verfeinern zu (16×16×16)
# 3. Details hinzufügen (32×32×32)
```

**Conditional VAE:**
```python
# Latent-Space Manipulation
# Interpolation zwischen Strukturen
# Stil-Transfer
```

**Multi-Scale Attention:**
```python
# Lange Distanz-Dependencies
# Globale Struktur-Kohärenz
# Schnellere Generierung
```

**Reinforcement Learning:**
```python
# Reward für:
# - Physikalisch stabil
# - Ästhetisch ansprechend
# - Prompt-Adherence
```

---

## 📚 11. Mathematische Formeln

### 11.1 Transformer Attention

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Q = Query Matrix (Was suche ich?)
K = Key Matrix (Was habe ich?)
V = Value Matrix (Was ist der Wert?)
d_k = Dimensions-Größe
```

**Multi-Head Attention:**
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)·W^O

head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

h = Anzahl Heads (8)
```

### 11.2 Loss-Funktion

**Cross-Entropy Loss:**
```
L = -∑_{t=1}^{4096} log P(b_t | b_{<t}, text)

b_t = Ground-Truth Block an Position t
P() = Model-Wahrscheinlichkeit
b_{<t} = Alle vorherigen Blöcke
```

### 11.3 Sampling

**Temperature Scaling:**
```
P_i = exp(logit_i / T) / ∑_j exp(logit_j / T)

T = Temperature
T→0: Deterministisch (argmax)
T=1: Original-Distribution
T→∞: Uniform random
```

---

## 🎯 Zusammenfassung

**Kernidee:**
Die KI kombiniert **Natural Language Understanding** (Text-Encoder) mit **Autoregressive Generation** (GPT) und **Cross-Attention**, um Minecraft-Strukturen zu erzeugen, die zu Text-Beschreibungen passen.

**Warum funktioniert es?**
1. **Große Trainings-Daten:** 2415 Strukturen mit Labels
2. **Starke Architektur:** Transformer mit 50M+ Parametern
3. **Conditioning:** Cross-Attention verbindet Text ↔ Struktur
4. **Autoregressive:** Jeder Block baut auf vorherigen auf

**Key Innovation:**
Die Kombination von Text-Conditioning mit strukturierter 3D-Generierung. Nicht einfach "Text → Bild", sondern "Text → 3D-Voxel-Struktur mit Spatial Coherence".

---

**Für weitere Fragen zur Nutzung:** Siehe `TUTORIAL.md`

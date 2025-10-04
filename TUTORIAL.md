# 🎮 Minecraft KI Tutorial - Von 0 bis zur ersten Struktur

## Was macht diese KI?

Diese KI generiert **Minecraft-Strukturen aus Text-Beschreibungen**!

**Beispiel:**
- Du schreibst: `"a big medieval house with oak wood and stone base"`
- Die KI erstellt: Eine 16×16×16 Minecraft-Struktur als `.schem` Datei

Die KI hat aus **2415 Beispiel-Strukturen** gelernt, wie verschiedene Gebäude-Typen aussehen.

---

## 📋 Schritt 1: Installation (5 Minuten)

### 1.1 Python-Pakete installieren

```powershell
# Navigiere zum Projektordner
cd "C:\Users\priva\Documents\MProductions\model a gpt"

# Aktiviere die virtuelle Umgebung (falls nicht schon aktiv)
.\venv\Scripts\Activate.ps1

# Installiere PyTorch mit GPU-Unterstützung
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Installiere andere Abhängigkeiten
pip install mcschematic pyyaml numpy tqdm tensorboard
```

### 1.2 Prüfe die Installation

```powershell
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

✅ Du solltest sehen: `PyTorch 2.x.x installed` und `CUDA available: True` (falls GPU vorhanden)

---

## 🚀 Schritt 2: Schnellstart (10 Minuten)

### Option A: Automatischer Start (Empfohlen)

```powershell
# Startet automatisch: Dataset-Split, Training, TensorBoard
python quickstart_conditional.py
```

Das Script macht alles automatisch:
- ✅ Teilt die 2415 Strukturen auf (Train/Val/Test)
- ✅ Startet das Training
- ✅ Speichert Checkpoints alle 10 Epochen
- ✅ Zeigt den Fortschritt an

### Option B: Test-Modus (Schnell ausprobieren)

```powershell
# Nur 5 Epochen zum Testen
python quickstart_conditional.py --test_mode
```

**⚠️ Wichtig:** Training dauert lange! 
- **Test-Modus**: ~30 Minuten (5 Epochen)
- **Vollständig**: ~20-30 Stunden (100 Epochen)

Du kannst das Training jederzeit unterbrechen (Strg+C) und später weitermachen!

---

## 📊 Schritt 3: Training überwachen

### 3.1 TensorBoard starten

Öffne ein **neues PowerShell-Fenster** und starte:

```powershell
cd "C:\Users\priva\Documents\MProductions\model a gpt"
.\venv\Scripts\Activate.ps1
tensorboard --logdir=runs
```

Dann öffne im Browser: **http://localhost:6006**

### 3.2 Was du siehst:

- **Train/Loss_Epoch**: Trainings-Loss (sollte sinken)
- **Val/Loss**: Validierungs-Loss (sollte auch sinken)
- **Train/Learning_Rate**: Lernrate über Zeit

**Gute Zeichen:**
- Loss sinkt kontinuierlich
- Validierungs-Loss ähnlich wie Trainings-Loss
- Nach ~20-30 Epochen stabilisiert sich der Loss

### 3.3 Wann ist das Modell fertig?

- **Minimum**: 20-30 Epochen (erste brauchbare Ergebnisse)
- **Gut**: 50 Epochen (solide Qualität)
- **Best**: 100 Epochen (höchste Qualität)

Du kannst jederzeit Strukturen generieren, auch während des Trainings!

---

## 🎨 Schritt 4: Strukturen generieren

### 4.1 Erste Struktur generieren

```powershell
python src/conditional_generate.py --prompt "a big medieval house with oak wood" --checkpoint checkpoints/conditional_model_best.pt
```

✅ Die Struktur wird gespeichert in: `generated/`

### 4.2 Mehrere Variationen generieren

```powershell
# Generiert 5 verschiedene Versionen der gleichen Beschreibung
python src/conditional_generate.py --prompt "a small church with stone walls" --checkpoint checkpoints/conditional_model_best.pt --num_samples 5
```

### 4.3 Kreativität steuern

```powershell
# Konservativ (näher am Training)
python src/conditional_generate.py --prompt "an abandoned barn" --temperature 0.7 --checkpoint checkpoints/conditional_model_best.pt

# Standard
python src/conditional_generate.py --prompt "an abandoned barn" --temperature 1.0 --checkpoint checkpoints/conditional_model_best.pt

# Kreativ (experimenteller)
python src/conditional_generate.py --prompt "an abandoned barn" --temperature 1.3 --checkpoint checkpoints/conditional_model_best.pt
```

**Temperature-Guide:**
- `0.7` = Sicherer, näher an den Trainings-Daten
- `1.0` = Ausgewogen (Standard)
- `1.3` = Kreativer, experimenteller

### 4.4 Batch-Generierung aus Datei

Erstelle eine Datei `meine_prompts.txt`:
```
a big medieval house with oak wood and stone base
a small church with pointed roof
an abandoned barn out of spruce wood
a fantasy tower with multiple floors
a stone castle with walls
```

Dann:
```powershell
python src/conditional_generate.py --prompts_file meine_prompts.txt --checkpoint checkpoints/conditional_model_best.pt --num_samples 2
```

Generiert 2 Variationen von jedem Prompt = 10 Strukturen total!

---

## 📚 Schritt 5: Gute Prompts schreiben

### 5.1 Dein Trainings-Vokabular

Die KI kennt diese Wörter am besten (aus den 2415 Trainings-Dateien):

**Häufigste Wörter:**
- `with` (9777×), `and` (6175×), `wood` (2783×), `interior` (2371×)
- `house` (2348×), `medieval` (2318×), `roof` (2348×), `walls` (1891×)
- `stone` (1843×), `oak` (1456×), `spruce` (1243×), `base` (1156×)

**Gebäude-Typen:**
- house, barn, church, tower, castle, ruin, lighthouse, windmill

**Materialien:**
- oak, spruce, birch, stone, sandstone, brick, deepslate
- wood, planks, logs, cobblestone

**Eigenschaften:**
- big, small, medium, abandoned, medieval, fantasy, arabic
- with interior, without interior, pointed roof, flat roof

### 5.2 Prompt-Rezepte

**Grundstruktur:**
```
[Größe] [Stil] [Typ] with/out of [Material] with [Features]
```

**Gute Beispiele:**
```
"a big medieval house with oak wood and stone base with interior"
"a small abandoned barn out of spruce wood with pointed roof"
"a medium fantasy tower with multiple floors and stone walls"
"an arabic desert house out of sandstone without interior"
"a big church with wooden roof and stone foundation"
```

**Tipps:**
- Verwende Wörter aus dem Trainings-Vokabular
- Kombiniere 3-5 Eigenschaften
- Sei spezifisch aber nicht zu kompliziert
- "with interior" → Inneneinrichtung
- "out of [Material]" → Hauptmaterial

### 5.3 Was funktioniert NICHT gut

❌ **Zu abstrakt:**
```
"something cool and epic"
"a magical building"
```

❌ **Zu spezifisch/neu:**
```
"a house with exactly 3 windows and a red door"
"a modern skyscraper with glass" (keine modernen Gebäude im Training)
```

❌ **Zu viele Details:**
```
"a big medieval house with oak wood and stone base with interior and a red pointed roof with chimney and balcony and garden and fence and pathway"
```

---

## 🔧 Schritt 6: Fortgeschrittene Nutzung

### 6.1 Training fortsetzen

```powershell
# Training wurde unterbrochen? Einfach weitermachen:
python src/conditional_train.py --data_path data/train --val_path data/val --resume checkpoints/conditional_model_epoch_50.pt
```

### 6.2 Eigene Konfiguration

Bearbeite `config.yaml` für andere Einstellungen:

```yaml
training:
  batch_size: 16        # Kleiner (8) wenn GPU-Memory-Error
  learning_rate: 0.0001 # Lernrate
  epochs: 100           # Anzahl Epochen
  
model:
  d_model: 512          # Modell-Größe (größer = mehr Parameter)
  n_layers: 12          # Transformer-Schichten
```

### 6.3 Verschiedene Checkpoints ausprobieren

```powershell
# Früheres Modell (evtl. unterschiedlicher Stil)
python src/conditional_generate.py --prompt "..." --checkpoint checkpoints/conditional_model_epoch_30.pt

# Bestes Modell (niedrigster Validierungs-Loss)
python src/conditional_generate.py --prompt "..." --checkpoint checkpoints/conditional_model_best.pt

# Finales Modell (nach allen Epochen)
python src/conditional_generate.py --prompt "..." --checkpoint checkpoints/conditional_model_final.pt
```

### 6.4 Manueller Workflow

Falls du jeden Schritt einzeln machen willst:

```powershell
# 1. Dataset aufteilen
python src/prepare_dataset.py split --source "fixed_all_files (1)/fixed_all_files" --output data

# 2. Text-Vokabular bauen (schon gemacht!)
# text_vocab.json existiert bereits

# 3. Training starten
python src/conditional_train.py --data_path data/train --val_path data/val --epochs 100

# 4. Strukturen generieren
python src/conditional_generate.py --prompt "..." --checkpoint checkpoints/conditional_model_best.pt
```

---

## 🎯 Schritt 7: Strukturen in Minecraft importieren

### 7.1 Mit WorldEdit (Empfohlen)

1. Kopiere die `.schem` Datei nach: `.minecraft/config/worldedit/schematics/`
2. In Minecraft (mit WorldEdit installiert):
   ```
   //schem load <dateiname>
   //paste
   ```

### 7.2 Mit MCEdit/Amulet Editor

1. Öffne deine Welt in MCEdit oder Amulet Editor
2. Import → Schematic → Wähle die `.schem` Datei
3. Platziere die Struktur

---

## 🐛 Problemlösung

### Problem: "CUDA out of memory"

```powershell
# Kleinere Batch-Size
python src/conditional_train.py --data_path data/train --batch_size 8
```

Oder in `config.yaml` ändern: `batch_size: 8` (oder 4)

### Problem: "ModuleNotFoundError: No module named 'torch'"

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Training ist sehr langsam

**Mit GPU:** ~30 Sekunden pro Epoche
**Ohne GPU (CPU):** ~5-10 Minuten pro Epoche

```powershell
# Prüfe ob GPU verwendet wird
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Falls `False` → GPU-Treiber/CUDA installieren oder CPU akzeptieren

### Problem: Strukturen sind "komisch" / schlechte Qualität

- ⏰ **Lösung 1**: Länger trainieren (mindestens 30-50 Epochen)
- 🎲 **Lösung 2**: Andere Temperature probieren (0.7 bis 1.3)
- 🔄 **Lösung 3**: Mehrere Variationen generieren (`--num_samples 5`)
- 📝 **Lösung 4**: Bessere Prompts aus dem Trainings-Vokabular

### Problem: Dataset-Split funktioniert nicht

```powershell
# Manuell Ordner erstellen
mkdir data\train, data\val, data\test

# Dann Dateien manuell kopieren:
# 80% nach data/train/
# 10% nach data/val/
# 10% nach data/test/
```

### Problem: Generation dauert zu lange

```powershell
# Nur 1 Sample generieren statt mehrere
python src/conditional_generate.py --prompt "..." --num_samples 1 --checkpoint checkpoints/conditional_model_best.pt
```

---

## 📈 Performance-Erwartungen

### Training (100 Epochen):

| Hardware | Zeit pro Epoche | Gesamt (100 Epochen) |
|----------|----------------|---------------------|
| GPU (RTX 3060+) | ~30 Sekunden | ~50 Minuten |
| GPU (GTX 1060) | ~1-2 Minuten | ~2-3 Stunden |
| CPU | ~5-10 Minuten | ~8-16 Stunden |

### Generierung (1 Struktur):

| Hardware | Zeit |
|----------|------|
| GPU | ~20-40 Sekunden |
| CPU | ~2-5 Minuten |

### Checkpoint-Größen:

- Model: ~200-250 MB pro Checkpoint
- Gesamt nach 100 Epochen: ~2-3 GB (10 Checkpoints + best + final)

---

## 🎓 Nächste Schritte

1. ✅ **Jetzt starten:**
   ```powershell
   python quickstart_conditional.py --test_mode
   ```

2. ⏱️ **Während Training läuft:**
   - TensorBoard öffnen: `tensorboard --logdir=runs`
   - Loss-Kurven anschauen
   - Nach ~10-20 Epochen erste Struktur generieren

3. 🎨 **Nach 30+ Epochen:**
   - Verschiedene Prompts ausprobieren
   - Mehrere Variationen generieren
   - Temperature experimentieren

4. 🏆 **Nach 100 Epochen:**
   - Beste Strukturen in Minecraft importieren
   - Eigene Prompt-Sammlung erstellen
   - Modell mit Freunden teilen

---

## 💡 Pro-Tipps

1. **Checkpoint-Management**: Das beste Modell ist nicht immer das finale! `conditional_model_best.pt` hat den niedrigsten Validierungs-Loss.

2. **Batch-Generierung**: Generiere viele auf einmal und wähle die besten aus:
   ```powershell
   python src/conditional_generate.py --prompt "a medieval house" --num_samples 10 --checkpoint checkpoints/conditional_model_best.pt
   ```

3. **Prompt-Variationen**: Kleine Änderungen → große Unterschiede:
   - "a house with oak wood" vs "a house out of oak wood"
   - "with interior" macht oft interessantere Strukturen

4. **Training über Nacht**: Starte Training am Abend, am nächsten Morgen ist es fertig!

5. **Experiment-Tagebuch**: Notiere welche Prompts gut funktionieren:
   ```
   Epoch 30, temp=1.0: "a medieval barn" → sehr gut!
   Epoch 50, temp=1.2: "a fantasy tower" → zu chaotisch
   Epoch 50, temp=0.9: "a fantasy tower" → perfekt!
   ```

---

## 🎮 Los geht's!

```powershell
python quickstart_conditional.py --test_mode
```

**Viel Erfolg mit deinem Minecraft-KI-Projekt!** 🏰✨

Bei Fragen: Siehe `FUNKTIONSWEISE.md` für technische Details.

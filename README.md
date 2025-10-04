# 🏰 Minecraft KI - Text zu Struktur Generator

**Generiere Minecraft-Strukturen aus Text-Beschreibungen mit KI!**

<div align="center">

```
"a big medieval house with oak wood and stone base"
                    ↓
         [KI verarbeitet...]
                    ↓
        16×16×16 .schem Datei
```

</div>

---

## 🎯 Was ist das?

Eine **Conditional GPT-basierte KI**, die Minecraft-Strukturen (16×16×16) aus Text-Prompts generiert.

**Trainiert auf:** 2415 Minecraft-Strukturen mit Beschreibungen  
**Technologie:** PyTorch, Transformer (GPT), Cross-Attention  
**Output:** `.schem` Dateien (für WorldEdit/MCEdit)

---

## ⚡ Quick Start

```powershell
# 1. Installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Training starten (automatisch optimiert!)
python src/conditional_train.py --data_path data/train --val_path data/val --batch_size 48 --epochs 100

# 3. Strukturen generieren
python src/conditional_generate.py --prompt "a big medieval house" --checkpoint checkpoints/conditional_model_best.pt
```

**Performance (H100 80GB mit automatischem Caching):**
- ⚡ **10-15 it/s** (automatisch optimiert)
- 🚀 **100 Epochen in ~15-20 Minuten**
- 💾 Alles wird beim Start in RAM gecacht (~500 MB)

**Fertig!** 🎉 Strukturen sind in `generated/`

---

## 📚 Dokumentation

### 🎓 [**TUTORIAL.md**](TUTORIAL.md) - Für Anfänger
- Schritt-für-Schritt Installation
- Training starten und überwachen
- Strukturen generieren
- Problemlösung
- **→ START HIER wenn du loslegen willst!**

### 🧠 [**FUNKTIONSWEISE.md**](FUNKTIONSWEISE.md) - Technische Details
- Wie die KI funktioniert
- Transformer-Architektur
- Text-Encoder & Cross-Attention
- Training-Prozess
- Mathematische Grundlagen
- **→ Für technisch Interessierte**

---

## 🎨 Beispiel-Prompts

```
"a big medieval house with oak wood and stone base with red pointed roof"
"an abandoned barn built out of spruce wood with interior"
"a small church with wooden roof and stone foundation"
"a fantasy tower with multiple floors and stone walls"
"an arabic desert house out of sandstone without interior"
```

**Tipp:** Die KI kennt 202 Wörter aus den Trainings-Daten. Verwende ähnliche Beschreibungen für beste Ergebnisse!

---

## 📁 Projekt-Struktur

```
model a gpt/
├── 📖 TUTORIAL.md                    # Anleitung für Benutzer
├── 📖 FUNKTIONSWEISE.md              # Technische Dokumentation
├── 🚀 quickstart_conditional.py      # Automatisches Setup & Training
├── ⚙️ config.yaml                    # Konfiguration
├── 📦 requirements.txt               # Python-Abhängigkeiten
├── 🗂️ text_vocab.json                # Text-Vokabular (202 Wörter)
│
├── src/                              # Source Code
│   ├── conditional_model.py          # KI-Modell (Conditional GPT)
│   ├── conditional_dataset.py        # Dataset-Loader
│   ├── conditional_train.py          # Training-Script
│   ├── conditional_generate.py       # Generierungs-Script
│   ├── text_tokenizer.py             # Text-Verarbeitung
│   ├── vocab.py                      # Block-Vokabular (121 Blöcke)
│   ├── schematic_parser.py           # .schem Datei-Parser
│   ├── prepare_dataset.py            # Dataset-Aufteilung
│   └── utils.py                      # Hilfsfunktionen
│
└── fixed_all_files (1)/              # Training-Daten
    └── fixed_all_files/              # 2415 .schem Dateien
```

---

## 🔧 Features

✅ **Text-zu-Struktur-Generierung** mit GPT-Architektur  
✅ **Cross-Attention** zwischen Text und Struktur  
✅ **121 Minecraft-Blöcke** aus 5 Kategorien  
✅ **Autoregressive Generierung** (Block-für-Block)  
✅ **Multiple Variationen** pro Prompt möglich  
✅ **Temperature/Top-k/Top-p Sampling** für Kreativitäts-Kontrolle  
✅ **TensorBoard Integration** für Training-Monitoring  
✅ **Checkpoint-System** (Best/Final/Epochen)  
✅ **GPU & CPU Support** (GPU empfohlen)  

---

## 📊 Technische Specs

| Komponente | Details |
|------------|---------|
| **Architektur** | Conditional Transformer GPT |
| **Parameter** | ~50-60 Millionen |
| **Text-Encoder** | 4-Layer Transformer, 256-dim |
| **Main Model** | 12-Layer, 512-dim, 8-heads |
| **Text-Vokabular** | 202 Wörter |
| **Block-Vokabular** | 121 Minecraft-Blöcke |
| **Struktur-Größe** | 16×16×16 (4096 Blöcke) |
| **Training-Daten** | 2415 annotierte .schem Dateien |
| **Framework** | PyTorch 2.0+ |

---

## ⚡ Performance

| Aufgabe | GPU (RTX 3060) | CPU |
|---------|----------------|-----|
| **Training** (1 Epoche) | ~30 Sekunden | ~5-10 Minuten |
| **Generierung** (1 Struktur) | ~20-40 Sekunden | ~2-5 Minuten |
| **Gesamt-Training** (100 Epochen) | ~50 Minuten | ~8-16 Stunden |

---

## 🎓 Erste Schritte

### 1️⃣ Für Einsteiger
👉 **Lies [TUTORIAL.md](TUTORIAL.md)**
- Installation
- Training starten
- Erste Struktur generieren

### 2️⃣ Für Entwickler
👉 **Lies [FUNKTIONSWEISE.md](FUNKTIONSWEISE.md)**
- Architektur verstehen
- Code-Struktur
- Eigene Anpassungen

### 3️⃣ Für Schnellstarter
👉 **Führe aus:**
```powershell
python quickstart_conditional.py --test_mode
```
(5 Epochen Test-Training)

---

## 🐛 Probleme?

**"CUDA out of memory"**
```powershell
python src/conditional_train.py --batch_size 8
```

**"ModuleNotFoundError: torch"**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Training zu langsam?**
- GPU verwenden (10-100x schneller)
- Kleinere Batch-Size
- Weniger Epochen im Test-Modus

➡️ **Mehr in [TUTORIAL.md](TUTORIAL.md) → Problemlösung**

---

## 📦 Requirements

```
Python 3.8+
torch >= 2.0.0
mcschematic >= 1.0.0
numpy
pyyaml
tqdm
tensorboard
```

Installation: `pip install -r requirements.txt`

---

## 🎮 Strukturen in Minecraft nutzen

### WorldEdit (Empfohlen)
1. Kopiere `.schem` nach `.minecraft/config/worldedit/schematics/`
2. In Minecraft: `//schem load <name>` → `//paste`

### MCEdit / Amulet Editor
1. Öffne Welt im Editor
2. Import → Schematic
3. Platzieren

---

## 💡 Tipps

1. **Trainiere länger:** 50-100 Epochen für beste Qualität
2. **Nutze Trainings-Vokabular:** Wörter wie "medieval", "oak", "interior" funktionieren gut
3. **Experimentiere mit Temperature:** 0.7 (sicher) bis 1.3 (kreativ)
4. **Generiere mehrere Variationen:** `--num_samples 5`
5. **Überwache mit TensorBoard:** `tensorboard --logdir=runs`

---

## 🚀 Nächste Schritte

```powershell
# Starte jetzt!
python quickstart_conditional.py --test_mode

# Öffne TensorBoard
tensorboard --logdir=runs

# Generiere nach Training
python src/conditional_generate.py \
    --prompt "a medieval house with oak wood" \
    --checkpoint checkpoints/conditional_model_best.pt \
    --num_samples 3
```

---

## 📄 Lizenz

Dieses Projekt ist für Bildungs- und persönliche Zwecke gedacht.

---

## 🙏 Credits

- **mcschematic** - Minecraft Schematic Library
- **PyTorch** - Deep Learning Framework
- **Transformer Architecture** - Vaswani et al. "Attention is All You Need"

---

<div align="center">

**Viel Erfolg mit deinem Minecraft-KI-Projekt!** 🏰✨

[Tutorial](TUTORIAL.md) • [Funktionsweise](FUNKTIONSWEISE.md) • [Config](config.yaml)

</div>

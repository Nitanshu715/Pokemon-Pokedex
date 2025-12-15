<div align="center">

# ⚡🟡 **POKÉMON POKÉDEX AI** 🔵⚡
### *Gotta Classify ’Em All!*

<img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/25.png" width="120"/>

🚀 **AI-powered Pokémon Image Recognition System**  
Built using **Deep Learning (CNNs)**, **TensorFlow**, and a **Pokémon-themed interactive UI**.

</div>

---

## 🌟 Project Overview

**Pokémon Pokédex AI** is a deep learning–based image classification system that identifies Pokémon from images.
Inspired by the classic Pokédex, this project combines **computer vision**, **convolutional neural networks**, and a **modern Pokémon-style interface**.

Upload an image → AI analyzes visual features → Pokémon identified with confidence ✨

---

## 🧠 How It Works (Deep Learning Explained)

### 🔍 1. Image Input
- User uploads a Pokémon image (JPG / PNG)
- Image resized to **224×224**

### ⚙️ 2. Preprocessing
- Normalization using **MobileNetV2 preprocess_input**
- Ensures consistent numerical representation

### 🧩 3. Convolutional Neural Network (CNN)
CNN layers automatically learn:
- Edges & contours
- Shapes & patterns
- Textures & color distributions

### 🧠 4. Feature Extraction
- Pooling layers reduce dimensionality
- Important visual features retained

### 🎯 5. Classification
- Fully connected layers classify **150 Pokémon**
- **Softmax** outputs probabilities
- Highest probability = prediction

### 📊 6. Confidence Threshold
- High confidence → Full Pokédex info
- Low confidence → Warning + suggestions

---

## 🧪 Tech Stack

| Layer | Technology |
|-----|-----------|
| 🧠 Model | TensorFlow / Keras |
| 📐 Architecture | CNN (MobileNet-based) |
| 🖼️ Image Processing | PIL, NumPy |
| 🌐 Frontend | Streamlit |
| 🎨 UI Theme | Pokémon-inspired design |
| 📦 Dataset | Kaggle Pokémon Classification Dataset |

---

## 📁 Project Structure

```text
pokemon-pokedex-ai/
│
├── app.py
├── train.py
├── pokedex.json
├── requirements.txt
├── README.md
│
├── model/
│   └── pokemon_model.keras
│
├── screenshots/
│   ├── ss1.png
│   ├── ss2.png
│   ├── ss3.png
│   ├── ss4.png
│   ├── ss5.png
│   └── ss6.png
```

---

## 📸 Screenshots

> 📂 Screenshots are available in the `/screenshots` folder

![UI Preview](screenshots/ss1.png)

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open browser at:
```
http://localhost:8501
```

---

## ⚠️ Dataset & Legal Disclaimer

### 📊 Dataset Used
**Pokémon Classification Dataset**  
🔗 https://www.kaggle.com/datasets/lantian773030/pokemonclassification

### ✅ Usage Compliance
- Dataset **NOT redistributed**
- Used strictly for **educational & non-commercial purposes**
- Proper credit given to dataset creator

### 🧾 Pokémon IP Notice
Pokémon names, images, and assets are © **Nintendo / Game Freak / The Pokémon Company**  
This project is **fan-made**, educational, and non-commercial.

---

## 💡 Why This Project Matters

✔ Real-world Deep Learning  
✔ End-to-end ML pipeline  
✔ Model deployment experience  
✔ UI + AI integration  
✔ Internship / Resume ready  

---

## 🧠 Future Enhancements

- Live camera Pokémon detection 📸
- Pokémon evolution prediction
- Sound-based Pokémon recognition 🔊
- Cloud deployment (AWS / GCP)
- Mobile-friendly Pokédex

---

<div align="center">

### ⚡ *Built with curiosity, caffeine, and a love for Pokémon* ⚡  
**If Ash had ML, this would be it.** 😤🔥

</div>


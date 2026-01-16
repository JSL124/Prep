

```markdown
# 🎮 THE DROP — Sealed Bid Elevator Game

**THE DROP** is a 1v1 sealed-bid strategy game where players compete to move passengers downward by outbidding their opponent under strict budget constraints.

> **Outbid. Outsmart. Get Down.**

---

## 🧠 Core Concept

- Each round, the player and the AI secretly place a bid.
- Higher bid wins the round and moves a passenger.
- Budgets are limited — overspending leads to defeat.
- The game ends early if a winner becomes mathematically guaranteed.

This game focuses on **prediction, bluffing, and resource management**.

---

## 🤖 AI System (LLM-powered)

The AI runs on a Node.js + OpenAI server and:

- Predicts the player’s bid range
- Adjusts aggression by stage
- Preserves budget when victory is secured
- Occasionally bluffs
- Explains its reasoning via an **AI Thought Panel**

### AI Stages
- **Stage 1:** Conservative AI  
- **Stage 2:** Adaptive AI  
- **Stage 3:** Aggressive AI  

---

## 🕹️ Features

- Sealed-bid bidding system
- LLM-based AI decisions
- Budget & score tracking
- Real-time timer
- Toggleable AI Thought panel
- Cyberpunk-style visuals (Phaser 3)
- Procedural sound effects

---

## 🧱 Tech Stack

**Frontend**
- HTML5 / JavaScript
- Phaser 3

**Backend**
- Node.js
- Express
- OpenAI API

---

## 📁 Project Structure

```

.
├── index_v5.html
├── title.png
├── README.md
├── .gitignore

└── server/
├── server.js
├── package.json
├── package-lock.json
└── .env.example

````

---

## ⚙️ Setup & Run

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
````

### 2. AI Server

```bash
cd server
npm install
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

Run server:

```bash
npm start
```

Server runs at:

```
http://localhost:3001
```

---

### 3. Run Game

Open `index_v5.html` in a browser
(or use VS Code Live Server).

---

## 🔐 Security

* `.env` is never committed
* `.env.example` is provided
* API keys are fully ignored by Git

---

## 🚀 Future Ideas

* Online multiplayer
* Ranked matchmaking
* More AI personalities
* Match analytics
* Cloud deployment

---

## 📜 Disclaimer

This project is a **simulation and design experiment**.
It does not control real-world elevators.

---

## 👤 Author

**Jinseo Lee**
Interactive AI Game & System Design Experiment

````

---


```bash
git add README.md
git commit -m "docs: add README"
git push
````



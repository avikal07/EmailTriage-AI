# 📬 Email Triage Environment

An **OpenEnv-compatible** reinforcement learning environment where an AI agent learns to triage incoming emails — classifying them, assigning priority, and generating professional replies.

---

## 🏗 Architecture

```
email-triage-env/
├── backend/
│   ├── app.py          # FastAPI application + API endpoints
│   ├── env.py          # OpenEnv environment logic
│   ├── models.py       # Pydantic models (Observation, Action, Reward)
│   ├── tasks.py        # Task definitions (Easy / Medium / Hard)
│   ├── grader.py       # Deterministic scoring logic
│   └── data/
│       └── emails.json # 18-email dataset with ground truth
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main UI
│   │   ├── api.js               # Axios client
│   │   └── components/
│   │       ├── ActionPanel.jsx  # Classification / priority / reply form
│   │       ├── EmailCard.jsx    # Email display card
│   │       ├── HistoryLog.jsx   # Step history panel
│   │       └── ScoreBar.jsx     # Reward visualizer
│   ├── index.html
│   ├── vite.config.js
│   └── tailwind.config.js
├── inference.py        # AI agent runner (OpenAI-compatible)
├── openenv.yaml        # Environment specification
├── requirements.txt    # Python dependencies
├── Dockerfile
├── .env.example
└── README.md
```

---

## 🎯 Tasks

| Task | Difficulty | Description | Max Score/Email |
|------|------------|-------------|-----------------|
| 1 | Easy | Classify emails (spam / important / normal) | 0.40 |
| 2 | Medium | Classify + assign priority (low / medium / high) | 0.70 |
| 3 | Hard | Classify + priority + generate professional reply | 1.00 |

---

## 📊 Reward Function

| Event | Reward |
|-------|--------|
| Classification correct | +0.40 |
| Classification wrong | −0.20 |
| Spam/important confusion | −0.30 |
| Priority correct | +0.30 |
| Priority off by one level | +0.12 |
| Priority off by two levels | −0.10 |
| Reply similarity ≥ 70% | +0.30 |
| Reply similarity 40–70% | +0.18 |
| Reply similarity 20–40% | +0.09 |
| Reply similarity < 20% | +0.03 |
| Reply generated for spam | −0.15 |
| Missing reply for important | −0.10 |

---

## 🚀 Running Locally

### 1. Backend

```bash
cd email-triage-env
pip install -r requirements.txt
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Backend is now available at `http://localhost:8000`

### 2. Frontend

```bash
cd email-triage-env/frontend
npm install
npm run dev
```

Frontend is now available at `http://localhost:5173`

### 3. Inference Script

```bash
cd email-triage-env
cp .env.example .env
# Edit .env with your API key
python inference.py
```

---

## 🐳 Docker

```bash
cd email-triage-env
docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env
```

Backend will be accessible at `http://localhost:8000`

---

## 🌐 API Usage

### Reset Environment

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'
```

**Response:**
```json
{
  "email": {
    "id": "email_001",
    "subject": "Q3 Budget Review - Action Required",
    "body": "Hi team, the Q3 budget review...",
    "sender": "cfo@company.com"
  },
  "task_id": 1,
  "task_description": "Classify each email as 'spam', 'important', or 'normal'...",
  "step_number": 0,
  "total_steps": 18,
  "current_score": 0.0
}
```

---

### Submit Action (Step)

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "email_id": "email_001",
    "classification": "important",
    "priority": "high",
    "reply": "Thank you for the reminder. I will attend the Q3 budget review meeting on Friday."
  }'
```

**Response:**
```json
{
  "observation": { ... },
  "reward": 0.82,
  "done": false,
  "info": {
    "step": 1,
    "grade_result": {
      "reward": 0.82,
      "breakdown": {
        "classification": { "predicted": "important", "expected": "important", "score": 0.4, "correct": true },
        "priority": { "predicted": "high", "expected": "high", "score": 0.3, "correct": true },
        "reply": { "score": 0.12, "note": "Similarity score: 0.421" }
      }
    },
    "total_reward": 0.82,
    "emails_remaining": 17
  }
}
```

---

### Get State

```bash
curl http://localhost:8000/state
```

**Response:**
```json
{
  "task_id": 3,
  "task_description": "Perform complete email triage...",
  "step_number": 1,
  "total_steps": 18,
  "current_score": 0.82,
  "total_reward": 0.82,
  "done": false,
  "emails_processed": 1,
  "current_email": { ... },
  "history": [ ... ]
}
```

---

## 🤖 Inference Agent

The `inference.py` script runs an LLM-powered agent through all 3 tasks.

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=http://localhost:8000
python inference.py

# HuggingFace / custom endpoint
export HF_TOKEN=hf-...
export MODEL_NAME=meta-llama/Llama-3-8B-Instruct
python inference.py
```

---

## 📈 Baseline Results

| Agent | Task 1 Avg | Task 2 Avg | Task 3 Avg |
|-------|-----------|-----------|-----------|
| Random | 0.06 | 0.02 | −0.05 |
| Rule-based | 0.31 | 0.22 | 0.18 |
| GPT-4o-mini | 0.38 | 0.61 | 0.74 |

---

## 📋 OpenEnv Spec

See `openenv.yaml` for the full environment specification including observation schema, action schema, task definitions, and reward tables.

---

## 📧 Dataset

18 emails covering:
- **5 spam** emails (phishing, scams, MLM, fake prizes)
- **8 important** emails (legal notices, security alerts, budget reviews, contract renewals)
- **5 normal** emails (newsletters, order confirmations, team updates)

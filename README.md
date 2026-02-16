# CoAct.AI

**Version:** 1.0
**Focus:** AI-Driven Behavioral Simulation & Coaching Analytics
**Objective:** Transforming high-stakes human interactions into measurable data.

---

## 1. System Architecture Overview

COACT.AI operates as a **Tri-Layer System** that manages the transition from live dialogue to structured coaching intelligence.

### A. The Interaction Layer (Roleplay)
- **Persona Engine:** Utilizes LLMs (GPT-4.1-Mini) with strictly defined personas to mhttps://coact-ai.comhavior (hesitation, defensiveness, or excitement).
- **Context Manager:** Injects scenario-specific data (e.g., store sales targets, customer budget) into the prompt window.

### B. The Intelligence Layer (The Orchestrator)
- **Turn-by-Turn Analysis:** Monitors the conversation for "Trigger Phrases" or "Communication Errors" in real-time.
- **Sentiment Tracker:** Measures the emotional distance between the AI and the Human.

### C. The Reporting Layer (Data Output)
- **Dynamic Report Engine:** Converts the transcript into structured feedback reports.
- **The Blueprint Generator:** Uses "Chain-of-Thought" reasoning to suggest specific alternative scripts for the human.

---

## 2. Standard Operating Procedures (SOP) by Scenario

|**Phase**|**Scenario 1 (Staff)**|**Scenario 2 (Sales)**|**Scenario 3 (Coach)**|
|---|---|---|---|
|**Initialization**|Focus on _Performance Gap_|Focus on _Price Resistance_|Focus on _Self-Reflection_|
|**AI Stance**|Defensive $\rightarrow$ Open|Sceptical $\rightarrow$ Sold|Curious $\rightarrow$ Guiding|
|**Constraint**|Must show "Ego"|Must compare competitors|**No numerical scoring**|
|**Success Metric**|Improvement Commitment|Value Articulation|Depth of Insight|

---

## 3. Data Flow & Logic

The platform follows a linear data pipeline to ensure accuracy:

1.  **Input:** Human user selects/creates a scenario.
2.  **Processing:**
    *   **Orchestrator (Flask):** Manages the conversation flow and state.
    *   **LLM (Azure OpenAI):** Generates responses and evaluates performance.
3.  **Synthesis:** The **Report Engine** (Python) populates the metrics and generates PDF reports.
4.  **Persistence:** Data is stored in **Supabase** (PostgreSQL) for long-term progress tracking and user history.

---

## 4. Technical Performance Standards

To maintain the "Realism" of COACT.AI, the following standards are targeted:

-   **Latency:** AI response time optimized for natural conversation flow.
-   **Persona Integrity:** The AI maintains character and does not break the "fourth wall".
-   **Hallucination Guardrails:** Strict system prompts prevent the AI from inventing facts outside the scenario context.

---

## 5. Security & Ethics

-   **Anonymization:** User data is handled securely via Supabase Auth.
-   **Bias Mitigation:** The AI is tuned to avoid discriminatory biases in retail/managerial settings.

---

## 6. High-Level Architecture Diagram

### 1. Client Layer (Frontend)
-   **Interface:** React (Vite) application providing a chat-based UI.
-   **State Management:** Tracks the session state and visual feedback.
-   **Communication:** REST APIs for communication with the Flask backend.

### 2. Service Layer (Backend)
-   **API Server:** Flask server (Python) handling authentication, scenario routing, and report generation.
-   **AI Integration:** Connects to Azure OpenAI Service for LLM capabilities.
-   **Report Generation:** Uses `fpdf` and `reportlab` to generate detailed PDF reports.

### 3. Data Layer (Persistence)
-   **Supabase:**
    -   **Auth:** Handles user sign-up, login, and session management.
    -   **Database:** PostgreSQL database storing user profiles and session history (via Supabase Client).

---

## 7. Technology Stack Summary

| **Category**        | **Tools**                     |
| ------------------- | ----------------------------- |
| **LLMs**            | Azure OpenAI (GPT-4o Mini)    |
| **Backend**         | Flask (Python)                |
| **Frontend**        | React (Vite), TailwindCSS     |
| **Database & Auth** | Supabase                      |
| **Deployment**      | Docker, Azure VM              |
| **Text-to-Speech**  | Azure Speech Services / OpenAI|

---

## 8. Getting Started

### Prerequisites
-   **Docker & Docker Compose**: For containerized deployment.
-   **Node.js**: If running frontend locally.
-   **Python 3.11+**: If running backend locally.
-   **Supabase Account**: For database and authentication.
-   **Azure OpenAI Keys**: For LLM access.

### Configuration
Create a `.env` file in the root directory (based on `.env.example`) with the following:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=...

# Supabase
SUPABASE_URL=...
SUPABASE_KEY=...
VITE_SUPABASE_URL=...
VITE_SUPABASE_KEY=...

# Azure Storage (if used)
AZURE_STORAGE_CONNECTION_STRING=...
```

### Running with Docker (Recommended)
```bash
docker-compose up --build
```
The frontend will be available at `http://localhost` (or configured domain) and backend at `http://localhost:8000`.

### Running Locally (Dev Mode)
**Backend:**
```bash
cd inter-ai-backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

**Frontend:**
```bash
cd inter-ai-frontend
npm install
npm run dev
```

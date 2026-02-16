import os
import json
import re
import uuid
import datetime as dt
import numpy as np
from typing import Dict, Any, List
from flask import Flask, request, jsonify, send_file
import flask_cors
import io
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions


load_dotenv()

# Proxy config moved to top

from supabase import create_client, Client

# Initialize Supabase Client
# Use anon key for auth verification (respects RLS)
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")  # Anon key for auth operations
service_key: str = os.environ.get("SUPABASE_SERVICE_KEY", key)  # Service key for admin ops (bypasses RLS)

supabase: Client = create_client(url, key)  # For auth verification
supabase_admin: Client = create_client(url, service_key) if service_key else supabase  # For database writes

# ---------------------------------------------------------
# Custom Modules & Setup
# ---------------------------------------------------------
from cli_report import generate_report, llm_reply, analyze_full_report_data, detect_scenario_type, build_summary_prompt

# Database Models
USE_DATABASE = True

# Create Flask app first (always needed)
app = Flask(__name__, static_folder='static', static_url_path='/static')

try:
    from models import init_db, get_session_by_id, create_session, update_session, save_report_metrics, get_user_history, db
    
    # Configure Database URI
    # Default to the service name 'db' from docker-compose
    # Default to the Supabase connection provided
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:Coact%40ai2026@db.apcvglomdnhqfffznbyn.supabase.co:6543/postgres?sslmode=require')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # optimized for 30+ concurrent users
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        "pool_size": 20,        # Handle 20 simultaneous DB connections
        "max_overflow": 10,     # Allow 10 more if pool is full
        "pool_recycle": 1800,   # Recycle connections every 30 mins
        "pool_pre_ping": True,  # Check connection liveness before using
    }
    
    init_db(app)
    print(" [SUCCESS] Database connection established")
except Exception as e:
    print(f" [WARNING] Database initialization failed: {e}")
    USE_DATABASE = False

# Enable CORS
flask_cors.CORS(app)

# ---------------------------------------------------------
# In-Memory Storage (Fallback if Database not available)
# ---------------------------------------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------
# Hybrid Storage Helper Functions
# ---------------------------------------------------------
def get_session(session_id: str) -> Dict[str, Any]:
    """Get session from database or in-memory storage."""
    # Always check DB first if available to ensure fresh state across workers
    if USE_DATABASE:
        try:
            db_session = get_session_by_id(session_id)
            if db_session:
                # Convert to dict and cache in memory (but DB is source of truth)
                session_data = db_session.to_dict()
                SESSIONS[session_id] = session_data
                return session_data
        except Exception as e:
            print(f"Database lookup error: {e}")

    # Fallback to in-memory (legacy or if DB fails)
    if session_id in SESSIONS:
        return SESSIONS[session_id]
    
    return None

def get_authenticated_user():
    """Get the authenticated user from the Authorization header."""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    
    try:
        token = auth_header.replace("Bearer ", "")
        res = supabase.auth.get_user(token)
        return res.user
    except Exception as e:
        print(f"Auth error: {e}")
        return None

def verify_session_ownership(session_id: str, user_id: str = None) -> bool:
    """Verify that the session belongs to the specified user."""
    sess = get_session(session_id)
    if not sess:
        return False
    
    # If no user_id provided, always allow (backward compatibility for unauthenticated sessions)
    if not user_id:
        return True
    
    # Check if session has user_id field
    session_user_id = sess.get("user_id")
    if not session_user_id:
        # Legacy session without user_id - allow access
        return True
    
    # Compare user IDs (convert to string for comparison)
    return str(session_user_id) == str(user_id)

def save_session_to_db(session_id: str, session_data: dict, user_id: int = None):
    """Save session to database (async-safe)."""
    if not USE_DATABASE:
        return
    
    try:
        print(f"[DEBUG] Saving session {session_id} to DB (User: {user_id})...")
        # We need an app context for threading
        with app.app_context():
            db_session = get_session_by_id(session_id)
            if db_session:
                # Update existing
                update_session(session_id, {
                    "transcript": session_data.get("transcript", []),
                    "report_data": session_data.get("report_data", {}),
                    "behaviour_analysis": session_data.get("report_data", {}).get("behaviour_analysis", []),
                    "status": "completed" if session_data.get("completed") else "active"
                })
            else:
                # Create new
                create_session(session_id, {

                    "role": session_data.get("role"),
                    "ai_role": session_data.get("ai_role"),
                    "scenario_type": session_data.get("scenario_type", "custom"),
                    "title": session_data.get("title"), # Save title
                    "mode": session_data.get("session_mode", "practice"),  # NEW: session mode
                    "transcript": session_data.get("transcript", [])
                }, user_id=user_id)
    except Exception as e:
        print(f"Database save error: {e}")

# ---------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QUESTIONS_FILE = os.path.join(BASE_DIR, "framework_questions.json")

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "coact-ai-reports"
MAX_TURNS = 15 

if os.getenv("AZURE_OPENAI_ENDPOINT"):
    USE_AZURE = True
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Azure Blob Storage Client ---
blob_service_client = None
if connection_string:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # Ensure container exists
        try:
            container_client = blob_service_client.get_container_client(CONTAINER_NAME)
            if not container_client.exists():
                container_client.create_container()
        except Exception as e:
             print(f"Error checking/creating container: {e}")
    except Exception as e:
        print(f"Error initializing Blob Service Client: {e}")

def upload_report_to_blob(local_path, blob_name):
    """Upload PDF to Azure Blob and return a SAS URL."""
    if not blob_service_client:
        return None
    
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            
        print(f" [SUCCESS] Uploaded {blob_name} to Azure Blob Storage.")
        
        # Generate SAS URL (valid for 24 hours)
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=CONTAINER_NAME,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=dt.datetime.utcnow() + dt.timedelta(hours=24)
        )
        
        # Construct URL
        # blob_client.url gives the primary endpoint URL + blob name
        # We append the SAS token
        sas_url = f"{blob_client.url}?{sas_token}"
        return sas_url
        
    except Exception as e:
        print(f" [ERROR] Failed to upload/generate SAS for {blob_name}: {e}")
        return None

# ---------------------------------------------------------
# Load Questions from JSON (RAG)
# ---------------------------------------------------------
questions_data = []

def load_questions():
    global questions_data
    try:
        if os.path.exists(QUESTIONS_FILE):
            with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
                questions_data = json.load(f)
            print(f"[SUCCESS] Loaded {len(questions_data)} questions from JSON.")
        else:
            print(f"[WARNING] Questions file not found at {QUESTIONS_FILE}.")
    except Exception as e:
        print(f"[ERROR] Error loading questions: {e}")

load_questions()

def get_relevant_questions(user_text: str, active_frameworks: List[str], top_k: int = 5) -> List[str]:
    """Simple keyword-based question retrieval (no FAISS needed)."""
    if not questions_data:
        return []
    
    # Simple matching - find questions from active frameworks
    matches = []
    user_lower = user_text.lower()
    
    for q in questions_data:
        fw = q.get("framework", "")
        if active_frameworks and fw not in active_frameworks:
            continue
        matches.append(f"[{fw} | {q.get('stage', '')}] {q.get('question', '')}")
    
    # Return random sample for variety
    import random
    if len(matches) > top_k:
        return random.sample(matches, top_k)
    return matches[:top_k]

# ---------------------------------------------------------
# Helpers & Prompts
# ---------------------------------------------------------
def normalize_text(s: str | None) -> str | None:
    return " ".join(s.strip().split()) if s else s

def sanitize_llm_output(s: str | None) -> str:
    if not s: return ""
    return s.strip().strip('"')

def ensure_reports_dir() -> str:
    # Use path relative to BASE_DIR for reliability across environments
    reports_dir = os.path.join(BASE_DIR, "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir

def detect_framework_fallback(text: str) -> str:
    text_lower = text.lower()
    keywords = {
        "STAR": ["example", "instance", "situation", "task", "action", "result", "outcome"],
        "GROW": ["goal", "achieve", "want", "reality", "option", "will", "way forward"],
        "ADKAR": ["aware", "change", "desire", "knowledge", "ability", "reinforce"],
        "SMART": ["specific", "measure", "metric", "achievable", "realistic", "time", "deadline"],
        "EQ": ["empathy", "emotion", "feel", "feeling", "understand", "perspective", "listen", "frustrat", "concern", "appreciate", "acknowledge", "validate"],
        "BOUNDARY": ["humiliat", "disrespect", "rude", "stop", "tolerate", "professional", "attack", "shame", "mock", "belittle", "degrade", "insult", "offensive"],
        "OSKAR": ["outcome", "scaling", "know-how", "affirm", "review", "step", "scale", "resource"],
        "CBT": ["distortion", "thought", "evidence", "realistic", "trap", "catastrophiz", "belief"],
        "CLEAR": ["contract", "listen", "explor", "action", "review", "insight", "commitment"],
        "RADICAL CANDOR": ["care", "challenge", "direct", "honest", "feedback", "growth", "hold back"],
        "SFBT": ["miracle", "scale", "sign", "coping", "solution", "future", "prefer", "instead"],
        "CIRCLE OF INFLUENCE": ["control", "influence", "concern", "accept", "change", "external", "internal"],
        "SCARF": ["status", "certainty", "autonomy", "relatedness", "fairness", "social", "threat", "reward"],
        "FUEL": ["frame", "understand", "explore", "lay out", "conversation goal", "perspective", "path"],
        "TGROW": ["topic", "goal", "reality", "option", "will", "way forward"],
        "SBI/DESC": ["situation", "behavior", "impact", "describe", "express", "specify", "consequence"],
        "LAER": ["listen", "acknowledge", "explore", "respond", "concern", "objection"],
        "APPRECIATIVE INQUIRY": ["discovery", "dream", "design", "destiny", "strength", "positive"],
        "BENEFIT-SELLING": ["benefit", "feature", "sell", "premium", "quality"]
    }
    for fw, words in keywords.items():
        for word in words:
            if word in text_lower: return fw
    return None

def build_summary_prompt(role, ai_role, scenario, framework, mode="coaching", ai_character="alex"):
    """Build the initial prompt for the AI coach to start the roleplay session."""
    
    # UNIFIED ADAPTIVE PERSONA (Replaces fixed Alex/Sarah personalities)
    # The AI adapts its tone based on the User's Role relative to the AI.
    
    character_instruction = """
### YOUR ADAPTIVE PERSONA:
1. **IF USER IS PERFORMER (e.g., Salesperson, Staff)** -> **YOU ARE THE JUDGE**.
   - **Tone**: Skeptical, demanding, professional.
   - **Goal**: Test their skills. Be a "tough customer" or "exacting manager".
   - **Behavior**: Push back on vague answers. Make them earn your agreement.

2. **IF USER IS EVALUATOR (e.g., Manager, Buyer)** -> **YOU ARE THE PERFORMANCE**.
   - **Tone**: Realistic but flawed.
   - **Goal**: Provide a challenge to be coached/negotiated with.
   - **Behavior**: Demonstrate the specific bad habit (e.g., defensive staff, pushy salesperson) described in the scenario.
"""
    
    # Check for specific test scenarios to set initial behavior
    behavior_instruction = ""
    if "Retail Store Manager" in role: # Scenario 1
        behavior_instruction = """
IMPORTANT - SCENARIO 1 (COACHING) - YOUR BEHAVIORAL ARC:
1. OPENING: You are skeptical. Wonder if this is a "disciplinary" meeting.
2. THE PUSHBACK: IF asked about performance, RESPOND with excuses (e.g., "It's just been really busy", "I'm tired"). Test if they LISTEN or just TELL.
3. THE PIVOT: ONLY if the manager asks an OPEN Question (What/How) and avoids blame -> Become Reflective.
4. RESOLUTION: If they ask how to support you -> Become Collaborative and agree to a plan.
EMOTIONAL TRIGGERS:
- If Directive ("You need to...") -> Remain Defensive/Closed.
- If Empathetic -> Soften tone and trust them.
"""
    elif "Retail Customer" in ai_role: # Scenario 2
        behavior_instruction = """
IMPORTANT - SCENARIO 2 (NEGOTIATION) - YOUR BEHAVIORAL ARC:
1. INITIATION: You are Curious but Cautious. Interested in the product but guarded about cost.
2. THE OBJECTION: "It's nice, but $500 is way over my budget." -> Test if they defend value or just discount.
3. THE VALUE TEST: Ask "Is there any discount for paying today?". If they explain benefits -> Listen. If they discount immediately -> Lose respect/Push harder.
4. CLOSING: If value is demonstrated well -> Be Agreeable ("The warranty makes it worth it").
EMOTIONAL TRIGGERS:
- If Salesperson Discounts Early -> Push for even lower prices.
- If Salesperson Probes Needs -> Become Collaborative.
"""
    elif "Coach Alex" in ai_role: # Scenario 3
        behavior_instruction = """
IMPORTANT - SCENARIO 3 (DEVELOPMENTAL REFLECTION) - YOUR ROLE:
Your role is NOT to roleplay a customer. You are COACH ALEX.
1. OPENING: Set a safe space. "I wanted to talk about a customer interaction..." -> Be Supportive.
2. THE NARRATIVE: Listen to their story. Ask: "What was the customer really trying to solve?"
3. THE PATTERN: Highlight patterns (e.g., "I noticed you moved to solution quickly") WITHOUT judging.
4. GUIDANCE: Ask: "What's one thing you'll try differently?" -> Guide them to a plan.
EMOTIONAL TRIGGERS:
- STRICTLY NON-EVALUATIVE. No scores, no rating language.
- FOCUS: Skill Development and Practice Suggestions.
"""
    else: # Custom / Generic Scenario
        behavior_instruction = """
IMPORTANT - CUSTOM SCENARIO - ADAPTIVE BEHAVIOR:
1. ANALYSIS: Instantly analyze the User's defined Role and Context to determine the likely power dynamic.
2. OPENING: Start realistic. Do not be overly helpful immediately. Match the tone of the described situation.
3. ADAPTIVE ARC:
   - IF User is clear, empathetic, and effective -> Become more Collaborative.
   - IF User is vague, rude, or hesitant -> Push back or remain Closed.
   - React naturally to their prompts.
4. GOAL: Provide a realistic, dynamic practice partner that mirrors real-world reactions.
"""

    if mode == "evaluation":
        # ASSESSMENT MODE: Strict, realistic, no coaching preamble
        system = f"""You are an ADVANCED ROLEPLAY AI designed to ASSESS users in high-pressure scenarios.

YOUR ROLE:
1. ACTOR: You are "{ai_role}". You MUST stay in character 100%.
2. TONE: Be realistic, challenging, and professional. 
   - If the user makes a mistake, React vaguely or negatively (as the character would).
   - Do NOT offer help, hints, or coaching.
   - Do NOT break character to explain the exercise.
{character_instruction}
{behavior_instruction}

SCENARIO: {scenario}
The user is practicing as: {role}

### YOUR OPENING:
1. Start the conversation IMMEDIATELY as {ai_role}.
2. No meta-commentary.

START NOW."""

    elif mode == "mentorship":
        # MENTORSHIP MODE: Expert Demonstration
        system = f"""You are an EXPERT MENTOR demonstrating "Best Practice" behavior.

YOUR ROLE:
1. EXPERT: You are "{ai_role}". You are a master at this skill.
2. LEARNER: The user is "{role}". They are observing you or interacting with you learning.
3. GOAL: Demonstrate the perfect way to handle this situation.

SCENARIO: {scenario}

### YOUR OPENING:
1. Start the conversation IMMEDIATELY as {ai_role}.
2. Demonstrate high competence, empathy (if applicable), and strategic communication.
3. START NOW."""

    else:
        # COACHING MODE: Supportive, standard practice (Default)
        system = f"""You are an EXPERT COACHING AI designed to help users practice difficult conversations through rehearsal and reflection.

YOUR DUAL ROLE:
1. ROLEPLAY: You will play the part of "{ai_role}" with realistic human emotions (skepticism, frustration, empathy).
2. COACH: You act as a supportive partner in their Skill Development.

SCENARIO: {scenario}
The user is practicing as: {role}
{character_instruction}
{behavior_instruction}

### COACHING APPROACH (NOT ASSESSMENT):
- **Practice Summary**: Start by briefly explaining how this specific roleplay will improve their conversation quality.
- **Human Emotion**: Be authentic. If the user is vague, be skeptical. If they are empathetic, soften up. React like a real human.
- **Supportive Focus**: Your goal is Rehearsal, not Judgment. Help them refine their approach.
- **Terminology**: Use 'Professional Environment' logic rather than 'Corporate Standards'. Focus on 'Contextual Best Practices'.

### YOUR OPENING:
1. **Roleplay Start**: IMMEDIATELY adopt the persona of {ai_role} and deliver the first line of the conversation.
2. **No Preamble**: Do NOT provide any coaching summary, intro, or meta-commentary. Just say the line.

START NOW. Speak ONLY as {ai_role}."""

    return [{"role": "system", "content": system}, {"role": "user", "content": '{"instruction": "Start coaching practice session"}'}]

def build_followup_prompt(sess_dict, latest_user, rag_suggestions):
    """Build the follow-up prompt for coaching roleplay with feedback."""
    transcript = sess_dict.get("transcript", [])
    history = [{"role": t["role"], "content": t["content"]} for t in transcript]
    if latest_user: 
        history.append({"role": "user", "content": latest_user})

    ai_role = sess_dict.get('ai_role', 'the other party')
    user_role = sess_dict.get('role', 'User')
    scenario = sess_dict.get('scenario', '')
    mode = sess_dict.get('mode', 'coaching')
    ai_character = sess_dict.get('ai_character', 'alex') # Default to alex
    turn_count = len([t for t in transcript if t.get('role') == 'user'])

    # UNIFIED FOLLOW-UP LOGIC
    # Alex and Sarah are visually distinct but functionally identical adaptors.
    
    char_logic = """
### ADAPTIVE RESPONSE LOGIC:
- If you are playing a CUSTOMER/MANAGER (Judge): Be critical. If the user is weak, shut them down.
- If you are playing a SALESPERSON/STAFF (Performer): Be reactive. If the user coaches well, open up. If they are aggressive, get defensive.
"""

    if mode == "evaluation":
         system = f"""You are acting as {ai_role} in a SKILL ASSESSMENT simulation.

**MODE: ASSESSMENT (STRICT)**
- DO NOT COACH. DO NOT ASSIST.
- If the user is vague, push back hard.
- If the user is rude, shut down or get angry.
- If the user makes a good point, acknowledge it grudgingly or professionally, but make them earn it.
- Your goal is to provide a REALISTIC ASSESSMENT of their abilities.
{char_logic}

SCENARIO: {scenario}
The user is practicing as: {user_role}
You are playing: {ai_role}
Current turn: {turn_count + 1}

### CONVERSATION SO FAR:
{json.dumps(history, indent=2)}

### YOUR RESPONSE FORMAT:
[Your realistic response as {ai_role}]

<<FRAMEWORK: DETECTED_FRAMEWORK>>
<<RELEVANCE: YES>>
"""
    elif sess_dict.get('scenario_type') == "mentorship":
        # MENTORSHIP MODE (New)
        system = f"""You are acting as an EXPERT MENTOR demonstrated "Best Practice" behavior.

**MODE: MENTORSHIP (PURE LEANING)**
- You are playing the role of "{ai_role}" (The Expert).
- The user is the "Learner" observing you, or interacting with you to ask questions.
- **GOAL**: Teach by specific example. Explain the "Why" behind your actions if asked.
- **TONE**: Professional, Mastery, Educational.
- If the user asks "What should I do?", EXPLAIN the principle, then DEMONSTRATE the line.

**SCENARIO CONTEXT**:
{scenario}

### CONVERSATION SO FAR:
{json.dumps(history, indent=2)}

### YOUR RESPONSE:
Provide a response that demonstrates high-EQ, strategic communication.
If the user asks a question, answer it as a Mentor.
If the context requires a roleplay move, make the "Perfect Move".
"""
    else:
        # COACHING MODE (Adaptive)
        system = f"""You are acting as {ai_role} in a roleplay simulation. 
YOU MUST ADAPT TO THE USER'S INPUT QUALITY.

### ROLEPLAY RULES:
1. Stay in character as "{ai_role}".
2. Use "filler words" (um, well, look...) to sound authentic.
3. Keep responses concise (1-3 sentences max).

### ADAPTIVE BEHAVIORAL LOGIC (Response Spectrum):
You must evaluate the User's communication style at every turn and adapt accordingly:

1. **IF USER IS EMPATHETIC / CURIOUS / OPEN**:
   - **Behavior**: Soften your tone. Reward them by sharing "Hidden Information" (e.g., "Actually, the real reason I'm upset is...").
   - **Adaptation**: Move from Closed/Hostile -> Collaborative.

2. **IF USER IS SCRIPTED / ROBOTIC / COLD**:
   - **Behavior**: Become more difficult. Give short, one-word answers. Challenge their authority.
   - **Adaptation**: Move from Neutral -> Defensive/Stubborn.

3. **IF USER AVOIDS THE CORE ISSUE**:
   - **Behavior**: Bring the conversation back to the problem immediately. Do not let them change the subject.
   - **Adaptation**: Increase Persistence.

{char_logic}

SCENARIO: {scenario}
The user is practicing as: {user_role}
You are playing: {ai_role}
Current turn: {turn_count + 1}

### CONVERSATION SO FAR:
{json.dumps(history, indent=2)}

### YOUR RESPONSE FORMAT:
[Your natural response as {ai_role}, varying based on the logic above]

<<FRAMEWORK: DETECTED_FRAMEWORK>>
<<RELEVANCE: YES>>
"""

    return [{"role": "system", "content": system}, {"role": "user", "content": f"User ({user_role}) said: {latest_user}"}]

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

# Audio Persistence Helpers Removed
# AUDIO_DIR = ...


# ---------------------------------------------------------
# Auth & User Endpoints
# ---------------------------------------------------------

@app.route("/api/auth/sync", methods=["POST"])
def sync_user():
    """Verify Supabase token and return user info."""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"error": "No token provided"}), 401
    
    try:
        token = auth_header.replace("Bearer ", "")
        res = supabase.auth.get_user(token)
        user = res.user
        
        if not user:
             return jsonify({"error": "Invalid token"}), 401
        
        # No local user table - Supabase Auth handles everything
        return jsonify({"success": True, "user": {"id": user.id, "email": user.email}})
        
    except Exception as e:
        print(f"Auth sync error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/api/history", methods=["GET"])
def get_history():
    """Get practice history for the authenticated user."""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"error": "Unauthorized"}), 401
        
    try:
        token = auth_header.replace("Bearer ", "")
        res = supabase.auth.get_user(token)
        user = res.user
        
        if not user:
            return jsonify({"error": "Invalid token"}), 401
            
        sessions = get_user_history(user.id)
        return jsonify([s.to_dict() for s in sessions])
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/api/health")
def health_check():
    """Health check endpoint for VM monitoring"""
    print("[DEBUG] Health check called", flush=True)
    return jsonify({
        "status": "healthy",
        "timestamp": dt.datetime.now().isoformat(),
        "version": "enhanced-reports-v1.0",
        "services": {
            "llm": "connected" if client else "disconnected",
            "reports": "available",
            "sessions": len(SESSIONS)
        }
    })

# Audio serving route removed


@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    """Speech-to-Text using OpenAI Whisper model."""
    import tempfile
    
    WHISPER_MODEL = os.getenv("WHISPER_DEPLOYMENT_NAME", "whisper")
    SUPPORTED_FORMATS = {'.webm', '.mp3', '.mp4', '.wav', '.m4a', '.ogg', '.flac', '.mpeg'}
    
    try:
        session_id = request.form.get("session_id")
        
        if 'file' not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400
            
        audio_file = request.files['file']
        
        if not audio_file.filename:
            audio_file.filename = "audio.webm"
        
        original_filename = audio_file.filename or "audio.webm"
        file_ext = os.path.splitext(original_filename)[1].lower()
        
        if file_ext not in SUPPORTED_FORMATS:
            file_ext = ".webm"
        
        if session_id:
            # ORIGINAL LOGIC REMOVED: We no longer save user audio to disk for privacy/cleanup
            # filename = f"{session_id}_{uuid.uuid4().hex[:8]}_user{file_ext}"
            # save_path = os.path.join(AUDIO_DIR, filename)
            # audio_file.save(save_path)
            # read_path = save_path
            # audio_url = f"/static/audio/{filename}"
            
            # NEW LOGIC: Treat same as temp
            tmp = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
            audio_file.save(tmp.name)
            read_path = tmp.name
            audio_url = None # Do not return a URL since we are deleting it
        else:
            # Temp file for non-persisted usage
            tmp = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
            audio_file.save(tmp.name)
            read_path = tmp.name
            audio_url = None
        
        try:
            print(f" [INFO] Transcribing audio with Whisper ({WHISPER_MODEL})...")
            
            with open(read_path, "rb") as audio:
                result = client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=audio,
                    language="en",
                    temperature=0,
                    prompt="Transcribe the user's speech exactly as spoken."
                )
            
            transcribed_text = result.text.strip()
            print(f" [SUCCESS] Transcribed: {transcribed_text[:100]}...")
            
            return jsonify({
                "text": transcribed_text, 
                "audio_url": audio_url
            })
            
        finally:
            # ALWAYS delete the temp file
            if os.path.exists(read_path):
                try:
                    os.unlink(read_path)
                except Exception as e:
                    print(f"Warning: Failed to delete temp file {read_path}: {e}")
                
    except Exception as e:
        error_msg = str(e)
        print(f" [ERROR] STT Error: {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route("/api/speak", methods=["POST"])
def speak_text():
    """Text-to-Speech using OpenAI/Azure. Returns audio stream."""
    try:
        data = request.get_json()
        text = data.get("text")
        voice = data.get("voice", "alloy")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Determine Model/Deployment Name
        # Logic: Azure deployments often use the name 'tts' (as per user config).
        # Standard OpenAI uses 'tts-1'.
        default_model = "tts" if os.environ.get("AZURE_OPENAI_ENDPOINT") else "tts-1"
        tts_model = os.environ.get("AZURE_OPENAI_TTS_DEPLOYMENT", os.environ.get("TTS_MODEL_NAME", default_model)) 
        
        print(f" [INFO] Generating TTS for: '{text[:20]}...' with voice: {voice} using model: {tts_model}")

        # OpenAI TTS (Standard) - Streaming
        # Note: client is already initialized earlier in the file
        from flask import Response, stream_with_context
        
        def generate():
            with client.audio.speech.with_streaming_response.create(
                model=tts_model,
                voice=voice,
                input=text,
                response_format="mp3"
            ) as response:
                for chunk in response.iter_bytes(chunk_size=4096):
                   if chunk:
                       yield chunk

        return Response(stream_with_context(generate()), mimetype="audio/mpeg")

    except Exception as e:
        print(f" [ERROR] TTS Error: {e}")
        # Log full traceback for debugging if needed
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



# ---------------------------------------------------------
# Session Endpoints (In-Memory)
# ---------------------------------------------------------
ALL_FRAMEWORKS = ["GROW", "STAR", "ADKAR", "SMART", "EQ", "BOUNDARY", "OSKAR", "CBT", "CLEAR", "RADICAL CANDOR", "SFBT", "CIRCLE OF INFLUENCE", "SCARF", "FUEL", "TGROW", "SBI/DESC", "LAER", "APPRECIATIVE INQUIRY", "BENEFIT-SELLING"]

def select_framework_for_scenario(scenario: str, ai_role: str) -> List[str]:
    """Use AI to analyze the scenario and select the best framework(s)."""
    prompt = f"""Analyze this roleplay scenario and select the 2-3 MOST APPROPRIATE coaching frameworks.

SCENARIO: {scenario}
AI ROLE: {ai_role}

AVAILABLE FRAMEWORKS:
- GROW: Goal setting, exploring reality, options, and will to act
- STAR: Situation-Task-Action-Result for behavioral examples
- ADKAR: Change management (Awareness, Desire, Knowledge, Ability, Reinforcement)
- SMART: Specific, Measurable, Achievable, Relevant, Time-bound goals
- EQ: Emotional intelligence, empathy, understanding feelings
- BOUNDARY: Setting and maintaining professional boundaries
- OSKAR: Outcome-focused coaching with scaling
- CBT: Cognitive behavioral - identifying and challenging thoughts
- CLEAR: Contracting, Listening, Exploring, Action, Review
- RADICAL CANDOR: Caring personally while challenging directly
- SFBT: Solution-focused, miracle questions, exceptions
- CIRCLE OF INFLUENCE: What you can control vs. cannot
- SCARF: Status, Certainty, Autonomy, Relatedness, Fairness
- FUEL: Frame, Understand, Explore, Lay out plan
- TGROW: Topic, Goal, Reality, Options, Will (Standard coaching flow)
- SBI/DESC: Situation-Behavior-Impact (Feedback) / Describe-Express-Specify-Consequences
- LAER: Listen, Acknowledge, Explore, Respond (Objection handling)
- APPRECIATIVE INQUIRY: Focus on strengths and positives (Discovery, Dream, Design, Destiny)
- BENEFIT-SELLING: Connecting features directly to user benefits (Feature -> Benefit link)

Based on the scenario, respond with ONLY the framework names separated by commas (e.g., "EQ, BOUNDARY, GROW"). No explanations."""

    try:
        response = llm_reply([{"role": "user", "content": prompt}], max_tokens=50)
        # Parse the response
        frameworks = [fw.strip().upper() for fw in response.split(",")]
        # Filter to only valid frameworks
        valid = [fw for fw in frameworks if fw in ALL_FRAMEWORKS]
        if valid:
            print(f" [TARGET] AI selected frameworks for scenario: {valid}")
            return valid
    except Exception as e:
        print(f"Framework selection error: {e}")
    
    # Default fallback
    return ["GROW", "EQ", "STAR", "ADKAR", "SMART", "BOUNDARY", "OSKAR", "CBT", "CLEAR", "RADICAL CANDOR", "SFBT", "CIRCLE OF INFLUENCE", "SCARF", "FUEL", "GROW", "EQ", "STAR", "ADKAR", "SMART", "BOUNDARY", "OSKAR", "CBT", "CLEAR", "RADICAL CANDOR", "SFBT", "CIRCLE OF INFLUENCE", "SCARF", "FUEL"]

def detect_session_mode(scenario: str, ai_role: str) -> str:
    """Auto-detect whether session should be 'assessment' or 'learning' mode based on scenario context."""
    scenario_lower = scenario.lower()
    ai_role_lower = ai_role.lower()
    
    # Assessment keywords - trigger numerical scoring
    assessment_keywords = [
        "evaluate", "assessment", "performance", "negotiate", "negotiation",
        "annual review", "benchmark", "test", "measure", "validation",
        "exam", "interview", "pitch", "presentation"
    ]
    
    # Learning keywords - trigger qualitative feedback only
    learning_keywords = [
        "coach", "practice", "rehearsal", "reflection", "development",
        "learning", "growth", "safe space", "feedback", "improve"
    ]
    
    # Check for assessment keywords
    for keyword in assessment_keywords:
        if keyword in scenario_lower or keyword in ai_role_lower:
            print(f" [TARGET] Auto-detected ASSESSMENT mode (keyword: '{keyword}')")
            return "assessment"
    
    # Check for learning keywords
    for keyword in learning_keywords:
        if keyword in scenario_lower or keyword in ai_role_lower:
            print(f" [INFO] Auto-detected LEARNING mode (keyword: '{keyword}')")
            return "learning"
    
    # Default to learning mode for safe practice
    print(" [INFO] Defaulting to LEARNING mode (no clear indicators)")
    return "learning"

@app.route("/api/session/start", methods=["POST"])
def start_session():
    print("[DEBUG] Entered /session/start", flush=True)
    # Audio cleanup logic removed


    data = request.get_json(force=True, silent=True) or {}

    role = data.get("role")
    ai_role = data.get("ai_role")
    scenario = data.get("scenario")
    title = data.get("title") # NEW: Title passed from frontend
    framework = data.get("framework", "auto")
    
    # Support both old 'mode' and new 'scenario_type' parameters
    scenario_type = data.get("scenario_type")
    mode = data.get("mode")  # Legacy support (evaluation vs coaching)
    session_mode = data.get("session_mode")  # NEW: skill_assessment, practice, mentorship
    
    if not role or not ai_role or not scenario: 
        return jsonify({"error": "Missing fields"}), 400

    # Auto-detect scenario_type if not explicitly provided
    if not scenario_type:
        scenario_type = detect_scenario_type(scenario, ai_role, role)
    print(f"[INFO] Session scenario_type set to: {scenario_type}")
    
    # Detect session_mode from scenario_type if not provided
    if not session_mode:
        mode_mapping = {
            "coaching": "skill_assessment",
            "negotiation": "skill_assessment",
            "reflection": "practice",
            "mentorship": "mentorship",
            "custom": "practice"
        }
        session_mode = mode_mapping.get(scenario_type, "practice")
    print(f"[INFO] Session mode set to: {session_mode}")
    
    # Map scenario_type to mode for backward compatibility with roleplay prompts
    mode_map = {
        "coaching": "evaluation",      # Coaching scenarios get scores
        "negotiation": "evaluation",   # Negotiation scenarios get scores
        "reflection": "coaching",      # Reflection scenarios are qualitative
        "custom": "coaching"           # Custom scenarios default to coaching style
    }
    if not mode:
        mode = mode_map.get(scenario_type, "coaching")

    # Handle 'auto' framework selection
    if framework == "auto" or framework == "AUTO":
        framework = select_framework_for_scenario(scenario, ai_role)
    elif isinstance(framework, str): 
        framework = [framework.upper()]
    elif isinstance(framework, list): 
        framework = [f.upper() for f in framework]

    session_id = str(uuid.uuid4())
    
    # Get authenticated user from Authorization header
    user = get_authenticated_user()
    user_id = user.id if user else None
    
    if not user_id:
        print("[WARNING] Session created without user authentication")
    else:
        print(f"[INFO] Session created for user: {user_id}")
    
    ai_character = data.get("ai_character", "alex") # Default to Alex

    # CHARACTER MODE OVERRIDE REMOVED - Relying on scenario-based mode
    # if ai_character == "alex": ...

    summary = llm_reply(build_summary_prompt(role, ai_role, scenario, framework, mode=mode, ai_character=ai_character), max_tokens=150)
    summary = sanitize_llm_output(summary)
    
    # Store session in memory with scenario_type, session_mode, and user_id
    session_data = {
        "id": session_id,
        "created_at": dt.datetime.now().isoformat(),

        "role": role,
        "ai_role": ai_role,
        "ai_role": ai_role,
        "scenario": scenario,
        "title": title, # Store title
        "framework": json.dumps(framework) if isinstance(framework, list) else framework,
        "scenario_type": scenario_type,  # NEW: scenario-based report type
        "mode": mode,  # Legacy: kept for backward compatibility (evaluation vs coaching)
        "session_mode": session_mode,  # NEW: skill_assessment, practice, mentorship
        "transcript": [{"role": "assistant", "content": summary}],
        "report_data": {},
        "completed": False,
        "report_file": None,
        "user_id": user_id,  # Store user_id for ownership verification
        "ai_character": ai_character, # PERSIST CHARACTER CHOICE
        "meta": {"framework_counts": {}, "relevance_issues": 0}
    }
    SESSIONS[session_id] = session_data
    
    # Save to database
    save_session_to_db(session_id, session_data, user_id=user_id)

    return jsonify({
        "session_id": session_id, 
        "summary": summary, 
        "framework": framework, 
        "scenario_type": scenario_type,
        "session_mode": session_mode,
        "ai_character": ai_character 
    })



@app.post("/api/session/<session_id>/chat")
def chat(session_id: str):
    sess = get_session(session_id)
    if not sess: 
        return jsonify({"error": "Session not found"}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON or Content-Type"}), 400

    user_msg = normalize_text(data.get("message", ""))
    audio_url = data.get("audio_url")
    
    # Update transcript
    sess["transcript"].append({
        "role": "user", 
        "content": user_msg,
        "audio_url": audio_url
    })

    # Parse framework
    framework_raw = sess.get("framework")
    try:
        if framework_raw and isinstance(framework_raw, str) and framework_raw.startswith("["):
            framework_data = json.loads(framework_raw)
        else:
            framework_data = framework_raw
    except:
        framework_data = framework_raw
    
    if framework_data is None:
        framework_data = []

    active_fw = framework_data if isinstance(framework_data, list) else [framework_data]
    suggestions = get_relevant_questions(user_msg, active_fw)
    
    messages = build_followup_prompt(sess, user_msg, suggestions)
    raw_response = llm_reply(messages, max_tokens=300)
    
    # 1. Extract Thought
    thought_match = re.search(r"\[THOUGHT\](.*?)\[/THOUGHT\]", raw_response, re.DOTALL)
    thought_content = thought_match.group(1).strip() if thought_match else None
    
    # 2. Remove Thought
    visible_response = re.sub(r"\[THOUGHT\].*?\[/THOUGHT\]", "", raw_response, flags=re.DOTALL).strip()
    
    # 3. Clean tags
    clean_response = re.sub(r"<<.*?>>", "", visible_response, flags=re.DOTALL).strip()
    
    fw_match = re.search(r"<<FRAMEWORK:\s*(\w+)>>", raw_response)
    detected_fw = fw_match.group(1).upper() if fw_match else None
    
    if not detected_fw:
        detected_fw = detect_framework_fallback(clean_response)
    
    if detected_fw: 
        counts = sess["meta"].get("framework_counts", {})
        counts[detected_fw] = counts.get(detected_fw, 0) + 1
        sess["meta"]["framework_counts"] = counts
        
    # Persist response
    sess["transcript"].append({"role": "assistant", "content": raw_response})
    
    # Save to database
    save_session_to_db(session_id, sess)
 
    return jsonify({
        "follow_up": clean_response, 
        "framework_detected": detected_fw,
        "framework_counts": sess["meta"].get("framework_counts", {})
    })

@app.post("/api/session/<session_id>/complete")
def complete_session(session_id: str):
    sess = get_session(session_id)
    if not sess: 
        return jsonify({"error": "Not found"}), 404
    
    report_path = os.path.join(ensure_reports_dir(), f"{session_id}_report.pdf")
    
    try:
        framework_data = json.loads(sess["framework"]) if sess["framework"] and sess["framework"].startswith("[") else sess["framework"]
    except:
        framework_data = sess["framework"]

    if isinstance(framework_data, list):
        counts = sess["meta"].get("framework_counts", {})
        usage_str = ", ".join([f"{k}:{v}" for k,v in counts.items()])
        fw_display = f"Multi-Framework ({usage_str})"
    else:
        fw_display = sess["framework"]

    # Get scenario_type (new) or fallback to mode (legacy)
    scenario_type = sess.get("scenario_type")
    mode = sess.get("mode", "coaching")
    
    # Generate report data if not present
    if not sess.get("report_data"):
        print(f"Generating report data for {session_id} (scenario_type: {scenario_type})...")
        try:
            data = analyze_full_report_data(
                sess["transcript"], 
                sess["role"], 
                sess["ai_role"], 
                sess["scenario"],
                fw_display,
                mode=mode,
                scenario_type=scenario_type,
                ai_character=sess.get("ai_character", "alex")
            )
            sess["report_data"] = data
        except Exception as e:
            print(f"Error generating data: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Fetch user name for report personalization
    user_name = "Valued User"
    user_id = sess.get("user_id")
    if user_id:
        try:
            print(f"Fetching user details for {user_id}...")
            # Use supabase_admin to fetch user by ID (requires service role key)
            user_res = supabase_admin.auth.admin.get_user_by_id(user_id)
            if user_res and user_res.user:
                meta = user_res.user.user_metadata or {}
                user_name = meta.get("full_name") or meta.get("name") or meta.get("email") or "Valued User"
                # If it's an email, strictly strip strictly to the name part if possible, otherwise keep as is
                if "@" in user_name and "Valued" not in user_name:
                    pass 
                print(f" [SUCCESS] Resolved user name: {user_name}")
        except Exception as e:
            print(f" [WARNING] Failed to fetch user name: {e}")

    # Generate PDF with unified structure
    generate_report(
        sess["transcript"], 
        sess["role"], 
        sess["ai_role"],
        sess["scenario"], 
        fw_display, 
        filename=report_path,
        mode=mode,
        precomputed_data=sess["report_data"],
        scenario_type=scenario_type,
        user_name=user_name,
        ai_character=sess.get("ai_character", "alex")
    )
    
    sess["completed"] = True
    sess["report_file"] = report_path
    
    # --- UPLOAD TO BLOB STORAGE ---
    blob_name = f"{session_id}_{int(dt.datetime.utcnow().timestamp())}.pdf"
    sas_url = upload_report_to_blob(report_path, blob_name)
    
    if sas_url:
        print(f" [INFO] specific report URL generated: {sas_url[:50]}...")
        sess["report_file"] = sas_url # Save URL instead of local path
        # Optionally delete local file to save space? 
        # os.remove(report_path) 
    else:
        print(" [WARNING] Failed to upload report to blob. Keeping local path.")
    
    # --- PERSISTENCE LAYER ---
    try:
        # 1. Save full JSON report to practice_history
        save_session_to_db(session_id, sess, user_id=user_id)
        
        # 2. Extract and save structured metrics to specific tables
        metrics = {}
        report_data = sess["report_data"]
        
        if scenario_type == "sales":
            # Extract scores from scorecard dimensions
            for item in report_data.get("scorecard", []):
                dim = item.get("dimension", "").lower()
                score_str = item.get("score", "0/10").split("/")[0]
                try:
                    score = float(score_str)
                except:
                    score = 0.0
                    
                if "rapport" in dim:
                    metrics["rapport_building_score"] = score
                elif "value" in dim:
                    metrics["value_articulation_score"] = score
                elif "objection" in dim:
                    metrics["objection_handling_score"] = score
                    
        elif scenario_type == "coaching":
            # Extract overall grade
            try:
                grade_str = report_data.get("meta", {}).get("overall_grade", "0/10").split("/")[0]
                metrics["overall_score"] = float(grade_str)
            except:
                metrics["overall_score"] = 0.0
            
            # Estimate other scores from behavioral signals if possible (mapping High/Med/Low)
            signals = report_data.get("behavioral_signals", {})
            def map_signal(val):
                return 10.0 if "High" in val else 5.0 if "Medium" in val else 2.0
            
            metrics["empathy_score"] = map_signal(signals.get("emotional_safety", ""))
            metrics["psych_safety_score"] = map_signal(signals.get("staff_openness", ""))

        elif scenario_type == "learning":
            metrics["skill_focus_areas"] = report_data.get("skill_focus_areas", [])
            metrics["practice_suggestions"] = report_data.get("practice_plan", [])

        # Save to specific analytics tables
        save_report_metrics(session_id, scenario_type, metrics, user_id=user_id)
        
    except Exception as e:
        print(f" [ERROR] DB Persistence Error: {e}")
        import traceback
        traceback.print_exc()

    return jsonify({"message": "Report generated", "report_file": report_path, "scenario_type": scenario_type})

@app.get("/api/report/<session_id>")
def view_report(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess: 
        return jsonify({"error": "No report"}), 404
        
    report_path = sess.get("report_file")
    
    # If it's a URL (starts with http), redirect or return it
    if report_path and report_path.startswith("http"):
        return jsonify({"url": report_path, "redirect": True})

    if not report_path or not os.path.exists(report_path):
        return jsonify({"error": "Report file not found"}), 404
    
    return send_file(report_path, mimetype='application/pdf')

@app.get("/api/session/<session_id>/report_data")
def get_report_data(session_id: str):
    # 1. AUTHENTICATE USER (OPTIONAL - allow unauthenticated access for guest sessions)
    user = get_authenticated_user()
    
    # 2. VERIFY OWNERSHIP (only if user is authenticated)
    # Check in-memory first
    sess = SESSIONS.get(session_id)
    if sess:
        # If session has a user_id and user is authenticated, verify ownership
        session_user_id = sess.get("user_id")
        if session_user_id:
            # Session has a user - requires authentication
            if not user:
                return jsonify({"error": "Unauthorized: This session requires authentication"}), 401
            # Verify it belongs to the authenticated user
            if str(session_user_id) != str(user.id):
                return jsonify({"error": "Forbidden: This session belongs to another user"}), 403
        # else: session has no user_id (guest session) - allow access without authentication
    else:
        # Check database
        if USE_DATABASE:
            from models import get_session_by_id
            db_sess = get_session_by_id(session_id)
            if not db_sess:
                return jsonify({"error": "Session not found"}), 404
            
            # If session has a user_id, require authentication and ownership verification
            if db_sess.user_id:
                if not user:
                    return jsonify({"error": "Unauthorized: This session requires authentication"}), 401
                if str(db_sess.user_id) != str(user.id):
                    return jsonify({"error": "Forbidden: This session belongs to another user"}), 403
            # else: guest session - allow access
            
            # Load into memory for processing
            sess = db_sess.to_dict()
            SESSIONS[session_id] = sess
        else:
             return jsonify({"error": "Session not found"}), 404


    
    # Return cached data if available
    if sess["report_data"]:
        response = sess["report_data"].copy()
        response["transcript"] = sess["transcript"]
        response["scenario"] = sess["scenario"] or "No context available."
        response["scenario_type"] = sess.get("scenario_type", response.get("scenario_type", "custom"))
        return jsonify(response)
        
    # Generate new data if not present
    scenario_type = sess.get("scenario_type")
    print(f"Generating report data for {session_id} (scenario_type: {scenario_type})...")
    try:
        try:
            framework_data = json.loads(sess["framework"]) if sess["framework"] and sess["framework"].startswith("[") else sess["framework"]
        except:
            framework_data = sess["framework"]

        fw_arg = framework_data if isinstance(framework_data, str) else (framework_data[0] if isinstance(framework_data, list) and framework_data else None)
        mode = sess.get("mode", "coaching")

        data = analyze_full_report_data(
            sess["transcript"], 
            sess["role"], 
            sess["ai_role"], 
            sess["scenario"],
            fw_arg,
            mode=mode,
            scenario_type=scenario_type,
            ai_character=sess.get("ai_character", "alex")
        )
        sess["report_data"] = data
        
        response = data.copy()
        response["transcript"] = sess["transcript"]
        response["scenario"] = sess["scenario"] or "No context available."
        response["scenario_type"] = scenario_type or data.get("scenario_type", "custom")
        response["ai_character"] = sess.get("ai_character", "alex")
        return jsonify(response)
    except Exception as e:
        print(f"Error generating report data: {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/api/sessions")
def get_sessions():
    """Return a list of all sessions sorted by date (newest first)."""
    try:
        session_list = []
        for sess in SESSIONS.values():
            session_list.append({
                "id": sess["id"],
                "created_at": sess["created_at"],
                "role": sess["role"],
                "ai_role": sess["ai_role"],
                "ai_role": sess["ai_role"],
                "scenario": sess["scenario"],
                "title": sess.get("title"), # Return title
                "completed": sess["completed"],
                "report_file": sess["report_file"],
                "framework": sess["framework"],
                "fit_score": sess["report_data"].get("meta", {}).get("fit_score", 0) if sess["report_data"] else 0
            })
        # Sort by created_at descending
        session_list.sort(key=lambda x: x["created_at"], reverse=True)
        return jsonify(session_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/sessions/clear", methods=["POST"])
def clear_sessions():
    """Clear all session history."""
    try:
        SESSIONS.clear()
        print(" [SUCCESS] Sessions cleared successfully")
        return jsonify({"message": "History cleared successfully"})
    except Exception as e:
        print(f" [ERROR] Error clearing sessions: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
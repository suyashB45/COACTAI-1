import json
import os
import math
import re
import unicodedata
import datetime as dt
from fpdf import FPDF
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import httpx
import concurrent.futures


load_dotenv()

import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import tempfile

USE_AZURE = True
# ... imports ... 
def setup_langchain_model():
    # Force httpx to ignore system proxies which cause hangs on Azure VMs
    http_client = httpx.Client(trust_env=False, timeout=30.0)
    
    if USE_AZURE:
        return AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", os.getenv("MODEL_NAME", "gpt-4.1-mini")),
            http_client=http_client,
            temperature=0.4
        )
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), 
        model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        http_client=http_client,
        temperature=0.4
    )

llm = setup_langchain_model()

MODEL_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", os.getenv("MODEL_NAME", "gpt-4.1-mini"))

# --- Premium Modern Palette ---
COLORS = {
    'text_main': (30, 41, 59),       # Slate 800
    'text_light': (100, 116, 139),   # Slate 500
    'white': (255, 255, 255),
    
    # Premium Glassmorphism Palette
    'primary': (15, 23, 42),         # Deep Slate 900
    'secondary': (51, 65, 85),       # Slate 700
    'accent': (59, 130, 246),        # Blue 500 (Primary Brand)
    'accent_light': (96, 165, 250), # Blue 400
    
    # Gradients & UI
    'header_grad_1': (15, 23, 42),   # Slate 900
    'header_grad_2': (30, 58, 138),  # Blue 900
    'score_grad_1': (236, 253, 245), # Emerald 50
    'score_grad_2': (209, 250, 229), # Emerald 100
    'score_text': (4, 120, 87),      # Emerald 700
    
    # Chart Colors
    'chart_fill': (59, 130, 246),    # Blue 500
    'chart_stroke': (37, 99, 235),   # Blue 600
    'sentiment_pos': (16, 185, 129), # Emerald 500
    'sentiment_neg': (239, 68, 68),  # Red 500
    
    # Section colors
    'section_skills': (99, 102, 241),    # Indigo 500
    'section_eq': (236, 72, 153),        # Pink 500
    'section_comm': (14, 165, 233),      # Sky 500
    'section_coach': (245, 158, 11),     # Amber 500
    
    'divider': (226, 232, 240),
    'bg_light': (248, 250, 252),
    'sidebar_bg': (248, 250, 252),
    
    # Status
    'success': (16, 185, 129),       # Emerald 500
    'warning': (245, 158, 11),       # Amber 500
    'danger': (239, 68, 68),         # Red 500
    'rewrite_good': (236, 253, 245), # Emerald 50
    'bad_bg': (254, 226, 226),       # Red 100
    'grey_text': (100, 116, 139),    # Slate 500
    'grey_bg': (241, 245, 249),      # Slate 100
    'purple': (139, 92, 246),        # Violet 500
    'nuance_bg': (236, 72, 153)      # Pink 500 (for EQ nuance badges)
}

# UNIVERSAL REPORT STRUCTURE DEFINITIONS
SCENARIO_TITLES = {
    "universal": {
        "pulse": "THE PULSE",
        "narrative": "THE NARRATIVE",
        "blueprint": "THE BLUEPRINT"
    }
}


def sanitize_text(text):
    if text is None: return ""
    text = str(text)
    # Common replacements to help latin-1
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2022': '*', '\u2026': '...',
        '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2015': '-',
        '\u2032': "'", '\u2033': '"', '\u2039': '<', '\u203a': '>',
        '\u00a0': ' ', '\u00b7': '*', '\u2027': '*', '\u25cf': '*',
        '\u25cb': 'o', '\u25a0': '*', '\u25a1': 'o', '\u2713': 'v',
        '\u2714': 'v', '\u2717': 'x', '\u2718': 'x', '\u2192': '->',
        '\u2190': '<-', '\u2194': '<->', '\u21d2': '=>', '\u2212': '-',
        '\u00d7': 'x', '\u00f7': '/', '\u2264': '<=', '\u2265': '>=',
        '\u2260': '!=', '\u00b0': ' deg', '\u00ae': '(R)', '\u00a9': '(C)',
        '\u2122': '(TM)', '\u00ab': '<<', '\u00bb': '>>', '\u201a': ',',
        '\u201e': '"', '\u2020': '+', '\u2021': '++', '\u00b6': 'P',
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    
    # Ultimate fallback: encode to latin-1 with replacement, then decode back
    # This guarantees no characters outside latin-1 are present
    return text.encode('latin-1', 'replace').decode('latin-1')

def build_summary_prompt(role, ai_role, scenario, framework=None, mode="coaching", ai_character="alex"):
    """
    Constructs the system prompt for the initial summary/greeting generation.
    """
    return [
        {"role": "system", "content": f"You are acting as {ai_character.capitalize()}, a professional coach."},
        {"role": "user", "content": f"Generate a brief welcoming sentence for a {scenario} session where the user plays {role} and you play {ai_role}."}
    ]

def sanitize_data(obj):
    """Recursively sanitize all strings in a dictionary or list for PDF compatibility."""
    if isinstance(obj, str):
        return sanitize_text(obj)
    elif isinstance(obj, dict):
        return {k: sanitize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_data(item) for item in obj]
    else:
        return obj

def get_score_theme(score):
    try: s = float(score)
    except: s = 0.0
    if s == 0.0: return COLORS['grey_bg'], COLORS['grey_text']
    if s >= 7.0: return COLORS['score_grad_1'], COLORS['score_text'] 
    if s >= 5.0: return (254, 249, 195), (161, 98, 7) 
    return (254, 226, 226), (185, 28, 28) 

def get_bar_color(score):
    try: s = float(score)
    except: s = 0.0
    if s >= 8.0: return COLORS['success']
    if s >= 5.0: return COLORS['warning']
    if s > 0.0: return COLORS['danger']
    return COLORS['grey_text']

def llm_reply(messages, max_tokens=4000):
    try:
        print(f" [DEBUG] llm_reply using LangChain model", flush=True)
        # LangChain accepts list of dicts directly in invoke
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "{}"

def detect_scenario_type(scenario: str, ai_role: str, role: str) -> str:
    """Detect scenario type based on content to determine report structure."""
    scenario_lower = scenario.lower()
    ai_role_lower = ai_role.lower()
    role_lower = role.lower()
    
    combined_text = f"{scenario_lower} {ai_role_lower} {role_lower}"

    # 1. REFLECTION / MENTORSHIP (No Scorecard)
    # Trigger if AI is strictly a "Coach" or "Mentor" (Role-based)
    if "coach" in ai_role_lower or "mentor" in ai_role_lower:
        return "reflection"
    
    # Trigger if explicit "learning" or "reflection" keywords in text (Topic-based)
    # Note: Avoid "coach" in text search to prevent matching "Manager coaching staff" (which should be Scored)
    reflection_keywords = ["reflection", "learning plan", "development plan", "self-reflection"]
    if any(kw in combined_text for kw in reflection_keywords):
        return "reflection"
    
    # 2. NEGOTIATION / SALES (Scorecard)
    negotiation_keywords = ["sales", "negotiat", "price", "discount", "buyer", "seller", "deal", "purchase"]
    if any(kw in combined_text for kw in negotiation_keywords):
        return "negotiation"
    
    # 3. COACHING / LEADERSHIP (Scorecard)
    # User is the one doing the coaching/managing
    coaching_keywords = ["coaching", "performance", "feedback", "manager", "supervisor", "staff", "employee"]
    if any(kw in combined_text for kw in coaching_keywords):
        return "coaching"
    
    # 4. DE-ESCALATION (Scorecard)
    deescalation_keywords = ["angry", "upset", "complaint", "calm", "de-escalate"]
    if any(kw in combined_text for kw in deescalation_keywords):
        return "custom" # Currently maps to Custom but we can add specific later
    
    # Default
    return "custom"


def detect_user_role_context(role: str, ai_role: str) -> str:
    """Detect the specific sub-role of the user (e.g., Manager vs Staff, Seller vs Buyer)."""
    role_lower = role.lower()
    
    # Coaching Context
    if any(k in role_lower for k in ["manager", "supervisor", "lead", "coach"]):
        return "manager"
    if any(k in role_lower for k in ["staff", "associate", "employee", "report", "subordinate"]):
        return "staff"
        
    # Sales/Negotiation Context
    if any(k in role_lower for k in ["sales", "account executive", "rep", "seller"]):
        return "seller"
    if any(k in role_lower for k in ["customer", "buyer", "client", "prospect"]):
        return "buyer"
        
    return "unknown"

# =====================================================================
# NEW: Parallel Analysis Functions for Speed Optimization
# =====================================================================

def analyze_character_traits(transcript, role, ai_role, scenario, scenario_type):
    """
    Analyze user's character/personality traits and assess fit for the scenario.
    This runs in PARALLEL with main report generation for speed.
    """
    user_msgs = [t for t in transcript if t['role'] == 'user']
    if not user_msgs:
        return {}
    
    conversation = "\n".join([f"USER: {t['content']}" for t in user_msgs])
    
    # Define required traits based on scenario type
    required_traits_map = {
        "coaching": ["Openness to Feedback", "Accountability", "Active Listening", "Growth Mindset"],
        "negotiation": ["Rapport Building", "Active Listening", "Value Focus", "Confidence"],
        "sales": ["Rapport Building", "Active Listening", "Value Focus", "Confidence"],
        "learning": ["Curiosity", "Reflection", "Openness"],
        "mentorship": ["Observation", "Question Asking", "Pattern Recognition"]
    }
    
    required_traits = required_traits_map.get(scenario_type, ["Professional Communication"])
    
    prompt = f"""
You are analyzing a user's CHARACTER and PERSONALITY in a {scenario_type} simulation.

USER ROLE: {role}
AI ROLE: {ai_role}
SCENARIO: {scenario}

REQUIRED TRAITS FOR SUCCESS: {', '.join(required_traits)}

Analyze the user's character based on their responses:

CONVERSATION:
{conversation}

Return VALID JSON with this EXACT structure:
{{
  "observed_traits": [
    {{
      "trait": "Trait Name (e.g., Defensiveness, Accountability, Curiosity)",
      "evidence_quote": "EXACT quote from conversation",
      "impact": "Positive" or "Negative",
      "insight": "Why this trait helps or hinders success in this scenario"
    }}
  ],
  "scenario_fit": {{
    "required_traits": {json.dumps(required_traits)},
    "user_strengths": ["Traits they demonstrated well"],
    "user_gaps": ["Traits they're missing or weak in"],
    "fit_score": "X/10",
    "fit_assessment": "Overall assessment of character fit",
    "development_priority": "The #1 character trait they need to develop"
  }},
  "character_development_plan": [
    "Specific behavior change 1 (e.g., Practice phrase: 'That's on me...')",
    "Specific behavior change 2 (e.g., Pause 3 seconds before defending)"
  ]
}}

Be SPECIFIC. Quote EXACT words. No generic advice.
"""
    
    try:
        parser = JsonOutputParser()
        prompt_template = PromptTemplate(
            template="{prompt}\n\n{format_instructions}",
            input_variables=["prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt_template | llm | parser
        result = chain.invoke({"prompt": prompt})
        print(" [SUCCESS] Character analysis completed")
        return result
        
    except Exception as e:
        print(f" [ERROR] Character analysis failed: {e}")
        return {
            "observed_traits": [],
            "scenario_fit": {
                "required_traits": required_traits,
                "user_strengths": [],
                "user_gaps": ["Analysis unavailable"],
                "fit_score": "N/A",
                "fit_assessment": "Unable to analyze",
                "development_priority": "N/A"
            },
            "character_development_plan": []
        }


def analyze_questions_missed(transcript, role, ai_role, scenario, scenario_type):
    """
    Identify questions the user SHOULD have asked but didn't.
    This runs in PARALLEL for speed.
    """
    user_msgs = [t for t in transcript if t['role'] == 'user']
    if not user_msgs:
        return {}
    
    conversation = "\n".join([
        f"{'USER' if t['role'] == 'user' else 'AI'}: {t['content']}" 
        for t in transcript
    ])
    
    # Count questions user actually asked
    questions_asked = sum(1 for msg in user_msgs if '?' in msg['content'])
    
    prompt = f"""
You are analyzing QUESTION QUALITY in a {scenario_type} simulation.

USER ROLE: {role}
AI ROLE: {ai_role}
SCENARIO: {scenario}

CONVERSATION:
{conversation}

Analyze what QUESTIONS the user SHOULD HAVE ASKED but DIDN'T.

For {scenario_type} scenarios, strong performers ask:
- Open-ended discovery questions (to understand needs)
- Probing questions (to uncover root causes)
- Clarifying questions (to remove ambiguity)
- Vision/Outcome questions (to align on goals)
- Closing/Action questions (to drive commitment)

Return VALID JSON with this EXACT structure:
{{
  "questions_asked_count": {questions_asked},
  "questions_missed": [
    {{
      "question": "The exact question they should have asked",
      "category": "Discovery | Probing | Clarifying | Vision | Closing",
      "timing": "Early | Mid | Late",
      "why_important": "Why this question matters for success",
      "when_to_ask": "At what point in the conversation (e.g., Turn 2, when X happened)",
      "impact_if_asked": "What outcome this question would have enabled"
    }}
  ],
  "question_quality_score": "X/10",
  "question_quality_feedback": "Overall assessment of their questioning approach",
  "questioning_improvement_tip": "Specific advice to ask better questions"
}}

Identify 5-8 HIGH-IMPACT questions they missed. Be SPECIFIC about WHEN and WHY.
Categorize each question and specify optimal timing in the conversation.
"""
    
    try:
        parser = JsonOutputParser()
        prompt_template = PromptTemplate(
            template="{prompt}\n\n{format_instructions}",
            input_variables=["prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt_template | llm | parser
        result = chain.invoke({"prompt": prompt})
        print(" [SUCCESS] Question analysis completed")
        return result
        
    except Exception as e:
        print(f" [ERROR] Question analysis failed: {e}")
        return {
            "questions_asked_count": questions_asked,
            "questions_missed": [],
            "question_quality_score": "N/A",
            "question_quality_feedback": "Analysis unavailable",
            "questioning_improvement_tip": "Ask more open-ended questions to discover deeper needs"
        }


def analyze_full_report_data(transcript, role, ai_role, scenario, framework=None, mode="coaching", scenario_type=None, ai_character="alex"):
    """
    Generate report data using SCENARIO-SPECIFIC structures.
    """
    # Auto-detect scenario type if not provided
    if not scenario_type:
        scenario_type = detect_scenario_type(scenario, ai_role, role)
    
    # Detect granular user role
    user_context = detect_user_role_context(role, ai_role)
    print(f"[INFO] User Context Detected: {user_context} (Scenario: {scenario_type})")

    # CHARACTER SCHEMA OVERRIDE REMOVED - Relying on scenario_type detection
    # if ai_character == 'sarah': ...
    
    # Extract only user messages for focused analysis
    user_msgs = [t for t in transcript if t['role'] == 'user']
    
    # Base metadata
    meta = {
        "scenario_id": scenario_type,
        "outcome_status": "Completed", 
        "overall_grade": "N/A",
        "summary": "Session analysis.",
        "scenario_type": scenario_type,
        "scenario": scenario  # Pass full scenario text to frontend
    }

    if not user_msgs:
        meta["outcome_status"] = "Not Started"
        meta["summary"] = "Session started but no interaction recorded."
        return { "meta": meta, "type": scenario_type }

    # Determine Report Mode based on User Role Context
    # RULE: If User is PERFORMER -> EVALUATION (Scored)
    # RULE: If User is EVALUATOR -> MENTORSHIP (Unscored)
    
    is_user_performer = False
    if scenario_type == "coaching":
        # User is Staff (Performer) vs Manager (Evaluator)
        if user_context == "staff": is_user_performer = True
    elif scenario_type == "negotiation":
        # User is Seller (Performer) vs Buyer (Evaluator)
        if user_context == "seller": is_user_performer = True
    
    # -------------------------------------------------------------
    # BUILD SPECIFIC PROMPTS BASED ON SCENARIO TYPE & ROLE
    # -------------------------------------------------------------
    
    unified_instruction = ""
    
    if scenario_type == "coaching":
        if is_user_performer: # User is STAFF
            unified_instruction = """
### SCENARIO: COACHABILITY ASSESSMENT (USER IS STAFF)
**GOAL**: Evaluate the user's openness to feedback and their ability to pivot behavior.
**MODE**: EVALUATION (Growth-Oriented Assessment).
**CRITICAL RULE**: IF YOU GIVE A LOW SCORE OR IDENTIFY A WEAKNESS, YOU MUST EXPLAIN "WHY" USING A DIRECT QUOTE.
**INSTRUCTIONS**:
1. **BEHAVIORAL ANALYSIS**: Focus on *micro-signals* (tone, pauses, word choice). Go deep into the psychological "why".
2. **SCORECARD**: Every score MUST have a "Proof" (quote) and a "Growth Tactic" (specific alternative action).
3. **JUSTIFY WEAKNESS**: If the user is "weak", do not just say they were defensive. Say: "You were defensive because when I asked X, you replied 'It's not my fault' (Turn 4). This destroys trust."
4. **ACTIONABLE TIPS**: Do NOT give generic advice (e.g., "Listen better"). Give specific DRILLS (e.g., "Use the 'Looping' technique: Repeat back the last 3 words...").
5. **TONE**: Constructive, Encouraging, but Direct. "Tough Love".

**OUTPUT JSON STRUCTURE**:
{
  "meta": { "scenario_id": "coaching", "outcome_status": "Success/Partial/Failure", "overall_grade": "X/10", "summary": "A high-level executive summary of their coachability profile." },
  "type": "coaching",
  "eq_analysis": [
    {
      "nuance": "Current Emotion (e.g., Defensive)",
      "observation": "Quote proving this emotion.",
      "suggestion": "Recommended emotional shift."
    }
  ],
  "behaviour_analysis": [
    {
      "behavior": "Name of the behavior (e.g., Defensive Deflection, Active Evaluation)",
      "quote": "The exact verbatim line from the transcript demonstrates this.",
      "insight": "Detailed psychological breakdown of why this behavior undermines or aids growth.",
      "impact": "Positive/Negative",
      "improved_approach": "The exact phrase they SHOULD have used to build trust."
    }
  ],
  "detailed_analysis": [
    {"topic": "Psychological Safety Creation", "analysis": "Did they make it safe to give feedback?"},
    {"topic": "Ownership & Accountability", "analysis": "Did they own the problem or blame external factors?"}
  ],
  "scorecard": [
    { 
      "dimension": "Professionalism", 
      "score": "X/10", 
      "reasoning": "PROVE WHY this score was given. Start with 'You scored X/10 because...' then cite specific evidence. If they made a mistake, explicitly state WHAT THEY SAID WRONG and WHY it was problematic.",
      "quote": "The EXACT verbatim line from the transcript that demonstrates this dimension (good or bad).",
      "suggestion": "Start with 'Instead of saying [what they said], you should say: [exact alternative phrase].' Be specific and actionable. If they did well, suggest how to refine it further.",
      "alternative_questions": [{"question": "I appreciate that perspective...", "rationale": "Validates before disagreeing"}]
    },
    { 
      "dimension": "Ownership", 
      "score": "X/10", 
      "reasoning": "PROVE WHY this score was given. If they deflected responsibility, say EXACTLY how they deflected it (quote required). If they owned it, show the evidence.",
      "quote": "The line where they accepted or dodged responsibility. Must be verbatim.",
      "suggestion": "Tell them EXACTLY what to say next time. Example: 'Instead of deflecting with excuse X, say: That's on me, here's my plan...' Be ultra-specific.",
      "alternative_questions": [{"question": "That's on me, here is my plan...", "rationale": "Total accountability"}]
    },
    { 
      "dimension": "Active Listening", 
      "score": "X/10", 
      "reasoning": "PROVE WHY this score was given. If they interrupted or failed to acknowledge, provide the exact moment. If they listened well, prove it with evidence.",
      "quote": "Evidence of listening (e.g., a reflective statement) or interrupting (the exact interruption).",
      "suggestion": "Prescribe the EXACT listening technique. Example: 'When they said X, instead of jumping to Y, you should have said: So what you're seeing is... [mirror back their concern].'",
      "alternative_questions": [{"question": "So what you're seeing is...", "rationale": "Reflective listening loop"}]
    },
    { 
      "dimension": "Solution Focus", 
      "score": "X/10", 
      "reasoning": "PROVE WHY this score was given. If their solution was vague or reactive, show the exact quote and explain what was missing. If strong, prove it.",
      "quote": "The specific proposal they made (verbatim).",
      "suggestion": "Give them the EXACT alternative approach. Example: 'Instead of proposing X, frame it as: What does success look like to you in this area?' Teach them a better script.",
      "alternative_questions": [{"question": "What does success look like to you?", "rationale": "Collaborative problem solving"}]
    }
  ],
  "strengths": ["Specific, high-impact strength observed..."],
  "missed_opportunities": ["Specific moment where they could have won..."],
  "actionable_tips": ["Tactic 1 (Do this tomorrow)...", "Tactic 2 (Mindset shift)..."]
}
"""
        else: # User is MANAGER (Evaluator -> Mentorship)
            unified_instruction = """
### SCENARIO: LEADERSHIP MENTORSHIP (USER IS MANAGER)
**GOAL**: specific guidance on improving the user's coaching style.
**MODE**: MENTORSHIP (No Scorecard).
**INSTRUCTIONS**:
1. **REFLECTIVE QUESTIONS**: Include a brief "Reference Answer" or "Key Insight" in parentheses for each question to guide self-reflection.
2. **PRACTICE PLAN**: Must be highly actionable, valid, and specific (e.g., "Use the 5-Whys technique...").

**OUTPUT JSON STRUCTURE**:
{
  "meta": { "scenario_id": "learning", "outcome_status": "Completed", "overall_grade": "N/A", "summary": "..." },
  "type": "learning",
  "eq_analysis": [
    {
      "nuance": "Detected Tone",
      "observation": "Evidence quote.",
      "suggestion": "How to shift tone."
    }
  ],
  "behaviour_analysis": [
    {
      "behavior": "Name of the behavior (e.g., Empathy)",
      "quote": "The exact sentence the user said.",
      "insight": "Why this mattered.",
      "impact": "Positive/Negative",
      "improved_approach": "Alternative phrasing."
    }
  ],
  "detailed_analysis": [
    {"topic": "Coaching Style", "analysis": "Analysis content..."},
    {"topic": "Empathy & Connection", "analysis": "Analysis content..."}
  ],
  "key_insights": ["Insight 1...", "Insight 2..."],
  "reflective_questions": ["Question 1? (Reference: Key principle...)", "Question 2? (Insight: Look for...)"],
  "growth_outcome": "Vision of the user as a better leader...",
  "practice_plan": ["Actionable Drill 1...", "Specific Technique 2..."]
}
"""
            # Override semantic type for Report.tsx to render Learning View
            scenario_type = "learning" 

    elif scenario_type == "negotiation": 
        if is_user_performer: # User is SELLER
            unified_instruction = """
### SCENARIO: SALES PERFORMANCE ASSESSMENT (USER IS SELLER)
**GOAL**: Generate a High-Performance Sales Audit.
**MODE**: EVALUATION (Commercial Excellence).
**CRITICAL RULE**: NO GENERIC FEEDBACK. PROVE YOUR CLAIMS WITH EVIDENCE.
**INSTRUCTIONS**:
1. **REVENUE FOCUS**: Evaluate every behavior based on its likelihood to CLOSE THE DEAL or KILL THE DEAL.
2. **EVIDENCE**: You cannot give a score without citing the exact quote that justifies it.
3. **JUSTIFY THE LOSS**: If they lost the deal or scored low, explain precisely WHERE they lost it. (e.g., "You lost the deal in Turn 3 when you ignored the budget objection.")
4. **GROWTH**: If a score is low, you must provide the *exact script* or *tactic* that would have worked.
5. **RECOMMENDATIONS**: Must be commercially focused and specific. "Ask open questions" is bad. "Ask 'How does this impact your Q3 goals?'" is good.

**OUTPUT JSON STRUCTURE**:
{
  "meta": { "scenario_id": "sales", "outcome_status": "Closed/Negotiating/Lost", "overall_grade": "X/10", "summary": "Executive summary of commercial impact." },
  "type": "sales",
  "eq_analysis": [
    {
      "nuance": "Buyer Sentiment / User EQ",
      "observation": "Evidence of emotional read.",
      "suggestion": "How to better align with buyer emotion."
    }
  ],
  "behaviour_analysis": [
    {
      "behavior": "Name of the behavior (e.g., Feature Dumping, Strategic Pausing)",
      "quote": "The exact sentence used.",
      "insight": "Why this behavior increased or decreased deal velocity.",
      "impact": "Positive/Negative",
      "improved_approach": "The winning line they should have used."
    }
  ],
  "suggested_questions": [
    "A specific, commercial question the user SHOULD HAVE ASKED.",
    "A high-impact discovery question they missed."
  ],
  "detailed_analysis": [
    {"topic": "Sales Methodology", "analysis": "Did they follow a structured path (e.g., MEDDIC/SPIN)?"},
    {"topic": "Value Proposition", "analysis": "Did they sell the 'Why' or just the 'What'?"}
  ],
  "scorecard": [
    { 
      "dimension": "Rapport Building", 
      "score": "X/10", 
      "reasoning": "PROVE WHY this score was given. Did they build genuine rapport or come across as transactional? Show the exact evidence.",
      "quote": "The exact rapport attempt (or lack thereof). Must be verbatim from transcript.",
      "suggestion": "Tell them EXACTLY what to say instead. Example: 'Instead of immediately pitching, say: I noticed you mentioned X on LinkedIn, how's that project going?' Be ultra-specific.",
      "alternative_questions": [{"question": "I noticed you... How is that impacting X?", "rationale": "Connects observation to business pain"}]
    },
    { 
      "dimension": "Needs Discovery", 
      "score": "X/10", 
      "reasoning": "PROVE WHY this score was given. Did they ask deep discovery questions or skip straight to pitching? Cite the exact moment they succeeded or failed.",
      "quote": "The critical discovery question asked or missed. Must be word-for-word from the transcript.",
      "suggestion": "Give them the EXACT question they should have asked. Example: 'Instead of asking about budget first, say: What happens if you don't solve this problem in the next 6 months?' Teach SPIN.",
      "alternative_questions": [{"question": "What happens if you don't solve this?", "rationale": "Implication question (SPIN)"}]
    },
    { 
      "dimension": "Value Articulation", 
      "score": "X/10", 
      "reasoning": "PROVE WHY this score was given. Did they articulate value in terms of customer outcomes or just list features? Show the evidence.",
      "quote": "The value pitch. Exact words from transcript.",
      "suggestion": "Prescribe the EXACT value statement. Example: 'Instead of saying our product has X feature, say: Based on what you told me about Y challenge, here's how Z capability directly solves that.' Connect benefit to pain.",
      "alternative_questions": [{"question": "Based on what you said about X, here is how Y helps...", "rationale": "Direct solution mapping"}]
    },
    { 
      "dimension": "Objection Handling", 
      "score": "X/10", 
      "reasoning": "PROVE WHY this score was given. Did they defend, deflect, or acknowledge the objection? Cite the exact exchange.",
      "quote": "Their response to the pushback. Word-for-word.",
      "suggestion": "Give them the EXACT objection handling script. Example: 'Instead of saying It's worth it, use Feel-Felt-Found: I understand how you feel, others felt the same way until they found that X resulted in Y ROI.' Be specific.",
      "alternative_questions": [{"question": "It sounds like cost is a major factor...", "rationale": "Labeling the objection"}]
    },
    { 
      "dimension": "Closing", 
      "score": "X/10", 
      "reasoning": "PROVE WHY this score was given. Did they ask for the business or leave it vague? Show the exact closing attempt (or lack thereof).",
      "quote": "The closing line. Must be verbatim.",
      "suggestion": "Give them the EXACT close. Example: 'Instead of saying Let me know what you think, try an assumptive close: When should we schedule the kickoff call for next week?' Be direct and specific.",
      "alternative_questions": [{"question": "Does it make sense to start the paperwork?", "rationale": "Assumptive Close"}]
    }
  ],
  "sales_recommendations": ["Commercial Insight 1...", "Commercial Insight 2..."]
}
"""
        else: # User is BUYER (Evaluator -> Mentorship)
            unified_instruction = """
### SCENARIO: BUYER STRATEGY MENTORSHIP (USER IS BUYER)
**GOAL**: specific guidance on how to negotiate better deals as a buyer.
**MODE**: MENTORSHIP (No Scorecard).
**INSTRUCTIONS**:
1. **REFLECTIVE QUESTIONS**: Provide a "Reference Insight" for each question.
2. **PRACTICE PLAN**: Detailed, realistic tactics for a buyer.

**OUTPUT JSON STRUCTURE**:
{
  "meta": { "scenario_id": "learning", "outcome_status": "Completed", "overall_grade": "N/A", "summary": "..." },
  "type": "learning",
  "eq_analysis": [
    {
      "nuance": "Negotiation stance",
      "observation": "Evidence.",
      "suggestion": "Adjustment."
    }
  ],
  "behaviour_analysis": [
    {
      "behavior": "Behavior Name",
      "quote": "Evidence quote.",
      "insight": "Analysis.",
      "impact": "Positive/Negative",
      "improved_approach": "Suggestion."
    }
  ],
  "detailed_analysis": [
    {"topic": "Negotiation Power", "analysis": "Analysis content..."},
    {"topic": "Leverage Usage", "analysis": "Analysis content..."}
  ],
  "key_insights": ["Insight 1...", "Insight 2..."],
  "reflective_questions": ["Question? (Insight: Power leverage comes from...)", "Question? (Reference: BATNA...)"],
  "growth_outcome": "Vision of the user as a stronger negotiator...",
  "practice_plan": ["Tactic: Walk away when...", "Drill: Ask for 3 concessions..."]
}
"""
            # Override semantic type for Report.tsx to render Learning View
            scenario_type = "learning"

    elif scenario_type == "mentorship":
        unified_instruction = """
### SCENARIO: EXPERT MENTORSHIP (REVERSE ROLE)
**GOAL**: Generate a "Key Takeaways" summary for the user who observed the AI.
**MODE**: MENTORSHIP (Observation).
**INSTRUCTIONS**:
1. **NO SCORES**: Do not grade the user (or the AI).
2. **FOCUS**: Explain the *strategy* behind what the AI did.
3. **SUGGESTED QUESTIONS**: List questions the user *could have asked* to learn more.

**OUTPUT JSON STRUCTURE**:
{
  "meta": { "scenario_id": "mentorship", "outcome_status": "Completed", "overall_grade": "N/A", "summary": "Brief summary of the lesson demonstrated." },
  "type": "mentorship",
  "eq_analysis": [
    {
      "nuance": "Expert Emotional Control",
      "observation": "How the expert handled emotion.",
      "suggestion": "What you can learn from this."
    }
  ],
  "behaviour_analysis": [
    {
      "behavior": "Key Technique Demonstrated",
      "quote": "The exact line the AI (Expert) used.",
      "insight": "Why this technique works.",
      "impact": "Positive",
      "improved_approach": "N/A"
    }
  ],
  "suggested_questions": [
    "Question you could ask to deeper understand the strategy...",
    "Question about alternative approaches..."
  ],
  "detailed_analysis": [
    {"topic": "Strategic Intent", "analysis": "Why the AI chose this path..."},
    {"topic": "Key Principles", "analysis": "The core rules applied here..."}
  ],
  "key_insights": ["Principle 1...", "Principle 2..."],
  "reflective_questions": ["How would you have handled X differently?", "What did you notice about Y?"],
  "growth_outcome": "Understanding of expert-level execution.",
  "practice_plan": ["Try this technique in your next real conversation..."]
}
"""

    elif scenario_type == "reflection" or scenario_type == "learning":
        unified_instruction = """
### SCENARIO: PERSONAL LEARNING PLAN
**GOAL**: Generate a Developmental Learning Plan.
**MODE**: MENTORSHIP (Supportive, Qualitative Only).
**INSTRUCTIONS**:
1. **REFLECTIVE QUESTIONS**: Include a self-reflection prompt and a reference insight.
2. **PRACTICE PLAN**: Valid, specific, and actionable exercises.

**OUTPUT JSON STRUCTURE**:
{
  "meta": { "scenario_id": "learning", "outcome_status": "Completed", "overall_grade": "N/A", "summary": "..." },
  "type": "learning",
  "eq_analysis": [
    {
      "nuance": "Self-Reflection Tone",
      "observation": "Evidence.",
      "suggestion": "Adjustment."
    }
  ],
  "behaviour_analysis": [
    {
      "behavior": "Behavior Name",
      "quote": "Evidence quote.",
      "insight": "Analysis.",
      "impact": "Positive/Negative",
      "improved_approach": "Suggestion."
    }
  ],
  "detailed_analysis": [
    {"topic": "Conversation Flow", "analysis": "Analysis content..."},
    {"topic": "Key Patterns", "analysis": "Analysis content..."}
  ],
  "key_insights": ["Pattern observed...", "Strength present..."],
  "reflective_questions": ["Question? (Insight: ...)", "Question? (Ref: ...)"],
  "practice_plan": ["Experiment 1: Try...", "Micro-habit: When X, do Y..."],
  "growth_outcome": "Vision of success..."
}
"""
    else: # Custom
        unified_instruction = """
### CUSTOM SCENARIO / ROLE PLAY
**GOAL**: Generate an Adaptive Feedback Report.
**OUTPUT JSON STRUCTURE**:
{
  "meta": { "scenario_id": "custom", "outcome_status": "Success/Partial", "overall_grade": "N/A", "summary": "..." },
  "type": "custom",
  "eq_analysis": [
    {
      "nuance": "Detected Emotion",
      "observation": "Evidence.",
      "suggestion": "Advice."
    }
  ],
  "behaviour_analysis": [
    {
      "behavior": "Behavior Name",
      "quote": "Evidence quote.",
      "insight": "Analysis.",
      "impact": "Positive/Negative",
      "improved_approach": "Suggestion."
    }
  ],
  "detailed_analysis": [
    {"topic": "Performance Overview", "analysis": "Analysis content..."},
    {"topic": "Specific Observations", "analysis": "Analysis content..."}
  ],
  "strengths_observed": ["..."],
  "development_opportunities": ["..."],
  "guidance": {
    "continue": ["..."],
    "adjust": ["..."],
    "try_next": ["..."]
  }
}
"""

    # ANALYST PERSONA (Layered on top of Scenario Logic)
    # The 'Content' (what is measured) is determined by the Scenario (above).
    # The 'Voice' (how it is written) is determined by the Character (below).
    
    analyst_persona = ""
    if ai_character == "sarah":
        analyst_persona = """
    ### ANALYST STYLE: COACH SARAH (MENTOR)
    - **Tone**: Warm, encouraging, high-EQ, "Sandwich Method" (Praise-Critique-Praise).
    - **Focus**: Psychological safety, "growth mindset", and emotional intelligence.
    - **Detail Level**: EXTREMELY HIGH. Write 2-3 distinct topic sections in `detailed_analysis`. Go deep into the "why" behind the user's choices.
    - **Signature**: Use phrases like "I just loved how you...", "Consider trying...", "A small tweak could be...".
    - **Evidence Requirement**: You MUST quote the user's exact words to support every insight. No generic praise.
    - **Tactical Advice**: Provide specific "Micro-Scripts" (e.g., "Next time say: 'I hear you...'") instead of general advice.
    """
    else: # Default to Alex
        analyst_persona = """
    ### ANALYST STYLE: COACH ALEX (EVALUATOR)
    - **Tone**: Professional, direct, analytical, "Bottom Line Up Front".
    - **Focus**: Efficiency, clear outcomes, negotiation leverage, and rapid improvement.
    - **Detail Level**: EXTREMELY HIGH. Audit the conversation mechanism by mechanism.
    - **Signature**: Use phrases like "The metrics show...", "You missed an opportunity to...", "To optimize, you must...".
    - **Evidence Requirement**: Every score or critique MUST be backed by a timestamped quote from the transcript.
    - **Tactical Advice**: Provide "High-Impact Power Moves" or specific phrasing adjustments. No fluff.
    """

    # Unified System Prompt
    system_prompt = (
        f"### SYSTEM ROLE\\n"
        f"You are {ai_character.title() if ai_character else 'The AI'}, an expert Soft Skills Coach. You just facilitated a simulation where you played the role of '{ai_role}' and the user played '{role}'. now, analyze the performance based on your direct interaction.\\n"
        f"Context: {scenario}\\n"
        f"User Role: {role} | AI Role: {ai_role}\\n"
        f"{analyst_persona}\\n"
        f"{unified_instruction}\\n"
        f"### GENERAL RULES\\n"
        "2. **JUSTIFICATION**: Do not just say 'Good job'. Explain 'You scored 8/10 because you asked X question at the start...'.\\n"
        "3. **DEPTH**: Avoid surface-level observations. Analyze subtext, tone, and strategy.\\n"
        "4. **EQ & NUANCE**: You MUST include the 'eq_analysis' section. Identify the user's *current emotion/tone* (nuance) and provide a specific *emotional adjustment* (suggestion) to improve.\\n"
        "5. OUTPUT MUST BE VALID JSON ONLY.\\n"
    )

    try:
        # Create conversation text for analysis
        full_conversation = "\\n".join([f"{'USER' if t['role'] == 'user' else 'ASSISTANT'}: {t['content']}" for t in transcript])
        
        # Setup LangChain Parser
        parser = JsonOutputParser()
        
        # Create Prompt Template
        prompt = PromptTemplate(
            template="{system_prompt}\n\n{format_instructions}\n\n### FULL CONVERSATION\n{conversation}",
            input_variables=["system_prompt", "conversation"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Create Chain WITHOUT parser initially - we'll handle JSON parsing manually
        chain_raw = prompt | llm
        
        # Invoke Chain - MAIN REPORT (Core scorecard and behavior)
        print(f" [INFO] Starting PARALLEL report generation (3 LLM calls)...", flush=True)
        
        # ===== PARALLEL EXECUTION FOR SPEED =====
        # Run 3 analysis functions in parallel:
        # 1. Main report (scorecard, behavior analysis)
        # 2. Character assessment (NEW)
        # 3. Question analysis (NEW)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all 3 tasks  
            future_main = executor.submit(
                lambda: chain_raw.invoke({
                    "system_prompt": system_prompt,
                    "conversation": full_conversation
                })
            )
            
            future_character = executor.submit(
                analyze_character_traits,
                transcript, role, ai_role, scenario, scenario_type
            )
            
            future_questions = executor.submit(
                analyze_questions_missed,
                transcript, role, ai_role, scenario, scenario_type
            )
            
            # Wait for all to complete
            t1 = dt.datetime.now()
            raw_response = future_main.result()
            t2 = dt.datetime.now()
            print(f" [PERF] Main Report Generation took: {(t2-t1).total_seconds():.2f}s")
            
            character_analysis = future_character.result()
            t3 = dt.datetime.now()
            print(f" [PERF] Character Analysis took: {(t3-t2).total_seconds():.2f}s (relative to main)")
            
            question_analysis = future_questions.result()
            t4 = dt.datetime.now()
            print(f" [PERF] Question Analysis took: {(t4-t3).total_seconds():.2f}s (relative to char)")
        
        print(f" [SUCCESS] All analyses completed in parallel!", flush=True)
        
        # === ROBUST JSON PARSING WITH CLEANUP ===
        try:
            # Extract text content from LLM response
            if hasattr(raw_response, 'content'):
                json_text = raw_response.content
            else:
                json_text = str(raw_response)
            
            # Log first 500 chars for debugging
            print(f" [DEBUG] Raw LLM response (first 500 chars): {json_text[:500]}", flush=True)
            
            # Clean up common JSON formatting issues:
            # 1. Remove markdown code fences (```json ... ```)
            json_text = re.sub(r'^```(?:json)?\\s*', '', json_text.strip())
            json_text = re.sub(r'```\\s*$', '', json_text.strip())
            
            # 2. Fix escaped quotes at start/end of string values: "key": \\"value\\" -> "key": "value"
            # This is the main issue causing the parse error
            json_text = re.sub(r':\\s*\\\\"([^"]*)\\\\"', r': "\\1"', json_text)
            
            # 3. Parse the cleaned JSON
            data = json.loads(json_text)
            print(f" [SUCCESS] JSON parsed successfully after cleanup", flush=True)
            
        except json.JSONDecodeError as je:
            print(f" [ERROR] JSON Parse Error after cleanup: {je}", flush=True)
            print(f" [ERROR] Problematic JSON (first 1000 chars): {json_text[:1000]}", flush=True)
            
            # Last resort: try LangChain's parser
            try:
                data = parser.parse(json_text)
                print(f" [SUCCESS] LangChain parser succeeded as fallback", flush=True)
            except Exception as parser_error:
                print(f" [ERROR] LangChain parser also failed: {parser_error}", flush=True)
                # Re-raise original error with more context
                raise je
        
        # Ensure meta exists
        if 'meta' not in data: data['meta'] = {}
        data['meta']['scenario_type'] = scenario_type
        # Add type if missing
        if 'type' not in data: data['type'] = scenario_type

        # ==== MERGE NEW ANALYSES INTO REPORT ====
        if character_analysis:
            data['character_assessment'] = character_analysis
        
        if question_analysis:
            data['question_analysis'] = question_analysis

        return data
        
    except Exception as e:
        print(f"JSON Parse Error: {e}")
        # Note: 'response' variable doesn't exist in this version, removing that debug line
        return {
            "meta": {
                "scenario_id": scenario_type,
                "outcome_status": "Failure", 
                "overall_grade": "F",
                "summary": "Error generating report. Please try again.",
                "scenario_type": scenario_type
            },
            "type": scenario_type
        }


class DashboardPDF(FPDF):
    def cell(self, w, h=0, txt='', border=0, ln=0, align='', fill=False, link=''):
        # Auto-sanitize all text going into cells
        txt = sanitize_text(txt) if txt else ''
        super().cell(w, h, txt, border, ln, align, fill, link)
    
    def multi_cell(self, w, h, txt, border=0, align='J', fill=False):
        # Auto-sanitize all text going into multi_cells  
        txt = sanitize_text(txt) if txt else ''
        # Use provided align, default to Justified if not specified for long text
        super().multi_cell(w, h, txt, border, align, fill)
    
    def footer(self):
        self.set_y(-15)
        # Add subtle line separator
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_y(-12)
        # Page number on left
        self.set_font('Arial', '', 8)
        self.set_text_color(128, 128, 128)
        super().cell(30, 10, f'Page {self.page_no()}', 0, 0, 'L')
        # Branding in center
        self.set_font('Arial', 'I', 8)
        super().cell(140, 10, 'Generated by CoAct.AI Coaching Engine', 0, 0, 'C')
        # Timestamp on right
        self.set_font('Arial', '', 7)
        super().cell(0, 10, dt.datetime.now().strftime('%Y-%m-%d'), 0, 0, 'R')

    def set_scenario_type(self, scenario_type):
        self.scenario_type = scenario_type

    def get_title(self, section_key):
        stype = getattr(self, 'scenario_type', 'custom')
        return SCENARIO_TITLES.get(stype, SCENARIO_TITLES['custom']).get(section_key, section_key.upper())

    def linear_gradient(self, x, y, w, h, c1, c2, orientation='H'):
        self.set_line_width(0)
        if orientation == 'H':
            for i in range(int(w)):
                r = c1[0] + (c2[0] - c1[0]) * (i / w)
                g = c1[1] + (c2[1] - c1[1]) * (i / w)
                b = c1[2] + (c2[2] - c1[2]) * (i / w)
                self.set_fill_color(int(r), int(g), int(b))
                self.rect(x + i, y, 1, h, 'F')
        else:
            for i in range(int(h)):
                r = c1[0] + (c2[0] - c1[0]) * (i / h)
                g = c1[1] + (c2[1] - c1[1]) * (i / h)
                b = c1[2] + (c2[2] - c1[2]) * (i / h)
                self.set_fill_color(int(r), int(g), int(b))
                self.rect(x, y + i, w, 1, 'F')

    def set_user_name(self, name):
        self.user_name = sanitize_text(name)

    def set_character(self, character):
        self.ai_character = sanitize_text(character).capitalize()

    def header(self):
        if self.page_no() == 1:
            # Premium gradient header
            self.linear_gradient(0, 0, 210, 40, COLORS['header_grad_1'], COLORS['header_grad_2'], 'H')
            # Main title
            self.set_xy(10, 8)
            self.set_font('Arial', 'B', 24)
            self.set_text_color(255, 255, 255)
            super().cell(0, 10, 'COACT.AI', 0, 0, 'L')
            # Subtitle - Dynamic based on Coach
            self.set_xy(10, 22)
            self.set_font('Arial', '', 11)
            self.set_text_color(147, 197, 253)
            
            coach_name = getattr(self, 'ai_character', 'Alex')
            super().cell(0, 5, f'Performance Analysis by Coach {coach_name}', 0, 0, 'L')
            
            # Date on right
            self.set_xy(140, 10)
            self.set_font('Arial', '', 9)
            self.set_text_color(200, 220, 255)
            super().cell(50, 5, dt.datetime.now().strftime('%B %d, %Y'), 0, 0, 'R')
            
            # User Name Display
            if hasattr(self, 'user_name') and self.user_name:
                self.set_xy(140, 16)
                self.set_font('Arial', 'I', 9)
                super().cell(50, 5, f"Prepared for: {self.user_name}", 0, 0, 'R')

            # Avatar Image (Removed as per request)
            # if hasattr(self, 'ai_character'):
            #     char_name = self.ai_character.lower()
            #     img_path = f"{char_name}.png"
            #     if os.path.exists(img_path):
            #          self.image(img_path, x=188, y=8, w=15)

            self.ln(35)
        else:
            # Slim header for subsequent pages
            self.set_fill_color(*COLORS['header_grad_1'])
            self.rect(0, 0, 210, 14, 'F')
            self.set_xy(10, 4)
            self.set_font('Arial', 'B', 10)
            self.set_text_color(255, 255, 255)
            super().cell(100, 6, 'CoAct.AI Report', 0, 0, 'L')
            
            # Avatar Icon Scalling Small (Removed)
            # if hasattr(self, 'ai_character'):
            #     char_name = self.ai_character.lower()
            #     img_path = f"{char_name}.png"
            #     if os.path.exists(img_path):
            #         self.image(img_path, x=5, y=2, w=10)
                    
            # Page indicator
            self.set_font('Arial', '', 9)
            self.set_text_color(180, 200, 255)
            super().cell(0, 6, f'Page {self.page_no()}', 0, 0, 'R')
            self.ln(18)

    def set_context(self, role, ai_role, scenario):
        self.user_role = sanitize_text(role)
        self.ai_role = sanitize_text(ai_role)
        self.scenario_text = sanitize_text(scenario)

    def check_space(self, height):
        if self.get_y() + height > self.page_break_trigger:
            self.add_page()

    def draw_context_summary(self):
        """Draw a summary of the scenario context and roles."""
        if not hasattr(self, 'user_role'): return
        
        self.check_space(40)
        self.ln(5)
        
        # Section Header
        self.set_font('Arial', 'B', 10)
        self.set_text_color(71, 85, 105) # Slate 600
        self.cell(0, 6, "SCENARIO CONTEXT", 0, 1)
        
        # Grid Background
        self.set_fill_color(248, 250, 252) # Slate 50
        start_y = self.get_y()
        self.rect(10, start_y, 190, 35, 'F')
        
        # Draw Roles
        self.set_xy(15, start_y + 4)
        self.set_font('Arial', 'B', 9)
        self.set_text_color(*COLORS['primary'])
        self.cell(20, 5, "Your Role:", 0, 0)
        self.set_font('Arial', '', 9)
        self.set_text_color(*COLORS['text_main'])
        self.cell(60, 5, self.user_role, 0, 0)
        
        self.set_font('Arial', 'B', 9)
        self.set_text_color(*COLORS['primary'])
        self.cell(20, 5, "Partner:", 0, 0)
        self.set_font('Arial', '', 9)
        self.set_text_color(*COLORS['text_main'])
        self.cell(60, 5, self.ai_role, 0, 1)
        
        # Draw Scenario Description
        self.set_xy(15, start_y + 12)
        self.set_font('Arial', 'B', 9)
        self.set_text_color(*COLORS['primary'])
        self.cell(0, 5, "Situation:", 0, 1)
        
        self.set_x(15)
        self.set_font('Arial', '', 9)
        self.set_text_color(*COLORS['text_light'])
        # Truncate if too long to fit in box
        # Truncate if too long to fit in box
        text = self.scenario_text
        
        # Clean up text: Remove CONTEXT: prefix and AI BEHAVIOR section
        # The user wants JUST the situation, not the "CONTEXT" label or "AI BEHAVIOR" section.
        text = text.replace("CONTEXT:", "").replace("Situation:", "").strip()
        
        # Split by typical behavioral markers to ensure we only get the situation description
        for marker in ["AI BEHAVIOR:", "AI ROLE:", "USER ROLE:", "SCENARIO:"]:
            if marker in text:
                text = text.split(marker)[0].strip()
        
        if len(text) > 300: text = text[:297] + "..."
        self.multi_cell(180, 5, text)
        
        # Move cursor past the box
        self.set_y(start_y + 40)

    def draw_scoring_methodology(self):
        """Draw the scoring rubric/methodology section."""
        self.check_space(50)
        self.ln(5)
        
        self.draw_section_header("SCORING METHODOLOGY (THE 'WHY')", COLORS['secondary'])
        
        # Grid Background
        self.set_fill_color(248, 250, 252)
        start_y = self.get_y()
        self.rect(10, start_y, 190, 35, 'F')
        
        # Scoring Levels
        levels = [
            ("9-10 (Expert)", "Exceptional application of skills. Creates deep psychological safety, handles conflict with mastery, and drives clear outcomes."),
            ("7-8 (Proficient)", "Strong performance. Meets all core objectives effectively. Good empathy and strategy, with minor opportunities for refinement."),
            ("4-6 (Competent)", "Functional performance. Achieves basic goals but may miss subtle cues, sound robotic, or struggle with difficult objections."),
            ("1-3 (Needs Ops)", "Struggles with core skills. May be defensive, dismissive, or completely miss the objective. Immediate practice required.")
        ]
        
        current_y = start_y + 4
        
        for grade, desc in levels:
            self.set_xy(15, current_y)
            self.set_font('Arial', 'B', 8)
            
            # Color coding for levels
            if "9-10" in grade: self.set_text_color(*COLORS['success'])
            elif "7-8" in grade: self.set_text_color(*COLORS['success']) # Lighter green ideally, but success works
            elif "4-6" in grade: self.set_text_color(*COLORS['warning'])
            else: self.set_text_color(*COLORS['danger'])
            
            self.cell(25, 6, grade, 0, 0)
            
            self.set_font('Arial', '', 8)
            self.set_text_color(*COLORS['text_light'])
            self.cell(3, 6, "|", 0, 0)
            
            self.set_text_color(*COLORS['text_main'])
            self.multi_cell(150, 6, desc)
            current_y += 7

        self.set_y(start_y + 42)

    def draw_detailed_analysis(self, analysis_data):
        """Draw the detailed analysis section (Supporting string or list of topics)."""
        if not analysis_data: return
        
        # 1. Handle Legacy String Format (Backward Compatibility)
        if isinstance(analysis_data, str):
            self.check_space(60)
            self.ln(5)
            self.draw_section_header("DETAILED ANALYSIS", COLORS['secondary'])
            
            # Background Box
            self.set_fill_color(255, 255, 255)
            self.set_draw_color(226, 232, 240)
            self.rect(10, self.get_y(), 190, 45, 'DF') # Fixed height fallback
            
            # Icon
            self.set_xy(15, self.get_y() + 5)
            self.set_font('Arial', 'B', 14)
            self.set_text_color(*COLORS['secondary'])
            self.cell(10, 10, "i", 0, 0, 'C') 
            
            # Text
            self.set_xy(25, self.get_y() + 2)
            self.set_font('Arial', '', 10)
            self.set_text_color(*COLORS['text_main'])
            
            text = sanitize_text(analysis_data)
            if len(text) > 800: text = text[:797] + "..."
            self.multi_cell(170, 6, text)
            self.set_y(self.get_y() + 10)
            return

        # 2. Handle New List Format (Topic-Wise)
        # Expected: [{"topic": "Title", "analysis": "Content..."}, ...]
        if isinstance(analysis_data, list):
            self.check_space(60)
            self.ln(5)
            self.draw_section_header("DETAILED ANALYSIS", COLORS['secondary'])
            
            for item in analysis_data:
                topic = sanitize_text(item.get('topic', 'Topic'))
                content = sanitize_text(item.get('analysis', ''))
                
                # Estimate height
                self.set_font('Arial', '', 10)
                # content width is ~190mm
                # Arial 10 avg char width ~2.2mm? 190/2.2 ~ 86 chars
                # Using 70 as a safe divisor for wrapping
                num_lines = math.ceil(len(content) / 70) 
                height = (num_lines * 5) + 20 # +20 for padding/header buffer
                
                self.check_space(height)
                
                # Draw Topic Header
                self.set_font('Arial', 'B', 10)
                self.set_text_color(*COLORS['primary'])
                self.cell(0, 6, topic.upper(), 0, 1)
                
                # Draw Content
                self.set_font('Arial', '', 10)
                self.set_text_color(*COLORS['text_main'])
                self.multi_cell(190, 5, content)
                self.ln(4) # Spacing between topics

    def draw_dynamic_questions(self, questions):
        """Draw the dynamic follow-up questions section."""
        if not questions: return
        
        self.check_space(60)
        self.ln(5)
        
        self.draw_section_header("DEEP DIVE QUESTIONS", COLORS['accent'])
        
        # Grid Background - Purple/Indigo theme
        self.set_fill_color(248, 250, 252) # Very light slate
        start_y = self.get_y()
        # Estimate height based on questions
        height = 15 + (len(questions) * 12)
        self.rect(10, start_y, 190, height, 'F')
        
        current_y = start_y + 5
        
        for i, q in enumerate(questions):
            self.set_xy(15, current_y)
            self.set_font('Arial', 'B', 12)
            self.set_text_color(*COLORS['accent'])
            self.cell(10, 8, "?", 0, 0, 'C')
            
            self.set_font('Arial', 'I', 10) # Italic for questions
            self.set_text_color(*COLORS['text_main'])
            self.multi_cell(160, 6, sanitize_text(q))
            
            # Update Y for next question, assuming single line or double line
            # Simple heuristic: add fixed spacing
            current_y = self.get_y() + 4
            
        self.set_y(start_y + height + 5)

    def draw_behaviour_analysis(self, analysis_data):
        """Draw the detailed Behaviour Analysis section."""
        if not analysis_data: return

        self.check_space(80)
        self.ln(5)
        self.draw_section_header("BEHAVIOURAL ANALYSIS", COLORS['primary'])

        for item in analysis_data:
            behavior = sanitize_text(item.get('behavior', 'Behavior'))
            quote = sanitize_text(item.get('quote', ''))
            insight = sanitize_text(item.get('insight', ''))
            impact = sanitize_text(item.get('impact', 'Neutral'))
            improved = sanitize_text(item.get('improved_approach', ''))

            # Determine color based on impact
            impact_color = COLORS['secondary']
            if 'positive' in impact.lower(): impact_color = COLORS['success']
            elif 'negative' in impact.lower(): impact_color = COLORS['danger']
            
            # Estimate height
            height = 35 # Base height
            if quote: height += len(quote) // 65 * 5 + 10
            if insight: height += len(insight) // 65 * 5 + 10
            if improved: height += len(improved) // 65 * 5 + 15
            
            self.check_space(height + 10)
            
            start_y = self.get_y()
            
            # Left Bar (Impact Color)
            self.set_fill_color(*impact_color)
            self.rect(10, start_y, 2, height, 'F')
            
            # Background
            self.set_fill_color(248, 250, 252)
            self.rect(12, start_y, 188, height, 'F')
            
            current_y = start_y + 2
            
            # Header: Behavior + Impact
            self.set_xy(15, current_y)
            self.set_font('Arial', 'B', 10)
            self.set_text_color(*COLORS['text_main'])
            self.cell(100, 6, behavior.upper(), 0, 0)
            
            self.set_font('Arial', 'B', 8)
            self.set_text_color(*impact_color)
            self.cell(80, 6, impact.upper(), 0, 1, 'R')
            
            current_y += 8
            
            # Quote (Proof)
            if quote:
                self.set_xy(15, current_y)
                self.set_font('Arial', 'BI', 9) # Bold Italic
                self.set_text_color(80, 80, 80)
                self.multi_cell(180, 5, f'"{quote}"')
                current_y = self.get_y() + 2
                
            # Insight (Analysis)
            if insight:
                self.set_xy(15, current_y)
                self.set_font('Arial', '', 9)
                self.set_text_color(*COLORS['text_main'])
                self.multi_cell(180, 5, insight, align='J')
                current_y = self.get_y() + 4
                
            # Improved Approach (Actionable Advice)
            if improved:
                self.set_xy(15, current_y)
                self.set_font('Arial', 'B', 9)
                self.set_text_color(*COLORS['accent'])
                self.cell(40, 5, "TRY THIS INSTEAD:", 0, 1) # Force new line
                
                # Draw a highlight box for the correction
                correction_y = self.get_y()
                self.set_fill_color(240, 249, 255) # Light blue bg
                # Calculate height roughly
                lines = max(1, len(improved) // 65)
                box_h = lines * 5 + 4
                self.rect(15, correction_y, 180, box_h, 'F')
                
                self.set_xy(17, correction_y + 2) # Indent slightly inside box
                self.set_font('Arial', 'I', 9)
                self.set_text_color(*COLORS['text_main'])
                self.multi_cell(175, 5, improved, align='J')
                current_y = self.get_y() + 4

            # Bottom Spacer
            self.set_y(start_y + height + 4)

    def draw_eq_analysis(self, eq_data):
        """Draw the Emotional Intelligence & Nuance section."""
        if not eq_data: return

        self.check_space(60)
        self.ln(5)
        self.draw_section_header("EMOTIONAL INTELLIGENCE (EQ) & NUANCE", COLORS['section_eq'])

        for item in eq_data:
            # Handle both dict and string items
            if isinstance(item, dict):
                nuance = sanitize_text(item.get('nuance', 'User Observation'))
                observation = sanitize_text(item.get('observation', ''))
                suggestion = sanitize_text(item.get('suggestion', '')) # Retain suggestion from dict
            elif isinstance(item, str):
                nuance = 'User Observation'
                observation = sanitize_text(item)
                suggestion = '' # No suggestion for string items
            else:
                continue  # Skip invalid items
                
            # Estimate height
            height = 30
            if observation: height += len(observation) // 65 * 5 + 10 
            if suggestion: height += len(suggestion) // 65 * 5 + 15
            
            self.check_space(height + 10)
            start_y = self.get_y()
            
            # Background
            self.set_fill_color(253, 242, 248) # Pink 50 (to match section_eq)
            self.rect(10, start_y, 190, height, 'F')
            
            # Left Bar
            self.set_fill_color(*COLORS['section_eq'])
            self.rect(10, start_y, 2, height, 'F')
            
            current_y = start_y + 3
            
            # Draw nuance badge
            self.set_xy(15, current_y)
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(*COLORS['nuance_bg'])
            self.cell(0, 6, nuance.upper(), ln=True)
            
            # Draw observation (previously 'proof')
            if observation:
                self.set_font('Helvetica', '', 9)
                self.set_text_color(40, 40, 40)
                self.multi_cell(0, 5, f"Observation: {observation}")
                self.ln(2)
            
            # Draw suggestion
            if suggestion:
                self.set_text_color(100, 116, 139)
                self.cell(20, 5, "SUGGESTION:", 0, 0)
                
                self.set_font('Arial', '', 9)
                self.set_text_color(*COLORS['text_main'])
                self.multi_cell(160, 5, suggestion)
                current_y = self.get_y() + 4
            
            self.set_y(start_y + height + 3)

    def draw_section_header(self, title, color):
        self.ln(3)
        self.set_font('Arial', 'B', 11)
        self.set_text_color(*color)
        self.cell(0, 8, title, 0, 1)
        # Add colored underline
        self.set_draw_color(*color)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 50, self.get_y())
        self.set_line_width(0.2)
        self.ln(4)

    def draw_banner(self, meta, scenario_type="custom"):
        """Draw the summary banner at the top of the report."""
        summary = meta.get('summary', '')
        emotional_trajectory = meta.get('emotional_trajectory', '')
        session_quality = meta.get('session_quality', '')
        key_themes = meta.get('key_themes', [])
        overall_grade = meta.get('overall_grade', 'N/A')
        
        self.set_y(self.get_y() + 3)
        start_y = self.get_y()
        
        # Calculate banner height based on content
        base_height = 50
        if emotional_trajectory: base_height += 8
        if session_quality: base_height += 8
        if key_themes: base_height += 10
        banner_height = base_height
        
        # Main Card with shadow effect
        self.set_fill_color(245, 247, 250)  # Subtle shadow
        self.rect(12, start_y + 2, 190, banner_height, 'F')
        
        # Main Card Background
        self.set_fill_color(255, 255, 255)
        self.set_draw_color(226, 232, 240)
        self.rect(10, start_y, 190, banner_height, 'DF')
        
        # Scenario-type specific colors and labels
        scenario_colors = {
            "coaching": (59, 130, 246),    # Blue
            "negotiation": (16, 185, 129), # Green  
            "reflection": (139, 92, 246),  # Purple
            "custom": (245, 158, 11),      # Orange/Amber
            "leadership": (99, 102, 241),  # Indigo (Authority)
            "customer_service": (239, 68, 68) # Red (Urgency/Resolution)
        }
        
        # New Labels matching frontend
        scenario_labels = {
            "coaching": "COACHING EFFICACY",
            "negotiation": "NEGOTIATION POWER",
            "reflection": "LEARNING INSIGHTS",
            "custom": "GOAL ATTAINMENT",
            "leadership": "LEADERSHIP & STRATEGY",
            "customer_service": "CUSTOMER SERVICE"
        }
        
        accent_color = scenario_colors.get(scenario_type, scenario_colors["custom"])
        verd_label = scenario_labels.get(scenario_type, scenario_labels["custom"])
        
        # Accent bar on left - scenario-specific color
        self.set_fill_color(*accent_color)
        self.rect(10, start_y, 4, banner_height, 'F')
        
        # Scenario-type label with icon
        self.set_xy(18, start_y + 6)
        self.set_font('Arial', 'B', 10)
        self.set_text_color(71, 85, 105)  # Slate 600
        icon_map = {
            "coaching": "[C]", "negotiation": "[N]", "reflection": "[R]", "custom": "[*]",
            "leadership": "[L]", "customer_service": "[S]"
        }
        icon = icon_map.get(scenario_type, "[*]")
        self.cell(100, 5, f"{icon} {verd_label}", 0, 1)
        
        # Grade Display (Top Right)
        if scenario_type != 'reflection':
             self.set_xy(150, start_y + 6)
             self.set_font('Arial', 'B', 24)
             self.set_text_color(*COLORS['accent']) # Uses main accent
             # Determine color based on grade if possible, else default accent
             self.cell(40, 10, str(overall_grade), 0, 0, 'R')
        
        # Summary text
        self.set_xy(18, start_y + 15)
        self.set_font('Arial', '', 10)
        self.set_text_color(51, 65, 85)
        self.multi_cell(130, 5, sanitize_text(summary))
        
        # Metrics row with visual indicators
        current_y = start_y + 35
        
        if emotional_trajectory:
            self.set_xy(18, current_y)
            self.set_font('Arial', 'B', 8)
            self.set_text_color(99, 102, 241)  # Indigo
            self.cell(3, 4, ">", 0, 0)
            self.set_text_color(100, 116, 139)
            self.cell(38, 4, "EMOTIONAL ARC:", 0, 0)
            self.set_font('Arial', '', 9)
            self.set_text_color(51, 65, 85)
            self.cell(0, 4, sanitize_text(emotional_trajectory), 0, 1)
            current_y += 7
        
        if session_quality:
            self.set_xy(18, current_y)
            self.set_font('Arial', 'B', 8)
            self.set_text_color(16, 185, 129)  # Emerald
            self.cell(3, 4, ">", 0, 0)
            self.set_text_color(100, 116, 139)
            self.cell(38, 4, "SESSION QUALITY:", 0, 0)
            self.set_font('Arial', '', 9)
            self.set_text_color(51, 65, 85)
            self.cell(0, 4, sanitize_text(session_quality), 0, 1)
            current_y += 7
        
        # Key themes with pill-style display
        if key_themes:
            self.set_xy(18, current_y)
            self.set_font('Arial', 'B', 8)
            self.set_text_color(236, 72, 153)  # Pink
            self.cell(3, 4, ">", 0, 0)
            self.set_text_color(100, 116, 139)
            self.cell(38, 4, "KEY THEMES:", 0, 0)
            self.set_font('Arial', 'I', 9)
            self.set_text_color(71, 85, 105)
            themes_text = " | ".join([sanitize_text(str(theme)) for theme in key_themes[:3]])
            self.cell(0, 4, themes_text, 0, 1)
        
        self.set_y(start_y + banner_height + 8)
    
    def draw_executive_summary(self, exec_summary):
        """Draw the Executive Summary section - NEW unified section for all reports."""
        if not exec_summary:
            return
        
        self.check_space(80)
        self.ln(5)
        
        # Section header with gradient-like background
        self.set_fill_color(30, 41, 59)  # Slate 800
        self.rect(10, self.get_y(), 190, 12, 'F')
        self.set_xy(15, self.get_y() + 3)
        self.set_font('Arial', 'B', 11)
        self.set_text_color(255, 255, 255)
        self.cell(0, 6, self.get_title("exec_summary"), 0, 1)
        self.ln(3)
        
        # Performance Overview
        overview = exec_summary.get('performance_overview', '')
        if overview:
            self.set_font('Arial', '', 10)
            self.set_text_color(*COLORS['text_main'])
            self.multi_cell(0, 6, sanitize_text(overview))
            self.ln(6)
        
        # Two-column layout for strengths and growth areas
        start_y = self.get_y()
        
        # Key Strengths (left column)
        strengths = exec_summary.get('key_strengths', [])
        if strengths:
            self.set_fill_color(240, 253, 244)  # Green 50
            self.rect(10, start_y, 90, 45, 'F')
            self.set_xy(15, start_y + 5)
            self.set_font('Arial', 'B', 9)
            self.set_text_color(*COLORS['success'])
            self.cell(80, 5, "KEY STRENGTHS", 0, 1)
            
            self.set_font('Arial', '', 9)
            self.set_text_color(*COLORS['text_main'])
            for i, strength in enumerate(strengths[:3]):
                self.set_x(15)
                self.multi_cell(80, 5, f"+ {sanitize_text(strength)}")
        
        # Areas for Growth (right column)
        growth = exec_summary.get('areas_for_growth', [])
        if growth:
            self.set_fill_color(254, 249, 195)  # Yellow 100
            self.rect(105, start_y, 95, 45, 'F')
            self.set_xy(110, start_y + 5)
            self.set_font('Arial', 'B', 9)
            self.set_text_color(*COLORS['warning'])
            self.cell(85, 5, "AREAS FOR GROWTH", 0, 1)
            
            self.set_font('Arial', '', 9)
            self.set_text_color(*COLORS['text_main'])
            for i, area in enumerate(growth[:3]):
                self.set_x(110)
                self.multi_cell(85, 5, f"- {sanitize_text(area)}")
        
        self.set_y(start_y + 50)
        
        # Recommended Next Steps
        next_steps = exec_summary.get('recommended_next_steps', '')
        if next_steps:
            self.set_fill_color(248, 250, 252)  # Slate 50
            self.rect(10, self.get_y(), 190, 20, 'F')
            self.set_xy(15, self.get_y() + 5)
            self.set_font('Arial', 'B', 9)
            self.set_text_color(*COLORS['accent'])
            self.cell(40, 5, "NEXT STEPS:", 0, 0)
            self.set_font('Arial', '', 9)
            self.set_text_color(*COLORS['text_main'])
            self.multi_cell(145, 5, sanitize_text(next_steps))
            self.ln(5)
        
        self.ln(5)
    
    def draw_personalized_recommendations(self, recs):
        """Draw the unified personalized recommendations section."""
        if not recs:
            return
        
        self.check_space(70)
        self.ln(5)
        
        # Dark header block
        self.set_fill_color(30, 41, 59)  # Slate 800
        self.rect(10, self.get_y(), 190, 60, 'F')
        
        start_y = self.get_y()
        self.set_xy(15, start_y + 5)
        self.set_font('Arial', 'B', 11)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, self.get_title("recs"), 0, 1)
        
        # Immediate Actions
        actions = recs.get('immediate_actions', [])
        if actions:
            self.set_font('Arial', 'B', 9)
            self.set_text_color(147, 197, 253)  # Blue 300
            self.cell(50, 6, "IMMEDIATE ACTIONS:", 0, 0)
            self.set_font('Arial', '', 9)
            self.set_text_color(255, 255, 255)
            actions_text = ", ".join([sanitize_text(a) for a in actions[:3]])
            self.multi_cell(135, 6, actions_text)
        
        # Focus Areas
        focus = recs.get('focus_areas', [])
        if focus:
            self.set_font('Arial', 'B', 9)
            self.set_text_color(147, 197, 253)
            self.cell(50, 6, "FOCUS AREAS:", 0, 0)
            self.set_font('Arial', '', 9)
            self.set_text_color(255, 255, 255)
            focus_text = ", ".join([sanitize_text(f) for f in focus[:3]])
            self.multi_cell(135, 6, focus_text)
        
        # Reflection Prompts
        prompts = recs.get('reflection_prompts', [])
        if prompts:
            self.ln(2)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(203, 213, 225)  # Slate 300
            for prompt in prompts[:2]:
                self.set_x(15)
                self.cell(0, 4, f"? {sanitize_text(prompt)}", 0, 1)
        
        self.set_y(start_y + 65)

    # --- ASSESSMENT MODE DRAWING METHODS ---

    def draw_assessment_table(self, scores, show_scores=True):
        if not scores: return
        self.check_space(80)
        self.ln(5)
        
        self.draw_section_header(self.get_title("skills"), COLORS['primary'])

        # Widths
        w_dim = 45 if show_scores else 50
        w_score = 15
        w_interp = 65 if show_scores else 70
        w_tip = 65 if show_scores else 70
        
        # Header
        self.set_fill_color(241, 245, 249)
        self.set_font('Arial', 'B', 9)
        self.set_text_color(*COLORS['text_main'])
        self.cell(w_dim, 8, "DIMENSION", 1, 0, 'L', True)
        if show_scores:
            self.cell(w_score, 8, "SCORE", 1, 0, 'C', True)
        self.cell(w_interp, 8, "INTERPRETATION", 1, 0, 'L', True)
        self.cell(w_tip, 8, "IMPROVEMENT TIP", 1, 1, 'L', True)

        for item in scores:
            dim = sanitize_text(item.get('dimension', ''))
            score = item.get('score', 0)
            interp = sanitize_text(item.get('interpretation', ''))
            tip = sanitize_text(item.get('improvement_tip', ''))

            # Calculate row height based on content
            row_height = max(15, len(interp) // 40 * 5 + 10, len(tip) // 40 * 5 + 10)

            self.set_font('Arial', 'B', 9)
            self.set_text_color(*COLORS['text_main'])
            self.cell(w_dim, row_height, dim, 1, 0, 'L')
            
            if show_scores:
                # Score Color
                if score >= 8: self.set_text_color(*COLORS['success'])
                elif score >= 6: self.set_text_color(*COLORS['warning'])
                else: self.set_text_color(*COLORS['danger'])
                
                self.cell(w_score, row_height, f"{score}/10", 1, 0, 'C')
            
            # Interpretation
            self.set_text_color(*COLORS['text_main'])
            self.set_font('Arial', '', 8)
            current_x = self.get_x()
            current_y = self.get_y()
            self.multi_cell(w_interp, 7.5, interp, border=1, align='L')
            
            # Improvement tip
            self.set_xy(current_x + w_interp, current_y)
            self.set_text_color(*COLORS['accent'])
            self.multi_cell(w_tip, 7.5, tip, border=1, align='L')
            
            # Move to next row
            self.set_xy(10, current_y + row_height)

        self.ln(5)

    def draw_conversation_analytics(self, analytics):
        if not analytics: return
        self.check_space(40)
        
        self.draw_section_header(self.get_title("analytics"), COLORS['secondary'])
        
        # Create a 2x3 grid of metrics
        metrics = [
            ("Total Exchanges", analytics.get('total_exchanges', 'N/A')),
            ("Talk Time Balance", f"{analytics.get('user_talk_time_percentage', 0)}% User"),
            ("Question/Statement Ratio", analytics.get('question_to_statement_ratio', 'N/A')),
            ("Emotional Progression", analytics.get('emotional_tone_progression', 'N/A')),
            ("Framework Adherence", analytics.get('framework_adherence', 'N/A')),
        ]
        
        self.set_fill_color(248, 250, 252)
        self.rect(10, self.get_y(), 190, 35, 'F')
        
        start_y = self.get_y()
        for i, (label, value) in enumerate(metrics):
            x_pos = 15 + (i % 3) * 60
            y_pos = start_y + 5 + (i // 3) * 15
            
            self.set_xy(x_pos, y_pos)
            self.set_font('Arial', 'B', 8)
            self.set_text_color(*COLORS['text_light'])
            self.cell(55, 5, label, 0, 1)
            
            self.set_xy(x_pos, y_pos + 5)
            self.set_font('Arial', '', 9)
            self.set_text_color(*COLORS['text_main'])
            self.cell(55, 5, str(value), 0, 0)
        
        self.set_y(start_y + 40)

    def draw_learning_path(self, path):
        if not path: return
        self.check_space(60)
        
        self.draw_section_header("PERSONALIZED LEARNING PATH", COLORS['accent'])
        
        for item in path:
            skill = sanitize_text(item.get('skill', ''))
            priority = item.get('priority', 'Medium')
            timeline = sanitize_text(item.get('timeline', ''))
            
            # Priority color coding
            if priority == 'High': color = COLORS['danger']
            elif priority == 'Medium': color = COLORS['warning']
            else: color = COLORS['success']
            
            self.set_font('Arial', 'B', 10)
            self.set_text_color(*color)
            self.cell(100, 6, f"• {skill}", 0, 0)
            
            self.set_font('Arial', '', 9)
            self.set_text_color(*COLORS['text_light'])
            self.cell(0, 6, f"Priority: {priority} | {timeline}", 0, 1)
            self.ln(2)
        
        self.ln(5)

    def _extract_score_value(self, score_str):
        try:
            # Remove /10 or similar
            clean = str(score_str).split('/')[0].strip()
            return float(clean)
        except:
            return 0.0

    # --- SCENARIO SPECIFIC DRAWING METHODS ---

    def draw_scorecard(self, scorecard):
        """Draw a standard scorecard table with zebra striping."""
        if not scorecard: return
        self.check_space(60)
        self.ln(8) # Extra spacing
        
        # Draw Radar Chart First
        self.draw_section_header("SKILL VISUALIZATION", COLORS['secondary'])
        self.draw_radar_chart(scorecard)
        
        self.draw_section_header("PERFORMANCE SCORECARD", COLORS['primary'])
        
        # Table Header
        self.set_fill_color(30, 41, 59) # Dark header
        self.set_font('Arial', 'B', 9)
        self.set_text_color(255, 255, 255) # White text
        self.cell(50, 9, "DIMENSION", 0, 0, 'L', True)
        self.cell(20, 9, "SCORE", 0, 0, 'C', True)
        self.cell(120, 9, "OBSERVATION", 0, 1, 'L', True)
        
        # Rows
        for i, item in enumerate(scorecard):
            dim = sanitize_text(item.get('dimension', ''))
            score = str(item.get('score', 'N/A'))
            desc = sanitize_text(item.get('description', ''))
            
            row_height = max(14, len(desc) // 70 * 5 + 10)
            self.check_space(row_height)
            
            # Zebra striping
            if i % 2 == 0:
                self.set_fill_color(248, 250, 252) # Very light gray
            else:
                self.set_fill_color(255, 255, 255) # White
            
            self.set_font('Arial', 'B', 9)
            self.set_text_color(*COLORS['text_main'])
            
            # Draw row background
            x_start = self.get_x()
            y_start = self.get_y()
            self.cell(50, row_height, dim, 0, 0, 'L', True)
            
            # Score Color
            try:
                s_val = float(score.split('/')[0])
                if s_val >= 8: self.set_text_color(*COLORS['success'])
                elif s_val <= 5: self.set_text_color(*COLORS['danger'])
                else: self.set_text_color(*COLORS['warning'])
            except:
                self.set_text_color(*COLORS['text_main'])
                
            self.cell(20, row_height, score, 0, 0, 'C', True)
            
            self.set_font('Arial', '', 9)
            self.set_text_color(*COLORS['text_light'])
            
            # Multi-cell handling with background fill
            self.set_xy(x_start + 70, y_start)
            self.multi_cell(120, row_height, desc, border=0, align='L', fill=True)
            
            # Reset position for next row manually if multi_cell didn't perfectly align
            self.set_xy(x_start, y_start + row_height)
            self.line(x_start, y_start + row_height, x_start + 190, y_start + row_height) # Bottom border
            self.set_text_color(*COLORS['text_main']) # Reset color
            
            # Alternative Questions / Try Asking Instead
            alt_qs = item.get('alternative_questions', [])
            if alt_qs:
                self.set_font('Arial', 'B', 8)
                self.set_text_color(*COLORS['accent'])
                self.set_xy(x_start + 70, self.get_y() + 2)
                self.cell(40, 4, "TRY ASKING INSTEAD:", 0, 1)
                
                self.set_font('Arial', 'I', 8)
                self.set_text_color(*COLORS['text_main'])
                for aq in alt_qs:
                    q_text = aq.get('question', '')
                    if q_text:
                        self.set_x(x_start + 70)
                        self.multi_cell(120, 4, f"- \"{sanitize_text(q_text)}\"", 0, 'L')
                
                self.ln(2) # Extra spacing after alts

    def draw_radar_chart(self, scorecard):
        """Draw a radar chart for the scorecard dimensions."""
        if not scorecard: return

        # Extract data
        labels = []
        scores = []
        for item in scorecard:
            dim = sanitize_text(item.get('dimension', ''))
            # Extract numeric score "8/10" -> 8.0
            s_str = str(item.get('score', '0'))
            try:
                val = float(s_str.split('/')[0])
            except:
                val = 0.0
            
            labels.append(dim)
            scores.append(val)

        if not labels: return

        # Number of variables
        N = len(labels)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1] # Close the loop
        
        scores += scores[:1] # Close the loop
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], labels, color='grey', size=8)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([2,4,6,8,10], ["2","4","6","8","10"], color="grey", size=7)
        plt.ylim(0, 10)
        
        # Plot data
        ax.plot(angles, scores, linewidth=2, linestyle='solid', color='#3b82f6') # Blue 500
        ax.fill(angles, scores, '#3b82f6', alpha=0.2)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, bbox_inches='tight', transparent=True)
            tmp_path = tmp.name
        
        plt.close(fig)
        
        # Embed in PDF
        # Center the chart
        self.check_space(90)
        self.image(tmp_path, x=60, y=self.get_y(), w=90)
        self.ln(90)
        
        # Cleanup
        try:
            os.remove(tmp_path)
        except:
            pass


    def draw_key_value_grid(self, title, data_dict, color=COLORS['secondary']):
        """Draw a grid of key-value pairs with better spacing."""
        if not data_dict: return
        self.check_space(50)
        self.ln(8)
        self.draw_section_header(title, color)
        
        self.set_fill_color(248, 250, 252) 
        self.rect(self.get_x(), self.get_y(), 190, len(data_dict)*8 + 5, 'F')
        self.ln(2)

        for key, value in data_dict.items():
            key_label = key.replace('_', ' ').title()
            val_text = sanitize_text(str(value))
            
            self.set_font('Arial', 'B', 9)
            self.set_text_color(*COLORS['text_main'])
            self.cell(60, 8, "  " + key_label + ":", 0, 0) # Indent
            
            self.set_font('Arial', '', 9)
            self.set_text_color(*COLORS['text_light'])
            self.multi_cell(0, 8, val_text)
        self.ln(2)

    def draw_list_section(self, title, items, color=COLORS['section_comm'], bullet="•"):
        """Draw a bulleted list section with icons."""
        if not items: return
        self.check_space(len(items) * 10 + 20)
        self.ln(8)
        self.draw_section_header(title, color)
        
        self.set_font('Arial', '', 9)
        self.set_text_color(*COLORS['text_main'])
        for item in items:
            self.set_text_color(*color)
            self.cell(8, 7, bullet, 0, 0, 'R')
            self.set_text_color(*COLORS['text_main'])
            self.multi_cell(0, 7, sanitize_text(str(item)))

    def draw_two_column_lists(self, title_left, items_left, color_left, title_right, items_right, color_right):
        """Draw two lists side-by-side with dynamic height calculation."""
        if not items_left and not items_right: return
        
        start_y = self.get_y() + 5
        mid_x = 105
        col_width = 90
        
        # Helper to calculate list height
        def calculate_list_height(items, width_mm, font_size=9):
            total_h = 0
            chars_per_line = (width_mm * 2.3) # approx const for Arial 9
            for item in items:
                txt = sanitize_text(str(item))
                # rough estimate
                lines = math.ceil(len(txt) / 50) # conservative 50 chars per line for 90mm
                total_h += (lines * 6) + 2 # 6mm line height, 2mm padding
            return total_h

        # Calculate heights
        h_left = calculate_list_height(items_left, col_width) + 10 # +10 header
        h_right = calculate_list_height(items_right, col_width) + 10
        max_h = max(h_left, h_right)
        
        self.check_space(max_h + 20)
        self.ln(5)
        
        # Recalculate start_y in case of page break
        start_y = self.get_y()
        
        # Draw Headers
        self.set_xy(10, start_y)
        self.draw_section_header(title_left, color_left)
        self.set_xy(mid_x + 5, start_y)
        self.draw_section_header(title_right, color_right)
        
        content_start_y = self.get_y()
        
        # Draw Backgrounds with dynamic height
        # Left Card
        self.set_fill_color(250, 250, 255) 
        self.rect(10, content_start_y, col_width, max_h, 'F')
        # Right Card
        self.rect(mid_x + 5, content_start_y, col_width, max_h, 'F')
        
        # Draw LEFT Items
        self.set_xy(10, content_start_y + 2)
        self.set_font('Arial', '', 9)
        current_y_left = content_start_y + 2
        
        for item in items_left:
            self.set_xy(15, current_y_left) # Indent
            self.set_text_color(*color_left)
            self.cell(5, 6, "+", 0, 0)
            self.set_text_color(*COLORS['text_main'])
            
            # Save X,Y
            x = self.get_x()
            y = self.get_y()
            
            self.multi_cell(col_width - 10, 6, sanitize_text(str(item)))
            current_y_left = self.get_y() + 1 # small gap
            
        # Draw RIGHT Items
        current_y_right = content_start_y + 2
        for item in items_right:
            self.set_xy(mid_x + 10, current_y_right) # Indent
            self.set_text_color(*color_right)
            self.cell(5, 6, "!", 0, 0)
            self.set_text_color(*COLORS['text_main'])
            
            self.multi_cell(col_width - 10, 6, sanitize_text(str(item)))
            current_y_right = self.get_y() + 1 

        # Move cursor to bottom of tallest column
        self.set_y(content_start_y + max_h + 5)

    def draw_transcript(self, transcript):
        """Draw the detailed chat transcript at the end."""
        if not transcript: return
        self.add_page()
        
        self.draw_section_header("SESSION TRANSCRIPT", COLORS['primary'])
        self.ln(5)
        
        for msg in transcript:
            role = msg.get('role', 'user')
            content = sanitize_text(msg.get('content', ''))
            
            self.set_font('Arial', 'B', 8)
            
            if role == 'user':
                # User (Right side)
                self.set_text_color(*COLORS['accent'])
                self.cell(0, 5, "YOU", 0, 1, 'R')
                
                self.set_font('Arial', '', 9)
                self.set_text_color(255, 255, 255)
                self.set_fill_color(*COLORS['accent']) # Blue bubble
                
                # Calculate height
                # FPDF multi_cell doesn't return height easily, so estimating
                # We'll use a fixed width for the bubble
                bubble_w = 140
                x_pos = 200 - bubble_w - 10 # Right align
                
                self.set_x(x_pos)
                self.multi_cell(bubble_w, 6, content, 0, 'J', True)
                
            else:
                # Assistant (Left side)
                self.set_text_color(*COLORS['text_light'])
                self.cell(0, 5, "COACH", 0, 1, 'L')
                
                self.set_font('Arial', '', 9)
                self.set_text_color(*COLORS['text_main'])
                self.set_fill_color(241, 245, 249) # Gray bubble
                
                bubble_w = 140
                self.set_x(10)
                self.multi_cell(bubble_w, 6, content, 0, 'J', True)
            
            self.ln(3)

    # --- MAIN SCENARIO DRAWING ---

    def draw_coaching_report(self, data):
        self.draw_scorecard(data.get('scorecard', []))
        self.draw_key_value_grid("BEHAVIORAL SIGNALS", data.get('behavioral_signals', {}))
        
        # Use 2-Column Layout for Strengths/Weaknesses
        self.draw_two_column_lists(
            "KEY STRENGTHS", data.get('strengths', []), COLORS['success'],
            "MISSED OPPORTUNITIES", data.get('missed_opportunities', []), COLORS['warning']
        )
        
        self.draw_key_value_grid("COACHING IMPACT", data.get('coaching_impact', {}), COLORS['purple'])
        self.draw_list_section("ACTIONABLE TIPS", data.get('actionable_tips', []), COLORS['accent'], "->")

    def draw_assessment_report(self, data):
        self.draw_scorecard(data.get('scorecard', []))
        self.draw_key_value_grid("SIMULATION ANALYSIS", data.get('simulation_analysis', {}))
        
        # Use 2-Column Layout
        self.draw_two_column_lists(
            "WHAT WORKED", data.get('what_worked', []), COLORS['success'],
            "LIMITATIONS", data.get('what_limited_effectiveness', []), COLORS['danger']
        )
        
        self.draw_key_value_grid("REVENUE IMPACT", data.get('revenue_impact', {}), COLORS['danger'])
        self.draw_list_section("RECOMMENDATIONS", data.get('sales_recommendations', []), COLORS['accent'])

    def draw_learning_report(self, data):
        self.draw_key_value_grid("CONTEXT", data.get('context_summary', {}))
        self.draw_list_section("KEY INSIGHTS", data.get('key_insights', []), COLORS['purple'])
        self.draw_list_section("REFLECTIVE QUESTIONS", data.get('reflective_questions', []), COLORS['accent'], "?")
        
        # Behavioral Shifts
        shifts = data.get('behavioral_shifts', [])
        if shifts:
            self.draw_section_header("BEHAVIORAL SHIFTS", COLORS['section_skills'])
            for s in shifts:
                self.set_font('Arial', '', 9)
                self.cell(90, 6, sanitize_text(s.get('from','')), 0, 0)
                self.cell(10, 6, "->", 0, 0)
                self.set_font('Arial', 'B', 9)
                self.multi_cell(0, 6, sanitize_text(s.get('to','')))
            self.ln(5)

        self.draw_list_section("PRACTICE PLAN", data.get('practice_plan', []), COLORS['success'])
        
        if data.get('growth_outcome'):
            self.ln(5)
            self.set_font('Arial', 'I', 11)
            self.set_text_color(*COLORS['primary'])
            self.multi_cell(0, 8, f"Growth Vision: {sanitize_text(data['growth_outcome'])}", align='C')

    def draw_custom_report(self, data):
        self.draw_key_value_grid("INTERACTION QUALITY", data.get('interaction_quality', {}))
        
        # Core Skills
        skills = data.get('core_skills', [])
        if skills:
            self.draw_section_header("CORE SKILLS", COLORS['section_skills'])
            for s in skills:
                self.set_font('Arial', 'B', 9)
                self.cell(50, 6, sanitize_text(s.get('skill', '')), 0, 0)
                self.cell(30, 6, sanitize_text(s.get('rating', '')), 0, 0)
                self.set_font('Arial', '', 9)
                self.multi_cell(0, 6, sanitize_text(s.get('feedback', '')))
        
        self.draw_list_section("STRENGTHS", data.get('strengths_observed', []), COLORS['success'])
        self.draw_list_section("DEVELOPMENT AREAS", data.get('development_opportunities', []), COLORS['warning'])
        
        # Guidance
        guidance = data.get('guidance', {})
        if guidance:
            self.draw_list_section("CONTINUE", guidance.get('continue', []), COLORS['success'])
            self.draw_list_section("ADJUST", guidance.get('adjust', []), COLORS['warning'])
            self.draw_list_section("TRY NEXT", guidance.get('try_next', []), COLORS['accent'])


def generate_report(transcript, role, ai_role, scenario, framework=None, filename="coaching_report.pdf", mode="coaching", precomputed_data=None, scenario_type=None, user_name="Valued User", ai_character="alex"):
    """
    Generate a UNIFIED PDF report for all scenario types.
    """
    # Auto-detect scenario type if not provided
    if not scenario_type:
        scenario_type = detect_scenario_type(scenario, ai_role, role)
    
    print(f"Generating Unified PDF Report (scenario_type: {scenario_type}) for user: {user_name}...")
    
    # Analyze data or use precomputed
    if precomputed_data:
        data = precomputed_data
        if 'scenario_type' not in data: 
            data['scenario_type'] = scenario_type
    else:
        print("Generating new report data...")
        data = analyze_full_report_data(transcript, role, ai_role, scenario, framework, mode, scenario_type)
    
    # Sanitize data for PDF
    def sanitize_data_recursive(obj):
        if isinstance(obj, str):
            return sanitize_text(obj)
        elif isinstance(obj, dict):
            return {k: sanitize_data_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_data_recursive(item) for item in obj]
        return obj
    
    data = sanitize_data_recursive(data)
    
    pdf = DashboardPDF()
    pdf.set_scenario_type(scenario_type)
    pdf.set_user_name(user_name)
    pdf.set_character(ai_character)
    pdf.set_context(role, ai_role, scenario)
    pdf.add_page()
    
    # Get scenario_type from data if available
    scenario_type = data.get('meta', {}).get('scenario_type', scenario_type)
    
    # 1. Banner
    meta = data.get('meta', {})
    pdf.draw_banner(meta, scenario_type=scenario_type)
    
    # 1.5 Context Summary (New)
    pdf.draw_context_summary()
    
    # 1.6 Detailed Analysis (New)
    pdf.draw_detailed_analysis(data.get('detailed_analysis', ''))

    # 1.65 Behaviour Analysis (New)
    pdf.draw_behaviour_analysis(data.get('behaviour_analysis', []))
    
    # 1.68 EQ & Nuance Analysis (New)
    pdf.draw_eq_analysis(data.get('eq_analysis', []))
    
    # 1.7 Dynamic Questions (New)
    # Check for suggested_questions (sales) or reflective_questions (learning) to populate dynamic_questions
    current_questions = data.get('dynamic_questions', [])
    if not current_questions:
        if 'suggested_questions' in data and data['suggested_questions']:
            current_questions = data['suggested_questions']
        elif 'reflective_questions' in data and data['reflective_questions']:
            current_questions = data['reflective_questions']
            
    pdf.draw_dynamic_questions(current_questions)
    
    # 2. Body based on Scenario Type
    stype = str(scenario_type).lower()
    
    try:
        if 'coaching' in stype:
            pdf.draw_coaching_report(data)
            pdf.draw_scoring_methodology() # Add methodology for coaching
        elif 'sales' in stype or 'negotiation' in stype:
            pdf.draw_assessment_report(data)
            pdf.draw_scoring_methodology() # Add methodology for sales
        elif 'learning' in stype or 'reflection' in stype:
            pdf.draw_learning_report(data)
            # No scoring rubric for learning/reflection as they are non-evaluative
        else:
            pdf.draw_custom_report(data)
            pdf.draw_scoring_methodology() # Add methodology for custom
            
        # 3. Transcript (New)
        if transcript:
            pdf.draw_transcript(transcript)

    except Exception as e:
        print(f"Error drawing report body: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to generic dump if drawing fails
        pdf.draw_key_value_grid("RAW DATA DUMP (Drawing Failed)", {k:str(v)[:100] for k,v in data.items() if k != 'meta'})

    
    pdf.output(filename)
    print(f"[SUCCESS] Unified report saved: {filename} (scenario: {scenario_type})")

import os
import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID, JSONB
from flask_bcrypt import Bcrypt
import uuid
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text

db = SQLAlchemy()
bcrypt = Bcrypt()

# ---------------------------------------------------------
# Database Models
# ---------------------------------------------------------
# NOTE: Users are managed by Supabase Auth (auth.users table).
# We only store practice_history linked to their UUID.

class PracticeHistory(db.Model):
    __tablename__ = 'practice_history'
    
    # We use session_id (String UUID) as the main identifier/PK to match app logic
    session_id = db.Column(db.String(50), primary_key=True)
    # user_id stores Supabase auth.users UUID directly (no FK since no local users table)
    user_id = db.Column(UUID(as_uuid=True), nullable=True) 

    scenario = db.Column(db.Text, nullable=True) # Full scenario text stored here
    score = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow) # Renamed from date
    scenario_type = db.Column(db.String(50), nullable=False)

    role = db.Column(db.String(100))
    ai_role = db.Column(db.String(100))
    title = db.Column(db.String(255)) # NEW: Short title for display
    transcript = db.Column(JSONB)
    report_data = db.Column(JSONB)
    behaviour_analysis = db.Column(JSONB) # Dedicated column for analysis
    completed = db.Column(db.Boolean, default=False)


    assessment_report = db.relationship("AssessmentReport", backref="session", uselist=False, cascade="all, delete-orphan")
    learning_plan = db.relationship("LearningPlan", backref="session", uselist=False, cascade="all, delete-orphan")
    coaching_report = db.relationship("CoachingReport", backref="session", uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "scenario": self.scenario,
            "title": self.title, # Include in dict
            "date": self.created_at.isoformat() if self.created_at else None,
            "scenario_type": self.scenario_type,

            "role": self.role,
            "ai_role": self.ai_role,
            "transcript": self.transcript,
            "report_data": self.report_data,
            "behaviour_analysis": self.behaviour_analysis,
            "completed": self.completed,
            "reports": {
                "coaching": self.coaching_report.to_dict() if self.coaching_report else None,
                "assessment": self.assessment_report.to_dict() if self.assessment_report else None,
                "learning": self.learning_plan.to_dict() if self.learning_plan else None
            }
        }

class CoachingReport(db.Model):
    __tablename__ = 'coaching_reports'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), ForeignKey('practice_history.session_id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)  # Added to match database
    overall_score = Column(Float)
    empathy_score = Column(Float)
    psych_safety_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "overall_score": self.overall_score,
            "empathy_score": self.empathy_score,
            "psych_safety_score": self.psych_safety_score,
        }

class AssessmentReport(db.Model):
    __tablename__ = 'assessment_reports'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), ForeignKey('practice_history.session_id'), nullable=False)
    scenario_name = Column(String(255))  # NEW: Scenario title for display
    rapport_building_score = Column(Float)
    value_articulation_score = Column(Float)
    objection_handling_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "scenario_name": self.scenario_name,
            "rapport_building_score": self.rapport_building_score,
            "value_articulation_score": self.value_articulation_score,
            "objection_handling_score": self.objection_handling_score
        }

class LearningPlan(db.Model):
    __tablename__ = 'learning_plans'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), ForeignKey('practice_history.session_id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    skill_focus_areas = Column(Text)
    practice_suggestions = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "skill_focus_areas": self.skill_focus_areas,
            "practice_suggestions": self.practice_suggestions,
        }

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def init_db(app):
    """Initialize the database with the Flask app context."""
    db.init_app(app)
    with app.app_context():
        db.create_all()

def get_user_by_email(email):
    return User.query.filter_by(email=email).first()

def get_user_history(user_id):
    """Get all practice sessions for a specific user, ordered by date."""
    return PracticeHistory.query.filter_by(user_id=user_id).order_by(PracticeHistory.created_at.desc()).all()

def get_session_by_id(session_id):
    return PracticeHistory.query.get(session_id)

def create_user(data):
    """Create a new user in the database."""
    # Check if exists
    if User.query.filter((User.email == data['email']) | (User.username == data.get('username'))).first():
        return None, "User already exists"
    
    # Hash the password
    password = data['password']
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
    new_user = User(
        username=data.get('username'),
        email=data['email'],
        password_hash=password_hash,
        full_name=data.get('full_name'),
        role=data.get('role', 'user')
    )
    db.session.add(new_user)
    db.session.commit()
    return new_user, None

def create_session(session_id, data, user_id=None):
    """Create a new session record."""
    new_session = PracticeHistory(
        session_id=session_id,
        user_id=user_id, # Link to logged-in user

        scenario=data.get("scenario"), # Renamed from scenario_id
        scenario_type=data.get("scenario_type", "custom"),
        role=data.get("role"),
        ai_role=data.get("ai_role"),
        title=data.get("title"), # Store title
        transcript=data.get("transcript", []),
        report_data={},
        behaviour_analysis=data.get("behaviour_analysis", []),
        completed=False
    )
    db.session.add(new_session)
    db.session.commit()
    return new_session

def update_session(session_id, data):
    """Update an existing session with new transcript/status."""
    session = PracticeHistory.query.get(session_id)
    if session:
        if "transcript" in data:
            session.transcript = data["transcript"]
        if "report_data" in data:
            session.report_data = data["report_data"]
        if "behaviour_analysis" in data:
            session.behaviour_analysis = data["behaviour_analysis"]
        if "status" in data:
            session.completed = (data["status"] == "completed")
        
        db.session.commit()

def save_report_metrics(session_id, report_type, metrics, user_id=None):
    """Save specific report metrics to corresponding tables."""
    try:
        if report_type == "coaching":
            # Delete existing if any (simplest update strategy)
            CoachingReport.query.filter_by(session_id=session_id).delete()
            report = CoachingReport(
                session_id=session_id,
                user_id=user_id,
                overall_score=metrics.get("overall_score"),
                empathy_score=metrics.get("empathy_score"),
                psych_safety_score=metrics.get("psych_safety_score")
            )
            db.session.add(report)
            
        elif report_type == "sales" or report_type == "assessment":
            AssessmentReport.query.filter_by(session_id=session_id).delete()
            report = AssessmentReport(
                session_id=session_id,
                scenario_name=metrics.get("scenario_name", "Assessment"),
                rapport_building_score=metrics.get("rapport_building_score"),
                value_articulation_score=metrics.get("value_articulation_score"),
                objection_handling_score=metrics.get("objection_handling_score")
            )
            db.session.add(report)
            
        elif report_type == "learning":
            LearningPlan.query.filter_by(session_id=session_id).delete()
            report = LearningPlan(
                session_id=session_id,
                user_id=user_id,
                skill_focus_areas=json.dumps(metrics.get("skill_focus_areas", [])),
                practice_suggestions=json.dumps(metrics.get("practice_suggestions", []))
            )
            db.session.add(report)
            
        db.session.commit()
        print(f" [SUCCESS] Saved {report_type} report metrics for session {session_id}")
    except Exception as e:
        print(f" [ERROR] Error saving report metrics: {e}")
        db.session.rollback()

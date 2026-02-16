-- Migration: Create scenarios table and link to practice_history
-- Author: CoAct.AI
-- Date: 2026-01-31

-- Step 1: Create scenarios table
CREATE TABLE IF NOT EXISTS scenarios (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    mode VARCHAR(50) NOT NULL,  -- skill_assessment, practice, mentorship
    scenario_type VARCHAR(50) NOT NULL,  -- coaching, negotiation, reflection, mentorship
    scenario_text TEXT NOT NULL,
    user_role VARCHAR(100) NOT NULL,
    ai_role VARCHAR(100) NOT NULL,
    icon VARCHAR(50),
    output_type VARCHAR(50),  -- scored_report, learning_plan (legacy)
    report_type VARCHAR(50) NOT NULL DEFAULT 'learning',  -- coaching, assessment, learning
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Step 2: Add scenario_id to practice_history
ALTER TABLE practice_history 
ADD COLUMN IF NOT EXISTS scenario_id INTEGER REFERENCES scenarios(id);

-- Step 3: Seed initial scenarios from Practice.tsx
INSERT INTO scenarios (title, description, mode, scenario_type, scenario_text, user_role, ai_role, icon, output_type, report_type)
VALUES 
-- Skill Assessment Scenario 1
('Scenario 1: Retail Coaching', 
 'A staff member''s recent performance has dropped (sales, energy, engagement). You are the staff member receiving coaching from the manager.', 
 'skill_assessment', 
 'coaching',
 'CONTEXT: A staff member''s recent performance has dropped (sales, energy, engagement). You (the AI) are the Retail Store Manager initiating a coaching conversation. 

FOCUS AREAS: Root cause analysis, empathy, and active listening. 

AI BEHAVIOR: You are the Retail Store Manager. The user is your staff member. You feel the staff member is burnt out. Start the conversation with empathy but be firm about the performance performance issues. Ask open questions to find the root cause.',
 'Retail Sales Associate',
 'Retail Store Manager',
 'Users',
 'scored_report',
 'coaching'),

-- Skill Assessment Scenario 2
('Scenario 2: Sales and Negotiation',
 'A customer is interested in a high-value item but is hesitant about the price. You need to build rapport, discover their actual needs, and articulate value before offering any discounts.',
 'skill_assessment',
 'negotiation',
 'CONTEXT: Sales and negotiation coaching for retail staff. 

FOCUS AREAS: Developing deeper need-based questioning and delaying price discounting until value is established. 

AI BEHAVIOR: Be a customer interested in a high-value item but hesitant about price. Push for a discount early. Only agree if the salesperson builds rapport, discovers your actual needs, and articulates value first.',
 'Salesperson',
 'Retail Customer',
 'ShoppingCart',
 'scored_report',
 'assessment'),

-- Practice Scenario 3
('Scenario 3: Skill Development & Learning',
 'You are the retail staff member receiving coaching from ''Coach Alex''. The goal is to reflect on a recent interaction, identify where you missed opportunities to ask questions, and practice better responses.',
 'practice',
 'reflection',
 'CONTEXT: AI coach developing employee skills. 

FOCUS AREAS: Transitioning from feature-focused explanations to needs exploration and implementing conversational pauses. 

AI BEHAVIOR: Do NOT judge or score. Act as a supportive coach helping the user reflect on a recent interaction. Guide them to realize they need to ask more questions and pause more often.',
 'Retail Staff',
 'Coach Alex',
 'GraduationCap',
 'learning_plan',
 'learning'),

-- Mentorship Scenario 4
('Mentorship: Retail Coaching (Expert Demo)',
 'Learn by watching an expert Manager coach a struggling employee. You play the Manager, the AI plays the struggling Employee.',
 'mentorship',
 'mentorship',
 'CONTEXT: MENTORSHIP SESSION. The user plays the Manager. The AI plays the ''Struggling Employee''. 

AI BEHAVIOR: You are the Struggling Employee. Be defensive at first. Make excuses for your low performance. Only open up if the Manager (user) shows real empathy and asks the right questions.',
 'Retail Store Manager',
 'Retail Sales Associate',
 'UserCog',
 'learning_plan',
 'learning'),

-- Mentorship Scenario 5
('Mentorship: Low-Price Negotiation (Expert Demo)',
 'Learn by watching an expert Salesperson handle a difficult customer. You play the Customer, the AI plays the expert Salesperson.',
 'mentorship',
 'mentorship',
 'CONTEXT: MENTORSHIP SESSION. The user plays a difficult customer. The AI plays the ''Expert Salesperson'' demonstrating value selling and objection handling. 

AI BEHAVIOR: VAlUE SELLING. Do not discount early. Ask questions to find needs. Pivot from price to value.',
 'Retail Buyer / Customer',
 'Expert Salesperson',
 'UserCog',
 'learning_plan',
 'learning');

-- Step 4: Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_scenarios_mode ON scenarios(mode);
CREATE INDEX IF NOT EXISTS idx_scenarios_active ON scenarios(is_active);

-- Step 5: Verification query
SELECT 
    mode,
    scenario_type,
    COUNT(*) as count,
    STRING_AGG(title, ', ') as titles
FROM scenarios
GROUP BY mode, scenario_type
ORDER BY mode, scenario_type;

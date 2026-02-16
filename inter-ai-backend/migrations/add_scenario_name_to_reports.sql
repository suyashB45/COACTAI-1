-- Migration: Add scenario_name to all report tables
-- Author: CoAct.AI
-- Date: 2026-01-31

-- Step 1: Add scenario_name to coaching_reports
ALTER TABLE coaching_reports 
ADD COLUMN IF NOT EXISTS scenario_name VARCHAR(255);

-- Step 2: Add scenario_name to assessment_reports
ALTER TABLE assessment_reports 
ADD COLUMN IF NOT EXISTS scenario_name VARCHAR(255);

-- Step 3: Add scenario_name to learning_plans
ALTER TABLE learning_plans 
ADD COLUMN IF NOT EXISTS scenario_name VARCHAR(255);

-- Step 4: Backfill scenario_name from practice_history → scenarios
UPDATE coaching_reports cr
SET scenario_name = s.title
FROM practice_history ph
JOIN scenarios s ON ph.scenario_id = s.id
WHERE cr.session_id = ph.session_id
AND cr.scenario_name IS NULL;

UPDATE assessment_reports ar
SET scenario_name = s.title
FROM practice_history ph
JOIN scenarios s ON ph.scenario_id = s.id
WHERE ar.session_id = ph.session_id
AND ar.scenario_name IS NULL;

UPDATE learning_plans lp
SET scenario_name = s.title
FROM practice_history ph
JOIN scenarios s ON ph.scenario_id = s.id
WHERE lp.session_id = ph.session_id
AND lp.scenario_name IS NULL;

-- Verification queries
SELECT 'Coaching Reports' as table_name, COUNT(*) as total, COUNT(scenario_name) as with_scenario_name FROM coaching_reports
UNION ALL
SELECT 'Assessment Reports', COUNT(*), COUNT(scenario_name) FROM assessment_reports
UNION ALL
SELECT 'Learning Plans', COUNT(*), COUNT(scenario_name) FROM learning_plans;

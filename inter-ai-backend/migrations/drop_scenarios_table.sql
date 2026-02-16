-- Migration: Drop scenarios table and remove scenario_id from practice_history
-- Author: CoAct.AI
-- Date: 2026-02-01
-- Purpose: Remove database scenario storage, use frontend hardcoded scenarios instead

-- Step 1: Drop foreign key constraint from practice_history
ALTER TABLE practice_history 
DROP CONSTRAINT IF EXISTS practice_history_scenario_id_fkey;

-- Step 2: Drop scenario_id column from practice_history
ALTER TABLE practice_history 
DROP COLUMN IF EXISTS scenario_id;

-- Step 3: Drop indexes related to scenarios table
DROP INDEX IF EXISTS idx_scenarios_mode;
DROP INDEX IF EXISTS idx_scenarios_active;

-- Step 4: Drop scenarios table
DROP TABLE IF EXISTS scenarios CASCADE;

-- Verification: Check that practice_history still has required columns
-- Run this manually to verify:
-- SELECT session_id, scenario, role, ai_role, created_at FROM practice_history LIMIT 5;

-- Migration: Add mode column to practice_history table
-- Date: 2026-01-31
-- Description: Adds mode categorization (skill_assessment, practice, mentorship)

-- Add mode column as nullable initially
ALTER TABLE practice_history 
ADD COLUMN IF NOT EXISTS mode VARCHAR(50);

-- Backfill existing records based on scenario_type
UPDATE practice_history 
SET mode = CASE 
    WHEN scenario_type IN ('coaching', 'negotiation') THEN 'skill_assessment'
    WHEN scenario_type = 'reflection' THEN 'practice'
    WHEN scenario_type = 'mentorship' THEN 'mentorship'
    ELSE 'practice'
END
WHERE mode IS NULL;

-- Optional: Make mode NOT NULL after backfill (uncomment if desired)
-- ALTER TABLE practice_history 
-- ALTER COLUMN mode SET NOT NULL;

-- Verify migration
SELECT scenario_type, mode, COUNT(*) as count 
FROM practice_history 
GROUP BY scenario_type, mode
ORDER BY scenario_type, mode;

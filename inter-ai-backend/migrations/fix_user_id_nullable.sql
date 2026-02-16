-- Fix user_id constraint to allow nullable values for unauthenticated sessions
-- This addresses the "null value in column user_id violates not-null constraint" error

ALTER TABLE practice_history ALTER COLUMN user_id DROP NOT NULL;

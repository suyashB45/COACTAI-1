-- Migration: Rename sales_reports table to assessment_reports
-- Author: CoAct.AI
-- Date: 2026-01-31

-- Step 1: Rename the table
ALTER TABLE sales_reports RENAME TO assessment_reports;

-- Verification query
SELECT COUNT(*) as total_assessment_reports FROM assessment_reports;

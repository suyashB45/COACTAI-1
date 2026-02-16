# Database Migration Guide

## Overview
You need to run the SQL migrations to set up the database schema. These migrations create the scenarios table and update report tables.

## Migration Order

Run these migrations **in order** via Supabase Dashboard → SQL Editor:

### 1. Create Scenarios Table ✅ **RUN FIRST**
**File:** `migrations/create_scenarios_table.sql`

This creates:
- `scenarios` table with all fields including `report_type`
- Foreign key link from `practice_history` to `scenarios`
- Seeds 5 initial scenarios
- Creates indexes

**How to run:**
1. Go to https://supabase.com/dashboard
2. Select your project
3. Navigate to **SQL Editor**
4. Copy the entire contents of `create_scenarios_table.sql`
5. Paste and click **Run**

**Expected output:**
```
CREATE TABLE
ALTER TABLE
INSERT 0 5
CREATE INDEX
CREATE INDEX
```

---

### 2. Rename sales_reports to assessment_reports
**File:** `migrations/rename_sales_to_assessment.sql`

This renames:
- `sales_reports` table → `assessment_reports`

**How to run:**
- Same process as above, use Supabase SQL Editor

---

### 3. Add scenario_name to Report Tables
**File:** `migrations/add_scenario_name_to_reports.sql`

This adds `scenario_name` column to:
- `coaching_reports`
- `assessment_reports`
- `learning_plans`

**How to run:**
- Same process as above, use Supabase SQL Editor

---

### 4. Add mode column to practice_history (if not already done)
**File:** `migrations/add_mode_column.sql`

This adds the `mode` field to `practice_history`.

**How to run:**
- Same process as above, use Supabase SQL Editor

---

## Verification

After running ALL migrations, verify with this query:

```sql
-- Check scenarios table
SELECT id, title, mode, scenario_type, report_type FROM scenarios;

-- Should return 5 rows

-- Check report tables have scenario_name
SELECT column_name 
FROM information_schema.columns 
WHERE table_name IN ('coaching_reports', 'assessment_reports', 'learning_plans')
AND column_name = 'scenario_name';

-- Should return 3 rows (one for each table)
```

## Quick Start Command

If you haven't created the scenarios table yet, **START HERE**:

1. Open Supabase Dashboard SQL Editor
2. Copy/paste `migrations/create_scenarios_table.sql`
3. Click **Run**
4. Refresh your app - errors should be gone!

## Troubleshooting

**Error: "column scenarios.report_type does not exist"**
→ You need to run `create_scenarios_table.sql` migration

**Error: "table scenarios does not exist"**
→ You need to run `create_scenarios_table.sql` migration

**Error: "relation sales_reports does not exist"**
→ You need to run `rename_sales_to_assessment.sql` migration

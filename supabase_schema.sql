-- ============================================================
-- CoAct.AI - FRESH DATABASE SCHEMA
-- ============================================================
-- Run this on a NEW/EMPTY Supabase database
-- ============================================================

-- ============================================================
-- 1. PRACTICE HISTORY TABLE (Main sessions table)
-- ============================================================
CREATE TABLE public.practice_history (
    session_id TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    scenario_type TEXT NOT NULL DEFAULT 'custom',
    role TEXT,
    ai_role TEXT,
    scenario TEXT,
    framework TEXT,
    transcript JSONB DEFAULT '[]'::jsonb,
    report_data JSONB DEFAULT '{}'::jsonb,
    completed BOOLEAN DEFAULT FALSE,
    score INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_practice_history_user_id ON public.practice_history(user_id);
CREATE INDEX idx_practice_history_created_at ON public.practice_history(created_at DESC);

ALTER TABLE public.practice_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own sessions" ON public.practice_history FOR SELECT USING ((select auth.uid()) = user_id);
CREATE POLICY "Users can insert own sessions" ON public.practice_history FOR INSERT WITH CHECK ((select auth.uid()) = user_id);
CREATE POLICY "Users can update own sessions" ON public.practice_history FOR UPDATE USING ((select auth.uid()) = user_id);
CREATE POLICY "Users can delete own sessions" ON public.practice_history FOR DELETE USING ((select auth.uid()) = user_id);


-- ============================================================
-- 2. COACHING REPORTS TABLE
-- ============================================================
CREATE TABLE public.coaching_reports (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES public.practice_history(session_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    overall_score FLOAT,
    empathy_score FLOAT,
    psych_safety_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_coaching_reports_session_id ON public.coaching_reports(session_id);
CREATE INDEX idx_coaching_reports_user_id ON public.coaching_reports(user_id);

ALTER TABLE public.coaching_reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own coaching" ON public.coaching_reports FOR SELECT USING ((select auth.uid()) = user_id);
CREATE POLICY "Users can insert own coaching" ON public.coaching_reports FOR INSERT WITH CHECK ((select auth.uid()) = user_id);
CREATE POLICY "Users can update own coaching" ON public.coaching_reports FOR UPDATE USING ((select auth.uid()) = user_id);
CREATE POLICY "Users can delete own coaching" ON public.coaching_reports FOR DELETE USING ((select auth.uid()) = user_id);


-- ============================================================
-- 3. SALES REPORTS TABLE
-- ============================================================
CREATE TABLE public.sales_reports (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES public.practice_history(session_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    rapport_building_score FLOAT,
    value_articulation_score FLOAT,
    objection_handling_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sales_reports_session_id ON public.sales_reports(session_id);
CREATE INDEX idx_sales_reports_user_id ON public.sales_reports(user_id);

ALTER TABLE public.sales_reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own sales" ON public.sales_reports FOR SELECT USING ((select auth.uid()) = user_id);
CREATE POLICY "Users can insert own sales" ON public.sales_reports FOR INSERT WITH CHECK ((select auth.uid()) = user_id);
CREATE POLICY "Users can update own sales" ON public.sales_reports FOR UPDATE USING ((select auth.uid()) = user_id);
CREATE POLICY "Users can delete own sales" ON public.sales_reports FOR DELETE USING ((select auth.uid()) = user_id);


-- ============================================================
-- 4. MENTORSHIP REPORTS TABLE
-- ============================================================
CREATE TABLE public.mentorship_reports (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES public.practice_history(session_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    mentorship_type TEXT NOT NULL, -- 'retail_coaching' or 'negotiation'
    empathy_score FLOAT,
    questioning_score FLOAT,
    value_selling_score FLOAT,
    overall_score FLOAT,
    key_takeaways TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_mentorship_reports_session_id ON public.mentorship_reports(session_id);
CREATE INDEX idx_mentorship_reports_user_id ON public.mentorship_reports(user_id);

ALTER TABLE public.mentorship_reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own mentorship" ON public.mentorship_reports FOR SELECT USING ((select auth.uid()) = user_id);
CREATE POLICY "Users can insert own mentorship" ON public.mentorship_reports FOR INSERT WITH CHECK ((select auth.uid()) = user_id);
CREATE POLICY "Users can update own mentorship" ON public.mentorship_reports FOR UPDATE USING ((select auth.uid()) = user_id);
CREATE POLICY "Users can delete own mentorship" ON public.mentorship_reports FOR DELETE USING ((select auth.uid()) = user_id);


-- ============================================================
-- 5. LEARNING PLANS TABLE
-- ============================================================
CREATE TABLE public.learning_plans (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES public.practice_history(session_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    skill_focus_areas TEXT,
    practice_suggestions TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_learning_plans_session_id ON public.learning_plans(session_id);
CREATE INDEX idx_learning_plans_user_id ON public.learning_plans(user_id);

ALTER TABLE public.learning_plans ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own learning" ON public.learning_plans FOR SELECT USING ((select auth.uid()) = user_id);
CREATE POLICY "Users can insert own learning" ON public.learning_plans FOR INSERT WITH CHECK ((select auth.uid()) = user_id);
CREATE POLICY "Users can update own learning" ON public.learning_plans FOR UPDATE USING ((select auth.uid()) = user_id);
CREATE POLICY "Users can delete own learning" ON public.learning_plans FOR DELETE USING ((select auth.uid()) = user_id);


-- ============================================================
-- 6. USER PROFILES TABLE (Optional - extends auth.users)
-- ============================================================
CREATE TABLE public.user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    full_name TEXT,
    avatar_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile" ON public.user_profiles FOR SELECT USING ((select auth.uid()) = id);
CREATE POLICY "Users can insert own profile" ON public.user_profiles FOR INSERT WITH CHECK ((select auth.uid()) = id);
CREATE POLICY "Users can update own profile" ON public.user_profiles FOR UPDATE USING ((select auth.uid()) = id);


-- ============================================================
-- 7. AUTO-CREATE PROFILE ON SIGNUP
-- ============================================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, full_name)
    VALUES (NEW.id, NEW.raw_user_meta_data->>'full_name');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();


-- ============================================================
-- DONE! All tables created with RLS enabled.
-- ============================================================
-- Don't forget to enable "Leaked password protection" in:
-- Authentication > Settings > Security
-- ============================================================

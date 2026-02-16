"use client"

import { useEffect, useState } from "react"
import { useParams, useNavigate } from "react-router-dom"
import { Download, AlertCircle, Target, History, Zap, Award, BookOpen, MessageSquare, ChevronRight, Check, X, ArrowLeft, Clock, CheckCircle2, AlertTriangle, Brain, Quote, Lightbulb } from "lucide-react"
import { motion, Variants } from "framer-motion"
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from "recharts"

import Navigation from "../components/landing/Navigation"
import { getApiUrl } from "@/lib/api"
import { supabase } from "@/lib/supabase"
import { Button } from "@/components/ui/button"

// --- TYPES & INTERFACES (UPDATED TO MATCH BACKEND) ---

interface BaseMeta {
    scenario_id: string
    outcome_status: string
    overall_grade: string
    summary: string
    scenario_type?: string
    session_mode?: string
    emotional_trajectory?: string
    session_quality?: string
    key_themes?: string[]
    scenario?: string
}

interface BehaviourItem {
    behavior: string
    quote: string
    insight: string
    impact: string
    improved_approach: string
}

interface DetailedAnalysisItem {
    topic: string
    analysis: string
}

interface ScorecardItem {
    dimension: string
    score: string
    description?: string // Keeping for backward compatibility
    reasoning?: string // New field for Proof of Marks
    quote?: string
    suggestion?: string
    alternative_questions?: { question: string; rationale: string }[]
}

interface GenericReportData {
    meta: BaseMeta
    type?: string
    transcript?: { role: "user" | "assistant", content: string }[]
    behaviour_analysis?: BehaviourItem[]
    detailed_analysis?: DetailedAnalysisItem[] | string
    question_analysis?: QuestionAnalysis // NEW: Enhanced question analysis
    [key: string]: any
}

// Scenario 1: Coaching
interface CoachingReportData extends GenericReportData {
    scorecard: ScorecardItem[]
    strengths: string[]
    missed_opportunities: string[]
    actionable_tips: string[]
    eq_analysis?: { nuance: string; proof: string; suggestion: string }[]
}

// Scenario 2: Sales
interface SalesReportData extends GenericReportData {
    scorecard: ScorecardItem[]
    sales_recommendations: string[]
    suggested_questions?: string[]
}

// Scenario 3: Learning
interface LearningReportData extends GenericReportData {
    key_insights: string[]
    reflective_questions: string[]
    practice_plan: string[]
    growth_outcome: string
    behavioral_shifts?: { from: string; to: string }[] // Legacy support
}

// Question Analysis (Backend Enhanced)
interface QuestionMissed {
    question: string
    category?: string // Discovery, Probing, Clarifying, Vision, Closing
    timing?: string // Early, Mid, Late
    why_important: string
    when_to_ask: string
    impact_if_asked: string
}

interface QuestionAnalysis {
    questions_asked_count: number
    questions_missed: QuestionMissed[]
    question_quality_score?: string
    question_quality_feedback?: string
    questioning_improvement_tip?: string
}

const containerVars: Variants = {
    hidden: { opacity: 0 },
    show: {
        opacity: 1,
        transition: { staggerChildren: 0.1 }
    }
}

const itemVars: Variants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 50 } }
}

export default function Report() {
    const params = useParams()
    const navigate = useNavigate()
    const sessionId = params.sessionId as string
    const [data, setData] = useState<GenericReportData | null>(null)
    const [loading, setLoading] = useState(true)
    const [showTranscript, setShowTranscript] = useState(false)

    useEffect(() => {
        const fetchReport = async () => {
            try {
                if (!sessionId) return

                const { data: { session } } = await supabase.auth.getSession();
                const headers: Record<string, string> = {
                    'Content-Type': 'application/json'
                };
                if (session?.access_token) {
                    headers['Authorization'] = `Bearer ${session.access_token}`;
                }

                const response = await fetch(getApiUrl(`/api/session/${sessionId}/report_data`), { headers })
                if (!response.ok) throw new Error("Failed to fetch report data")
                const data: GenericReportData = await response.json()
                setData(data)
                setLoading(false)
            } catch (err) {
                console.error("Error generating report:", err)
                setLoading(false)
            }
        }
        fetchReport()
    }, [sessionId])

    const handleDownload = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const headers: Record<string, string> = {};
            if (session?.access_token) {
                headers['Authorization'] = `Bearer ${session.access_token}`;
            }

            const response = await fetch(getApiUrl(`/api/report/${sessionId}`), { headers })
            if (!response.ok) throw new Error("Failed to generate PDF")

            // Check if backend returned a JSON with a redirect URL (Blob Storage)
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                const data = await response.json();
                if (data.redirect && data.url) {
                    // Open the pre-signed URL in a new tab/window for download
                    window.open(data.url, "_blank");
                    return;
                }
            }

            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `CoActAI_Report_${sessionId}.pdf`
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
            window.URL.revokeObjectURL(url)
        } catch (error) {
            console.error("Error downloading PDF:", error)
            alert("PDF export failed. Please ensure the backend is running.")
        }
    }

    if (loading) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center bg-background gap-6 font-sans">
                <div className="relative">
                    <div className="w-16 h-16 rounded-full border-4 border-primary/30 border-t-primary animate-spin"></div>
                    <div className="absolute inset-0 flex items-center justify-center">
                        <Zap className="w-6 h-6 text-primary animate-pulse" />
                    </div>
                </div>
                <p className="text-muted-foreground animate-pulse font-medium tracking-wide">GENERATING ANALYSIS...</p>
            </div>
        )
    }

    if (!data || !data.meta) {
        return (
            <div className="min-h-screen bg-background p-12 flex flex-col items-center justify-center font-sans">
                <AlertCircle className="h-16 w-16 text-amber-500 mb-6" />
                <h2 className="text-3xl font-bold text-foreground mb-3">Report Unavailable</h2>
                <Button onClick={() => navigate("/")} variant="secondary">Return Home</Button>
            </div>
        )
    }

    const type = (data.type || data.meta.scenario_type || 'custom').toLowerCase()

    const renderContent = () => {
        if (type.includes('coaching')) return <CoachingView data={data as CoachingReportData} />
        if (type.includes('sales') || type.includes('negotiation')) return <SalesView data={data as SalesReportData} />
        if (type.includes('learning') || type.includes('reflection') || type.includes('mentorship')) return <LearningView data={data as LearningReportData} />
        return <CustomView data={data as GenericReportData} />
    }

    return (
        <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/30">
            <Navigation />
            <div className="fixed inset-0 pointer-events-none -z-10">
                <div className="absolute top-[-20%] right-[-10%] w-[500px] h-[500px] bg-primary/5 rounded-full blur-[120px]" />
                <div className="absolute bottom-[-20%] left-[-10%] w-[400px] h-[400px] bg-purple-600/5 rounded-full blur-[120px]" />
            </div>

            <main className="relative container mx-auto px-4 sm:px-6 py-24 sm:py-32 space-y-12">
                {/* HEADER & BANNER */}
                <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
                    <div className="flex flex-col md:flex-row gap-8 justify-between items-start">
                        <div>
                            <button onClick={() => navigate('/history')} className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors mb-6 group">
                                <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
                                Back to History
                            </button>
                            <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
                                <div>
                                    <div className="flex items-center gap-3 mb-2">
                                        <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-primary/10 text-primary border border-primary/20">
                                            {data.meta.scenario_type || 'Custom Scenario'}
                                        </span>
                                        {data.meta.session_mode && (
                                            <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border ${data.meta.session_mode === 'skill_assessment'
                                                ? 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20'
                                                : data.meta.session_mode === 'practice'
                                                    ? 'bg-blue-500/10 text-blue-600 border-blue-500/20'
                                                    : 'bg-purple-500/10 text-purple-600 border-purple-500/20'
                                                }`}>
                                                {data.meta.session_mode.replace('_', ' ')}
                                            </span>
                                        )}
                                        <span className="text-muted-foreground text-sm font-medium flex items-center gap-1">
                                            <Clock className="w-3.5 h-3.5" />
                                            {new Date().toLocaleDateString()}
                                        </span>
                                    </div>
                                    <h1 className="text-4xl md:text-5xl font-black text-foreground mb-2">Session Analysis</h1>
                                    <p className="text-xl text-muted-foreground">{data.meta.summary}</p>
                                </div>
                            </div>
                        </div>
                        <div className="flex flex-col items-end gap-4 min-w-[200px]">
                            {type !== 'learning' && type !== 'reflection' && type !== 'mentorship' && (
                                <div className="text-right">
                                    <div className="text-sm font-bold text-muted-foreground uppercase tracking-widest mb-1">Overall Grade</div>
                                    <div className={`text-7xl font-black text-transparent bg-clip-text bg-gradient-to-br from-blue-400 to-indigo-500 leading-none`}>
                                        {data.meta.overall_grade}
                                    </div>
                                </div>
                            )}
                            <Button onClick={handleDownload} variant="outline" className="gap-2 border-border hover:bg-accent w-full">
                                <Download className="w-4 h-4" /> Export PDF Report
                            </Button>
                        </div>
                    </div>

                    {/* METRICS BANNER (Matches PDF Banner) */}
                    <div className="grid md:grid-cols-3 gap-6">
                        {data.meta.emotional_trajectory && (
                            <GlassCard className="p-4 flex flex-col bg-indigo-500/5 border-indigo-500/10">
                                <span className="text-xs font-bold text-indigo-500 uppercase tracking-wider mb-2">Emotional Arc</span>
                                <span className="text-sm font-medium text-foreground">{data.meta.emotional_trajectory}</span>
                            </GlassCard>
                        )}
                        {data.meta.session_quality && (
                            <GlassCard className="p-4 flex flex-col bg-emerald-500/5 border-emerald-500/10">
                                <span className="text-xs font-bold text-emerald-500 uppercase tracking-wider mb-2">Session Quality</span>
                                <span className="text-sm font-medium text-foreground">{data.meta.session_quality}</span>
                            </GlassCard>
                        )}
                        {data.meta.key_themes && (
                            <GlassCard className="p-4 flex flex-col bg-pink-500/5 border-pink-500/10">
                                <span className="text-xs font-bold text-pink-500 uppercase tracking-wider mb-2">Key Themes</span>
                                <div className="flex flex-wrap gap-2">
                                    {data.meta.key_themes.slice(0, 3).map((t, i) => (
                                        <span key={i} className="text-xs bg-background/50 px-2 py-1 rounded border border-border/50">{t}</span>
                                    ))}
                                </div>
                            </GlassCard>
                        )}
                    </div>
                </motion.div>

                {/* SCENARIO CONTEXT (Added to sync with PDF) */}
                {data.meta.scenario && (
                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                        <ScenarioContextSection scenario={data.meta.scenario} />
                    </motion.div>
                )}

                <motion.div variants={containerVars} initial="hidden" animate="show">
                    {renderContent()}
                </motion.div>

                {/* TRANSCRIPT */}
                {data.transcript && (
                    <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} className="rounded-3xl bg-card border border-border overflow-hidden backdrop-blur-sm">
                        <div className="px-8 py-6 flex items-center justify-between cursor-pointer hover:bg-accent transition-colors group" onClick={() => setShowTranscript(!showTranscript)}>
                            <div className="flex items-center gap-4">
                                <div className="p-3 rounded-xl bg-primary/10 transition-colors group-hover:bg-primary/20 text-primary"><History className="w-5 h-5" /></div>
                                <div>
                                    <h3 className="text-lg font-bold text-foreground">Session Transcript</h3>
                                    <p className="text-sm text-muted-foreground">Review the full conversation log</p>
                                </div>
                            </div>
                            <ChevronRight className={`w-5 h-5 text-muted-foreground transition-transform ${showTranscript ? 'rotate-90' : ''}`} />
                        </div>
                        {showTranscript && (
                            <div className="p-8 max-h-[600px] overflow-y-auto space-y-6 border-t border-border bg-muted/20">
                                {data.transcript.map((msg, idx) => (
                                    <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                        <div className={`p-4 rounded-2xl max-w-[80%] text-sm leading-relaxed shadow-sm ${msg.role === 'user' ? 'bg-primary text-primary-foreground rounded-br-none' : 'bg-card text-foreground border border-border rounded-bl-none'}`}>
                                            {msg.content}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </motion.div>
                )}
            </main>
        </div>
    )
}

// --- SHARED COMPONENTS ---

const GlassCard = ({ children, className = "" }: any) => (
    <motion.div variants={itemVars} className={`bg-card rounded-2xl border border-border p-6 shadow-sm ${className}`}>{children}</motion.div>
)

const SectionHeader = ({ icon: Icon, title, colorClass = "text-primary", bgClass = "bg-primary/10" }: any) => (
    <div className="flex items-center gap-4 mb-6">
        <div className={`p-3 rounded-xl ${bgClass} ring-1 ring-border/50`}><Icon className={`w-6 h-6 ${colorClass}`} /></div>
        <h2 className="text-xl font-bold text-foreground tracking-wide uppercase">{title}</h2>
    </div>
)

const ProgressBar = ({ value, colorClass = "bg-primary" }: { value: number, colorClass?: string }) => (
    <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
        <motion.div initial={{ width: 0 }} whileInView={{ width: `${value}%` }} transition={{ duration: 1 }} className={`h-full ${colorClass}`} />
    </div>
)



const ScenarioContextSection = ({ scenario }: { scenario: string }) => {
    // Clean up scenario text similar to PDF logic
    const cleanScenario = scenario
        .replace(/CONTEXT:/g, "")
        .replace(/Situation:/g, "")
        .replace(/AI BEHAVIOR:[\s\S]*/g, "") // Remove AI Behavior section
        .replace(/AI ROLE:[\s\S]*/g, "")
        .replace(/USER ROLE:[\s\S]*/g, "")
        .trim();

    return (
        <GlassCard className="bg-slate-50 dark:bg-slate-900 border-l-4 border-l-primary/50">
            <SectionHeader icon={BookOpen} title="Scenario Context" colorClass="text-slate-500" bgClass="bg-slate-500/10" />
            <p className="text-base text-foreground/80 leading-relaxed font-serif italic">
                "{cleanScenario}"
            </p>
        </GlassCard>
    )
}

const EQAnalysisSection = ({ items }: { items?: { nuance: string; proof: string; suggestion: string }[] }) => {
    if (!items || items.length === 0) return null
    return (
        <GlassCard>
            <SectionHeader icon={Brain} title="Emotional Intelligence (EQ) & Nuance" colorClass="text-pink-500" bgClass="bg-pink-500/10" />
            <div className="space-y-6">
                {items.map((item, i) => (
                    <div key={i} className="p-4 rounded-xl bg-pink-500/5 border border-pink-500/10">
                        <h3 className="font-bold text-lg text-pink-600 mb-2">{item.nuance}</h3>
                        <div className="space-y-3">
                            {item.proof && (
                                <div className="flex gap-2 text-sm text-foreground/80">
                                    <span className="font-bold text-xs uppercase text-slate-500 mt-1">Proof:</span>
                                    <span className="italic">"{item.proof}"</span>
                                </div>
                            )}
                            {item.suggestion && (
                                <div className="flex gap-2 text-sm text-foreground/80">
                                    <span className="font-bold text-xs uppercase text-slate-500 mt-1">Suggestion:</span>
                                    <span>{item.suggestion}</span>
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </GlassCard>
    )
}

const BehaviourAnalysisSection = ({ items }: { items?: BehaviourItem[] }) => {
    if (!items || items.length === 0) return null
    return (
        <GlassCard className="col-span-full">
            <SectionHeader icon={Brain} title="Behavioral Analysis" colorClass="text-purple-500" bgClass="bg-purple-500/10" />
            <div className="space-y-6">
                {items.map((item, i) => (
                    <div key={i} className="flex flex-col md:flex-row gap-6 p-6 rounded-xl bg-background border border-border hover:shadow-md transition-shadow">
                        <div className="flex-1 space-y-3">
                            <div className="flex items-center gap-3">
                                <h3 className="font-bold text-lg text-foreground">{item.behavior}</h3>
                                <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${item.impact.toLowerCase().includes('positive') ? 'bg-emerald-500/10 text-emerald-500' : 'bg-rose-500/10 text-rose-500'}`}>
                                    {item.impact}
                                </span>
                            </div>
                            <p className="text-muted-foreground leading-relaxed">{item.insight}</p>
                            {item.quote && (
                                <div className="flex gap-3 text-sm text-muted-foreground bg-muted/50 p-3 rounded-lg border-l-4 border-primary/20 italic">
                                    <Quote className="w-4 h-4 shrink-0 opacity-50" /> "{item.quote}"
                                </div>
                            )}
                        </div>
                        {item.improved_approach && (
                            <div className="md:w-1/3 bg-blue-500/5 border border-blue-500/10 p-5 rounded-lg">
                                <h4 className="flex items-center gap-2 text-xs font-bold text-blue-500 uppercase tracking-wider mb-3">
                                    <Lightbulb className="w-4 h-4" /> Try This Instead
                                </h4>
                                <p className="text-sm text-foreground/90 leading-relaxed font-medium">{item.improved_approach}</p>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </GlassCard>
    )
}

const DetailedAnalysisSection = ({ items }: { items?: DetailedAnalysisItem[] | string }) => {
    if (!items) return null
    return (
        <GlassCard>
            <SectionHeader icon={BookOpen} title="Deep Dive Analysis" colorClass="text-indigo-500" bgClass="bg-indigo-500/10" />
            <div className="space-y-6">
                {Array.isArray(items) ? items.map((item, i) => (
                    <div key={i} className="space-y-2">
                        <h3 className="font-bold text-foreground border-l-4 border-indigo-500 pl-3">{item.topic}</h3>
                        <p className="text-muted-foreground leading-relaxed pl-4">{item.analysis}</p>
                    </div>
                )) : (
                    <p className="text-muted-foreground leading-relaxed">{items}</p>
                )}
            </div>
        </GlassCard>
    )
}

const SkillRadarChart = ({ items }: { items: ScorecardItem[] }) => {
    // Only show if we have data
    if (!items || items.length < 3) return null

    const data = items.map(item => ({
        subject: item.dimension,
        A: parseFloat(item.score.split('/')[0]) || 0,
        fullMark: 10
    }))

    return (
        <div className="w-full h-[300px] -ml-4 sm:ml-0 mb-8">
            <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
                    <PolarGrid stroke="currentColor" className="text-muted-foreground/20" />
                    <PolarAngleAxis
                        dataKey="subject"
                        tick={{ fill: 'currentColor', fontSize: 11, fontWeight: 600 }}
                        className="text-muted-foreground"
                    />
                    <Radar
                        name="Score"
                        dataKey="A"
                        stroke="#8b5cf6"
                        fill="#8b5cf6"
                        fillOpacity={0.3}
                    />
                </RadarChart>
            </ResponsiveContainer>
        </div>
    )
}

// Enhanced Questions Section
const QuestionsSection = ({ analysis }: { analysis?: QuestionAnalysis }) => {
    if (!analysis || !analysis.questions_missed || analysis.questions_missed.length === 0) return null

    // Group questions by timing
    const questionsByTiming = {
        Early: analysis.questions_missed.filter(q => q.timing === 'Early'),
        Mid: analysis.questions_missed.filter(q => q.timing === 'Mid'),
        Late: analysis.questions_missed.filter(q => q.timing === 'Late'),
        Uncategorized: analysis.questions_missed.filter(q => !q.timing)
    }

    const getCategoryColor = (category?: string) => {
        switch (category) {
            case 'Discovery': return 'bg-blue-500/10 text-blue-600 border-blue-500/20'
            case 'Probing': return 'bg-purple-500/10 text-purple-600 border-purple-500/20'
            case 'Clarifying': return 'bg-amber-500/10 text-amber-600 border-amber-500/20'
            case 'Vision': return 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20'
            case 'Closing': return 'bg-rose-500/10 text-rose-600 border-rose-500/20'
            default: return 'bg-slate-500/10 text-slate-600 border-slate-500/20'
        }
    }

    const getTimingColor = (timing?: string) => {
        switch (timing) {
            case 'Early': return 'bg-green-500/10 text-green-700'
            case 'Mid': return 'bg-yellow-500/10 text-yellow-700'
            case 'Late': return 'bg-orange-500/10 text-orange-700'
            default: return 'bg-gray-500/10 text-gray-700'
        }
    }

    return (
        <GlassCard className="border-l-4 border-l-primary">
            <SectionHeader icon={MessageSquare} title="Questions You Should Have Asked" colorClass="text-primary" bgClass="bg-primary/10" />

            {/* Quality Score Summary */}
            {analysis.question_quality_score && (
                <div className="mb-6 p-4 rounded-xl bg-primary/5 border border-primary/10">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Question Quality Score</span>
                        <span className="text-2xl font-black text-primary">{analysis.question_quality_score}</span>
                    </div>
                    {analysis.question_quality_feedback && (
                        <p className="text-sm text-foreground/80 mb-2">{analysis.question_quality_feedback}</p>
                    )}
                    {analysis.questioning_improvement_tip && (
                        <div className="flex gap-2 items-start mt-3 p-3 bg-blue-500/5 rounded-lg border border-blue-500/10">
                            <Lightbulb className="w-4 h-4 text-blue-600 mt-0.5 shrink-0" />
                            <span className="text-sm text-foreground/90 font-medium">{analysis.questioning_improvement_tip}</span>
                        </div>
                    )}
                    <div className="mt-3 text-xs text-muted-foreground">
                        You asked {analysis.questions_asked_count} question{analysis.questions_asked_count !== 1 ? 's' : ''} • {analysis.questions_missed.length} key question{analysis.questions_missed.length !== 1 ? 's' : ''} missed
                    </div>
                </div>
            )}

            {/* Questions by Timing */}
            <div className="space-y-6">
                {(['Early', 'Mid', 'Late', 'Uncategorized'] as const).map(timing => {
                    const questions = questionsByTiming[timing]
                    if (questions.length === 0) return null

                    return (
                        <div key={timing}>
                            {timing !== 'Uncategorized' && (
                                <div className="flex items-center gap-2 mb-3">
                                    <Clock className="w-4 h-4 text-muted-foreground" />
                                    <h4 className="text-sm font-bold text-foreground uppercase tracking-wider">{timing} Conversation</h4>
                                </div>
                            )}
                            <div className="space-y-4">
                                {questions.map((q, idx) => (
                                    <div key={idx} className="p-5 rounded-xl bg-background border border-border hover:shadow-md transition-all group">
                                        {/* Question with badges */}
                                        <div className="flex flex-wrap items-start gap-2 mb-3">
                                            <span className="text-base font-semibold text-foreground italic flex-1 min-w-[200px]">"{q.question}"</span>
                                            {q.category && (
                                                <span className={`px-2 py-1 rounded text-xs font-bold uppercase tracking-wider border ${getCategoryColor(q.category)}`}>
                                                    {q.category}
                                                </span>
                                            )}
                                            {q.timing && (
                                                <span className={`px-2 py-1 rounded text-xs font-semibold ${getTimingColor(q.timing)}`}>
                                                    {q.timing}
                                                </span>
                                            )}
                                        </div>

                                        {/* Details */}
                                        <div className="space-y-2 text-sm">
                                            <div className="flex gap-2">
                                                <span className="font-bold text-xs text-primary uppercase tracking-wider shrink-0">Why:</span>
                                                <span className="text-muted-foreground">{q.why_important}</span>
                                            </div>
                                            <div className="flex gap-2">
                                                <span className="font-bold text-xs text-emerald-600 uppercase tracking-wider shrink-0">When:</span>
                                                <span className="text-muted-foreground">{q.when_to_ask}</span>
                                            </div>
                                            <div className="flex gap-2">
                                                <span className="font-bold text-xs text-amber-600 uppercase tracking-wider shrink-0">Impact:</span>
                                                <span className="text-muted-foreground">{q.impact_if_asked}</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )
                })}
            </div>
        </GlassCard>
    )
}

const ScorecardSection = ({ items }: { items: ScorecardItem[] }) => (
    <GlassCard>
        <SectionHeader icon={Target} title="Performance Scorecard" colorClass="text-emerald-500" bgClass="bg-emerald-500/10" />

        {/* Radar Chart Visualization */}
        <SkillRadarChart items={items} />

        <div className="space-y-8">
            {items?.map((item, i) => {
                const numScore = parseFloat(item.score.split('/')[0] || "0")
                const color = numScore >= 8 ? 'bg-emerald-500' : numScore >= 5 ? 'bg-amber-500' : 'bg-rose-500'

                return (
                    <div key={i} className="group border-b border-border/50 last:border-0 pb-8 last:pb-0">
                        {/* Header: Dimension + Score */}
                        <div className="flex justify-between items-end mb-3">
                            <h3 className="font-bold text-lg text-foreground group-hover:text-primary transition-colors">{item.dimension}</h3>
                            <span className={`font-mono font-black text-2xl ${numScore >= 8 ? 'text-emerald-500' : numScore >= 5 ? 'text-amber-500' : 'text-rose-500'}`}>{item.score}</span>
                        </div>

                        <ProgressBar value={numScore * 10} colorClass={color} />

                        <div className="mt-5 space-y-4">
                            {/* PROOF / REASONING */}
                            <div className="bg-muted/30 rounded-xl p-4 border border-border/50">
                                <span className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-2 block">Proof of Marks</span>
                                <p className="text-sm text-foreground leading-relaxed">
                                    {item.reasoning || item.description}
                                </p>
                                {item.quote && (
                                    <div className="mt-3 flex gap-2 text-sm text-muted-foreground/80 italic">
                                        <Quote className="w-4 h-4 shrink-0 opacity-50" />
                                        <span>"{item.quote}"</span>
                                    </div>
                                )}
                            </div>

                            {/* SUGGESTION */}
                            {item.suggestion && (
                                <div className="bg-emerald-500/5 rounded-xl p-4 border border-emerald-500/10 relative overflow-hidden">
                                    <div className="absolute top-0 left-0 w-1 h-full bg-emerald-500/30" />
                                    <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-widest mb-2 block flex items-center gap-2">
                                        <Lightbulb className="w-3.5 h-3.5" /> Suggestion
                                    </span>
                                    <p className="text-sm font-medium text-foreground/90 leading-relaxed">
                                        {item.suggestion}
                                    </p>
                                </div>
                            )}

                            {/* ALTERNATIVE QUESTIONS */}
                            {item.alternative_questions && item.alternative_questions.length > 0 && (
                                <div className="mt-2">
                                    <p className="text-xs font-bold text-primary/70 uppercase tracking-wider mb-2">Try asking instead:</p>
                                    <div className="grid sm:grid-cols-2 gap-3">
                                        {item.alternative_questions.map((aq, idx) => (
                                            <div key={idx} className="p-3 bg-background rounded-lg border border-border text-xs shadow-sm">
                                                <span className="font-semibold text-primary block mb-1">"{aq.question}"</span>
                                                <span className="text-muted-foreground text-[10px] uppercase tracking-wide opacity-80">{aq.rationale}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )
            })}
        </div>
    </GlassCard>
)

// --- VIEW COMPONENTS ---

const CoachingView = ({ data }: { data: CoachingReportData }) => (
    <div className="space-y-8">
        <div className="grid lg:grid-cols-2 gap-8">
            <ScorecardSection items={data.scorecard} />
            <div className="space-y-8">
                <DetailedAnalysisSection items={data.detailed_analysis as DetailedAnalysisItem[]} />
                <EQAnalysisSection items={data.eq_analysis} />
            </div>
        </div>

        <BehaviourAnalysisSection items={data.behaviour_analysis} />

        {/* Enhanced Questions Section */}
        {data.question_analysis && <QuestionsSection analysis={data.question_analysis} />}

        <div className="grid md:grid-cols-2 gap-6">
            <GlassCard>
                <SectionHeader icon={CheckCircle2} title="Key Strengths" colorClass="text-emerald-500" bgClass="bg-emerald-500/10" />
                <ul className="space-y-3">
                    {data.strengths?.map((s, i) => (
                        <li key={i} className="flex gap-3 text-sm text-foreground/90 bg-emerald-500/5 p-3 rounded-lg border border-emerald-500/10">
                            <Check className="w-4 h-4 text-emerald-500 mt-0.5" /> {s}
                        </li>
                    ))}
                </ul>
            </GlassCard>
            <GlassCard>
                <SectionHeader icon={AlertTriangle} title="Missed Opportunities" colorClass="text-amber-500" bgClass="bg-amber-500/10" />
                <ul className="space-y-3">
                    {data.missed_opportunities?.map((s, i) => (
                        <li key={i} className="flex gap-3 text-sm text-foreground/90 bg-amber-500/5 p-3 rounded-lg border border-amber-500/10">
                            <X className="w-4 h-4 text-amber-500 mt-0.5" /> {s}
                        </li>
                    ))}
                </ul>
            </GlassCard>
        </div>

        <GlassCard className="border-t-4 border-t-purple-500">
            <SectionHeader icon={Zap} title="Actionable Drills" colorClass="text-purple-500" bgClass="bg-purple-500/10" />
            <div className="grid md:grid-cols-2 gap-6">
                {data.actionable_tips?.map((tip, i) => (
                    <div key={i} className="flex gap-4 p-4 rounded-xl bg-purple-500/5 border border-purple-500/10">
                        <div className="w-8 h-8 rounded-full bg-purple-500/10 flex items-center justify-center shrink-0 text-purple-600 font-bold text-sm">#{i + 1}</div>
                        <p className="text-sm text-foreground leading-relaxed font-medium">{tip}</p>
                    </div>
                ))}
            </div>
        </GlassCard>
    </div>
)

const SalesView = ({ data }: { data: SalesReportData }) => (
    <div className="space-y-8">
        <div className="grid lg:grid-cols-2 gap-8">
            <ScorecardSection items={data.scorecard} />
            <div className="space-y-8">
                <DetailedAnalysisSection items={data.detailed_analysis as DetailedAnalysisItem[]} />
            </div>
        </div>

        <BehaviourAnalysisSection items={data.behaviour_analysis} />

        {/* Enhanced Questions Section */}
        {data.question_analysis && <QuestionsSection analysis={data.question_analysis} />}

        <GlassCard>
            <SectionHeader icon={Award} title="Strategic Recommendations" colorClass="text-emerald-500" bgClass="bg-emerald-500/10" />
            <div className="grid md:grid-cols-2 gap-6">
                {data.sales_recommendations?.map((rec, i) => (
                    <div key={i} className="bg-background p-5 rounded-xl border border-border text-foreground text-sm leading-relaxed hover:shadow-md transition-shadow">
                        <CheckCircle2 className="w-5 h-5 text-emerald-500 mb-3" />
                        {rec}
                    </div>
                ))}
            </div>
        </GlassCard>
    </div>
)

const LearningView = ({ data }: { data: LearningReportData }) => (
    <div className="space-y-8">
        <div className="grid lg:grid-cols-2 gap-8">
            <GlassCard className="bg-gradient-to-br from-primary/5 to-transparent">
                <SectionHeader icon={BookOpen} title="Key Insights" colorClass="text-primary" />
                <ul className="space-y-4">
                    {data.key_insights?.map((insight, i) => (
                        <li key={i} className="flex gap-3 text-foreground/90 leading-relaxed">
                            <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 shrink-0" />
                            {insight}
                        </li>
                    ))}
                </ul>
            </GlassCard>
            <DetailedAnalysisSection items={data.detailed_analysis as DetailedAnalysisItem[]} />
        </div>

        <BehaviourAnalysisSection items={data.behaviour_analysis} />

        {/* Enhanced Questions Section */}
        {data.question_analysis && <QuestionsSection analysis={data.question_analysis} />}

        <div className="grid lg:grid-cols-2 gap-8">
            <GlassCard>
                <SectionHeader icon={MessageSquare} title="Reflective Questions" colorClass="text-indigo-500" bgClass="bg-indigo-500/10" />
                <ul className="space-y-6">
                    {data.reflective_questions?.map((q, i) => (
                        <li key={i} className="text-lg font-medium text-foreground/80 italic pl-6 border-l-4 border-indigo-500/30">"{q}"</li>
                    ))}
                </ul>
            </GlassCard>
            <GlassCard>
                <SectionHeader icon={Zap} title="Practice Plan" colorClass="text-amber-500" bgClass="bg-amber-500/10" />
                <div className="space-y-4">
                    {data.practice_plan?.map((plan, i) => (
                        <div key={i} className="p-4 bg-amber-500/5 rounded-xl border border-amber-500/10">
                            <h4 className="text-xs font-bold text-amber-600 uppercase mb-1">Exercise {i + 1}</h4>
                            <p className="text-sm text-foreground">{plan}</p>
                        </div>
                    ))}
                </div>
            </GlassCard>
        </div>

        <div className="bg-primary/5 p-8 rounded-3xl text-center border border-primary/10">
            <h3 className="text-xs font-bold text-primary uppercase tracking-widest mb-4">Growth Vision</h3>
            <p className="text-2xl font-serif italic text-foreground max-w-2xl mx-auto leading-relaxed">{data.growth_outcome}</p>
        </div>
    </div>
)

const CustomView = ({ data }: { data: GenericReportData }) => (
    <div className="space-y-8">
        <DetailedAnalysisSection items={data.detailed_analysis as DetailedAnalysisItem[]} />
        <BehaviourAnalysisSection items={data.behaviour_analysis} />
        {data.strengths_observed && (
            <GlassCard>
                <SectionHeader icon={CheckCircle2} title="Strengths Observed" />
                <div className="flex flex-wrap gap-2">
                    {data.strengths_observed.map((s: string, i: number) => (
                        <span key={i} className="px-3 py-1 bg-emerald-500/10 text-emerald-600 rounded-full text-sm font-medium border border-emerald-500/20">{s}</span>
                    ))}
                </div>
            </GlassCard>
        )}
    </div>
)

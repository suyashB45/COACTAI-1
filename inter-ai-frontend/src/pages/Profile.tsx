import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Mail, Clock, Trophy, Calendar, ArrowRight, LogOut, PlayCircle } from 'lucide-react';
import Navigation from '../components/landing/Navigation';
import { supabase } from '../lib/supabase';
import { getApiUrl } from '../lib/api';

interface Session {
    session_id: string;
    scenario_type: string;
    role: string;
    ai_role: string;
    completed: boolean;
    date: string;
    score?: number;
}

const Profile: React.FC = () => {
    const navigate = useNavigate();
    const [user, setUser] = useState<any>(null);
    const [sessions, setSessions] = useState<Session[]>([]);
    const [loading, setLoading] = useState(true);
    const [stats, setStats] = useState({ total: 0, completed: 0, avgScore: 0 });

    useEffect(() => {
        fetchUserAndHistory();
    }, []);

    const fetchUserAndHistory = async () => {
        try {
            // Get current user from Supabase
            const { data: { user: currentUser } } = await supabase.auth.getUser();

            if (!currentUser) {
                navigate('/login');
                return;
            }

            setUser(currentUser);

            // Get session for API calls
            const { data: { session } } = await supabase.auth.getSession();

            if (session) {
                // Fetch practice history
                const res = await fetch(getApiUrl('/api/history'), {
                    headers: {
                        'Authorization': `Bearer ${session.access_token}`,
                        'Content-Type': 'application/json'
                    }
                });

                if (res.ok) {
                    const historyData = await res.json();
                    setSessions(historyData);

                    // Calculate stats
                    const completed = historyData.filter((s: Session) => s.completed).length;
                    const scores = historyData.filter((s: Session) => s.score).map((s: Session) => s.score || 0);
                    const avgScore = scores.length > 0 ? Math.round(scores.reduce((a: number, b: number) => a + b, 0) / scores.length) : 0;

                    setStats({
                        total: historyData.length,
                        completed,
                        avgScore
                    });
                }
            }
        } catch (error) {
            console.error('Error fetching profile data:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = async () => {
        await supabase.auth.signOut();
        localStorage.removeItem('user');
        navigate('/login');
    };

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
        });
    };

    if (loading) {
        return (
            <div className="min-h-screen hero-ultra-modern flex items-center justify-center">
                <div className="animate-spin w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full"></div>
            </div>
        );
    }

    return (
        <div className="min-h-screen hero-ultra-modern font-sans">
            <Navigation />

            <div className="max-w-4xl mx-auto px-4 py-8 sm:py-12 pb-16 sm:pb-32">
                {/* Profile Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="card-ultra-glass p-4 sm:p-8 mb-8"
                >
                    <div className="flex flex-col sm:flex-row items-center sm:justify-between gap-6">
                        <div className="flex flex-col sm:flex-row items-center gap-6 text-center sm:text-left">
                            <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center font-bold text-2xl sm:text-3xl text-white">
                                {(user?.user_metadata?.full_name || user?.email || 'U').charAt(0).toUpperCase()}
                            </div>
                            <div>
                                <h1 className="text-2xl sm:text-3xl font-bold text-foreground mb-1 sm:mb-2">
                                    {user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'User'}
                                </h1>
                                <div className="flex items-center justify-center sm:justify-start gap-2 text-muted-foreground text-sm">
                                    <Mail className="w-4 h-4" />
                                    <span>{user?.email}</span>
                                </div>
                            </div>
                        </div>
                        <button
                            onClick={handleLogout}
                            className="flex items-center gap-2 px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors text-sm"
                        >
                            <LogOut className="w-4 h-4" />
                            Logout
                        </button>
                    </div>
                </motion.div>

                {/* Stats Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="card-ultra-glass p-4 sm:p-6 text-center"
                    >
                        <PlayCircle className="w-6 h-6 sm:w-8 sm:h-8 text-purple-400 mx-auto mb-2" />
                        <div className="text-2xl sm:text-3xl font-bold text-foreground">{stats.total}</div>
                        <div className="text-muted-foreground text-xs sm:text-sm">Total Sessions</div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="card-ultra-glass p-4 sm:p-6 text-center flex flex-col items-center justify-center"
                    >
                        <Trophy className="w-6 h-6 sm:w-8 sm:h-8 text-green-400 mx-auto mb-2" />
                        <div className="text-2xl sm:text-3xl font-bold text-foreground">{stats.completed}</div>
                        <div className="text-muted-foreground text-xs sm:text-sm">Completed</div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="card-ultra-glass p-4 sm:p-6 text-center"
                    >
                        <Clock className="w-6 h-6 sm:w-8 sm:h-8 text-blue-400 mx-auto mb-2" />
                        <div className="text-2xl sm:text-3xl font-bold text-foreground">{stats.avgScore || '-'}</div>
                        <div className="text-muted-foreground text-xs sm:text-sm">Avg Score</div>
                    </motion.div>
                </div>

                {/* Practice History */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="card-ultra-glass p-4 sm:p-6"
                >
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-xl font-bold text-foreground">Practice History</h2>
                        <Link
                            to="/practice"
                            className="text-primary hover:text-primary/80 transition-colors flex items-center gap-2"
                        >
                            New Session <ArrowRight className="w-4 h-4" />
                        </Link>
                    </div>

                    {sessions.length === 0 ? (
                        <div className="text-center py-12">
                            <PlayCircle className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                            <p className="text-muted-foreground">No practice sessions yet.</p>
                            <Link
                                to="/practice"
                                className="inline-block mt-4 px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                            >
                                Start Your First Session
                            </Link>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {sessions.slice(0, 10).map((session, index) => (
                                <motion.div
                                    key={session.session_id}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: index * 0.05 }}
                                    className="flex items-center justify-between p-4 bg-card/50 rounded-lg hover:bg-card/80 transition-colors cursor-pointer border border-border/50"
                                    onClick={() => navigate(`/report/${session.session_id}`)}
                                >
                                    <div className="flex items-center gap-4">
                                        <div className={`w-3 h-3 rounded-full ${session.completed ? 'bg-green-500' : 'bg-yellow-500'}`} />
                                        <div>
                                            <div className="text-foreground font-medium">{session.role || 'Practice Session'}</div>
                                            <div className="text-muted-foreground text-sm">{session.scenario_type || 'Custom'}</div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        {session.score && (
                                            <span className="text-primary font-medium">{session.score}/10</span>
                                        )}
                                        <div className="flex items-center gap-1 text-muted-foreground text-sm">
                                            <Calendar className="w-4 h-4" />
                                            {session.date ? formatDate(session.date) : 'Recent'}
                                        </div>
                                        <ArrowRight className="w-4 h-4 text-muted-foreground" />
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    );
};

export default Profile;

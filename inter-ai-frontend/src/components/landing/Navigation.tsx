import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Menu, X, User, LogOut } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { supabase } from '../../lib/supabase';
import { ModeToggle } from '../mode-toggle';
import ContactSalesModal from './ContactSalesModal';

const Navigation = () => {
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const [isSalesModalOpen, setIsSalesModalOpen] = useState(false);
    const [user, setUser] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();
    const location = useLocation();

    // Check auth state
    useEffect(() => {
        const checkAuth = async () => {
            const { data: { session } } = await supabase.auth.getSession();
            setUser(session?.user ?? null);
            setLoading(false);
        };

        checkAuth();

        // Listen for auth changes
        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
            setUser(session?.user ?? null);
        });

        return () => subscription.unsubscribe();
    }, []);

    const handleLogout = async () => {
        await supabase.auth.signOut();
        setUser(null);
        setIsMobileMenuOpen(false);
        navigate('/login');
    };

    const navItems = [
        { name: 'Features', href: '#features', page: '/' },
        { name: 'How It Works', href: '#how-it-works', page: '/' },
        { name: 'Practice', href: '/practice', page: '/practice' },
        { name: 'History', href: '/history', page: '/history' },
    ];

    const handleNavClick = (item: { name: string, href: string, page: string }) => {
        setIsMobileMenuOpen(false);

        if (item.href.startsWith('/')) {
            navigate(item.href);
            return;
        }

        // Handle hash links
        if (location.pathname !== '/') {
            navigate('/', { state: { scrollTo: item.href.substring(1) } });
        } else {
            // Add a small delay for mobile menu to close smoothly before scrolling
            setTimeout(() => {
                const id = item.href.startsWith('#') ? item.href.substring(1) : item.href;
                const element = document.getElementById(id);
                if (element) {
                    element.scrollIntoView({ behavior: 'smooth' });
                    navigate(item.href, { replace: true });
                }
            }, 100);
        }
    };

    // Handle scroll after navigation from another page
    useEffect(() => {
        if (location.pathname === '/' && location.state && (location.state as any).scrollTo) {
            const id = (location.state as any).scrollTo;
            setTimeout(() => {
                const element = document.getElementById(id);
                if (element) element.scrollIntoView({ behavior: 'smooth' });
                // clear state
                window.history.replaceState({}, document.title);
            }, 100);
        }
    }, [location]);

    return (
        <>
            <motion.nav
                className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 border-b bg-background/80 backdrop-blur-xl border-border py-2 shadow-md`}
                initial={{ y: -100 }}
                animate={{ y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <div className="container mx-auto px-6">
                    <div className="flex items-center justify-between">
                        {/* Logo */}
                        <motion.div
                            className="flex items-center gap-2 cursor-pointer group"
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={() => navigate('/')}
                        >
                            <div className="relative w-10 h-10">
                                <div className="absolute inset-0 bg-gradient-to-tr from-blue-600 to-purple-600 rounded-xl rotate-6 group-hover:rotate-12 transition-transform duration-300 opacity-80" />
                                <div className="absolute inset-0 bg-slate-950 rounded-xl border border-white/10 flex items-center justify-center">
                                    <Sparkles className="w-5 h-5 text-blue-400" />
                                </div>
                            </div>
                            <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-foreground to-muted-foreground">
                                CoAct<span className="text-primary">.AI</span>
                            </span>
                        </motion.div>

                        {/* Desktop Navigation */}
                        <div className="hidden md:flex items-center gap-8">
                            {navItems.map((item, index) => {
                                const isActive = location.pathname === item.page && (item.href.startsWith('#') ? location.hash === item.href : true);
                                return (
                                    <motion.button
                                        key={item.name}
                                        onClick={() => handleNavClick(item)}
                                        className={`text-sm font-medium transition-colors relative group ${isActive ? 'text-primary' : 'text-muted-foreground hover:text-foreground'}`}
                                        initial={{ opacity: 0, y: -10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: index * 0.1 }}
                                    >
                                        {item.name}
                                        <span className={`absolute -bottom-1 left-0 h-px bg-blue-500 transition-all duration-300 ${isActive ? 'w-full' : 'w-0 group-hover:w-full'}`} />
                                    </motion.button>
                                );
                            })}
                            <button
                                onClick={() => setIsSalesModalOpen(true)}
                                className="text-sm font-medium text-muted-foreground hover:text-blue-500 transition-colors"
                            >
                                Contact Sales
                            </button>
                        </div>

                        {/* Auth Buttons */}
                        <div className="hidden md:flex items-center gap-4">
                            {!loading && (
                                user ? (
                                    <>
                                        <motion.button
                                            className="flex items-center gap-2 text-muted-foreground hover:text-foreground font-medium text-sm transition-colors"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            onClick={() => navigate('/profile')}
                                        >
                                            <User className="w-4 h-4" />
                                            Profile
                                        </motion.button>
                                        <motion.button
                                            className="flex items-center gap-2 text-red-400 hover:text-red-300 font-medium text-sm transition-colors"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            onClick={handleLogout}
                                        >
                                            <LogOut className="w-4 h-4" />
                                            Logout
                                        </motion.button>
                                    </>
                                ) : (
                                    <>
                                        <motion.button
                                            className="text-muted-foreground hover:text-foreground font-medium text-sm transition-colors"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ delay: 0.2 }}
                                            onClick={() => navigate('/login')}
                                        >
                                            Sign In
                                        </motion.button>

                                    </>
                                )
                            )}
                            <ModeToggle />
                        </div>

                        {/* Mobile Menu Button + Toggle */}
                        <div className="flex md:hidden items-center gap-2">
                            <ModeToggle />
                            <motion.button
                                className="text-foreground p-2"
                                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                                whileTap={{ scale: 0.9 }}
                            >
                                {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                            </motion.button>
                        </div>
                    </div>

                    {/* Mobile Menu */}
                    <AnimatePresence>
                        {isMobileMenuOpen && (
                            <motion.div
                                className="md:hidden border-t border-border mt-4 bg-background/95 backdrop-blur-xl rounded-2xl overflow-hidden shadow-xl"
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                            >
                                <div className="p-4 space-y-2">
                                    {navItems.map((item) => {
                                        const isActive = location.pathname === item.page && (item.href.startsWith('#') ? location.hash === item.href : true);
                                        return (
                                            <button
                                                key={item.name}
                                                onClick={() => handleNavClick(item)}
                                                className={`block w-full text-left px-4 py-3 rounded-xl font-medium transition-colors ${isActive ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:bg-muted hover:text-foreground'}`}
                                            >
                                                {item.name}
                                            </button>
                                        );
                                    })}
                                    <button
                                        onClick={() => {
                                            setIsSalesModalOpen(true);
                                            setIsMobileMenuOpen(false);
                                        }}
                                        className="block w-full text-left px-4 py-3 rounded-xl font-medium text-blue-500 bg-blue-500/10 hover:bg-blue-500/20 transition-colors"
                                    >
                                        Contact Sales
                                    </button>
                                    <div className="pt-4 flex flex-col gap-3">
                                        {user ? (
                                            <>
                                                <button
                                                    onClick={() => {
                                                        navigate('/profile');
                                                        setIsMobileMenuOpen(false);
                                                    }}
                                                    className="flex items-center justify-center gap-2 w-full px-4 py-3 rounded-xl font-medium text-white bg-purple-600"
                                                >
                                                    <User className="w-4 h-4" /> Profile
                                                </button>
                                                <button
                                                    onClick={handleLogout}
                                                    className="flex items-center justify-center gap-2 w-full px-4 py-3 rounded-xl font-medium text-red-400 hover:bg-red-500/20"
                                                >
                                                    <LogOut className="w-4 h-4" /> Logout
                                                </button>
                                            </>
                                        ) : (
                                            <>

                                            </>
                                        )}
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </motion.nav>
            <ContactSalesModal isOpen={isSalesModalOpen} onClose={() => setIsSalesModalOpen(false)} />
        </>
    );
};

export default Navigation;

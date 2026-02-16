import { motion } from 'framer-motion';
import { ArrowRight, Star } from 'lucide-react';
import Scene3D from './Scene3D';
import { useNavigate } from 'react-router-dom';

const HeroSection = () => {
    const navigate = useNavigate();

    // Generate random particles
    const particles = Array.from({ length: 20 }, (_, i) => ({
        id: i,
        left: `${Math.random() * 100}%`,
        top: `${Math.random() * 100}%`,
        delay: `${Math.random() * 5}s`,
        size: 3 + Math.random() * 4,
    }));

    return (
        <section className="hero-ultra-modern section-lg overflow-x-hidden pt-24 pb-8 min-h-[70vh] flex items-center">

            {/* Floating Particles */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                {particles.map((p) => (
                    <div
                        key={p.id}
                        className="particle bg-foreground/20"
                        style={{
                            left: p.left,
                            top: p.top,
                            width: `${p.size}px`,
                            height: `${p.size}px`,
                            animationDelay: p.delay,
                        }}
                    />
                ))}
            </div>

            {/* Animated Background Elements */}
            <div className="absolute inset-0 overflow-hidden">
                <motion.div
                    className="absolute morph-blob"
                    style={{ top: '5rem', left: '2.5rem', width: '18rem', height: '18rem', background: 'rgba(139, 92, 246, 0.15)', filter: 'blur(60px)' }}
                    animate={{
                        x: [0, 100, 0],
                        y: [0, -50, 0],
                    }}
                    transition={{
                        duration: 20,
                        repeat: Infinity,
                        ease: "linear"
                    }}
                />
                <motion.div
                    className="absolute morph-blob"
                    style={{ top: '10rem', right: '5rem', width: '22rem', height: '22rem', background: 'rgba(236, 72, 153, 0.15)', filter: 'blur(70px)' }}
                    animate={{
                        x: [0, -80, 0],
                        y: [0, 60, 0],
                    }}
                    transition={{
                        duration: 25,
                        repeat: Infinity,
                        ease: "linear"
                    }}
                />
                <motion.div
                    className="absolute morph-blob"
                    style={{ bottom: '5rem', left: '25%', width: '16rem', height: '16rem', background: 'rgba(59, 130, 246, 0.12)', filter: 'blur(50px)' }}
                    animate={{
                        x: [0, 60, 0],
                        y: [0, -40, 0],
                    }}
                    transition={{
                        duration: 18,
                        repeat: Infinity,
                        ease: "linear"
                    }}
                />
            </div>

            <motion.div
                className="absolute z-20 inline-flex items-center gap-2 px-3 py-1.5 sm:px-4 sm:py-2 bg-purple-500/10 dark:bg-purple-500/10 border border-purple-500/30 backdrop-blur-md shadow-lg"
                style={{
                    top: '90px',
                    left: '100px',
                    borderRadius: '9999px'
                }}
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0, y: [0, -5, 0] }}
                transition={{
                    x: { duration: 0.8, ease: "easeOut" },
                    y: { duration: 2, repeat: Infinity, ease: "easeInOut" }
                }}
            >
                <Star className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-slate-950 dark:text-purple-400 fill-slate-950 dark:fill-purple-400" />
                <span className="text-[10px] sm:text-sm font-bold text-slate-950 dark:text-purple-300">AI-Powered Coaching Platform</span>
            </motion.div>

            <div className="container mx-auto px-6 relative z-10">
                <div className="grid md:grid-cols-2 gap-12 items-center">
                    {/* Left Content */}
                    <motion.div
                        initial={{ opacity: 0, x: -50 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
                        className="text-left order-2 lg:order-1 mt-12 sm:mt-0"
                    >
                        {/* Badge */}


                        {/* Headline */}
                        <motion.h1
                            className="text-3xl sm:text-5xl lg:text-7xl font-bold mb-6 leading-tight"
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 }}
                        >
                            <span className="text-foreground block">AI-Powered Coaching  </span>
                            <span className="block text-gradient-animate text-glow">for Real World Conversation</span>
                        </motion.h1>

                        {/* Subheadline */}
                        <motion.p
                            className="text-lg text-muted-foreground mb-10 leading-relaxed max-w-2xl mx-auto lg:mx-0"
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.4 }}
                        >
                            Personalized AI coaching that adapts to your goals, delivering real-time guidance, actionable insights, and measurable performance improvement.
                        </motion.p>

                        {/* CTA Buttons */}
                        <motion.div
                            className="flex flex-col sm:flex-row gap-8 mb-12 justify-center lg:justify-start"
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.5 }}
                        >
                            <motion.button
                                className="btn-ultra-modern text-base px-6 py-3 w-full sm:w-auto flex items-center justify-center"
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => navigate('/practice')}
                            >
                                Get Started
                                <ArrowRight className="w-5 h-5 ml-2" />
                            </motion.button>
                            <motion.button
                                className="btn-glass text-base px-6 py-3 w-full sm:w-auto border border-foreground/20 hover:border-foreground/40 rounded-xl"
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}
                            >
                                How It Works
                            </motion.button>
                        </motion.div>


                    </motion.div>

                    {/* Right Content - 3D Scene */}
                    <motion.div
                        className="relative hidden md:block"
                        initial={{ opacity: 0, x: 50 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.8, delay: 0.3, ease: [0.4, 0, 0.2, 1] }}
                    >
                        <div className="relative" style={{ width: '100%', height: '560px' }}>
                            <Scene3D />

                            {/* Floating Elements */}
                            <motion.div
                                className="absolute"
                                style={{ top: '1rem', right: '1rem', background: 'rgba(255,255,255,0.9)', backdropFilter: 'blur(6px)', border: '1px solid rgba(139,92,246,0.3)', borderRadius: '1rem', padding: '0.75rem' }}
                                animate={{ y: [0, -10, 0] }}
                                transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                            >
                                <div className="text-center">
                                    <div className="text-2xl font-bold text-black">95%</div>
                                    <div className="text-sm text-black font-medium">Success Rate</div>
                                </div>
                            </motion.div>



                            <motion.div
                                className="absolute"
                                style={{ bottom: '1rem', left: '1rem', background: 'rgba(255,255,255,0.9)', backdropFilter: 'blur(6px)', border: '1px solid rgba(59,130,246,0.3)', borderRadius: '1rem', padding: '0.75rem' }}
                                animate={{ y: [0, 10, 0] }}
                                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                            >
                                <div className="text-center">
                                    <div className="text-2xl font-bold text-black">24/7</div>
                                    <div className="text-sm text-black font-medium">AI Support</div>
                                </div>
                            </motion.div>
                        </div>
                    </motion.div>
                </div>
            </div>
        </section>
    );
};

export default HeroSection;

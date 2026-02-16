import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { Brain, Target, Users, Zap, Shield, TrendingUp } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const FeaturesSection = () => {
    const ref = useRef(null);
    const isInView = useInView(ref, { once: true, amount: 0.1 });
    const navigate = useNavigate();

    const features = [
        {
            icon: Brain,
            title: "AI-Powered Insights",
            description: "Advanced machine learning algorithms analyze your patterns and provide personalized recommendations for growth.",
            gradient: "from-purple-500 to-pink-500"
        },
        {
            icon: Target,
            title: "Goal Tracking",
            description: "Set clear objectives and track your progress with intelligent milestone management and achievement analytics.",
            gradient: "from-blue-500 to-cyan-500"
        },
        {
            icon: Users,
            title: "Community Support",
            description: "Connect with like-minded individuals, share experiences, and learn from others in a supportive environment.",
            gradient: "from-green-500 to-emerald-500"
        },
        {
            icon: Zap,
            title: "Real-time Feedback",
            description: "Get instant feedback and adjustments to your strategies as you progress through your journey.",
            gradient: "from-orange-500 to-red-500"
        },
        {
            icon: Shield,
            title: "Privacy First",
            description: "Your data is protected with enterprise-grade security and privacy controls you can trust.",
            gradient: "from-indigo-500 to-purple-500"
        },
        {
            icon: TrendingUp,
            title: "Progress Analytics",
            description: "Visualize your growth with detailed analytics and insights into your development journey.",
            gradient: "from-pink-500 to-rose-500"
        }
    ];

    return (
        <section ref={ref} className="relative section-enhanced py-12" id="features">
            {/* Background */}
            <div className="absolute inset-0 bg-background pointer-events-none -z-10">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_30%,oklch(from_var(--primary)_l_c_h_/_0.1),transparent_50%),radial-gradient(circle_at_80%_70%,oklch(from_var(--secondary)_l_c_h_/_0.1),transparent_50%)]" />
            </div>

            <div className="container mx-auto px-6 relative z-10">
                <div className="text-center mb-16">
                    {/* Badge */}
                    <motion.div
                        className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/5 border border-purple-400/15 rounded-full text-sm font-bold text-slate-950 dark:text-purple-300 mb-8"
                        initial={{ opacity: 0, y: 20 }}
                        animate={isInView ? { opacity: 1, y: 0 } : {}}
                        transition={{ delay: 0.1 }}
                    >
                        Features
                    </motion.div>

                    {/* Main Heading */}
                    <motion.h2
                        className="text-4xl lg:text-5xl font-bold mb-8 leading-tight text-foreground"
                        initial={{ opacity: 0, y: 30 }}
                        animate={isInView ? { opacity: 1, y: 0 } : {}}
                        transition={{ delay: 0.2 }}
                    >
                        <span className="text-foreground">Powerful Features</span>
                        <br />
                        <span className="text-ultra-gradient">for Performance Improvement</span>
                    </motion.h2>

                    {/* Subheading */}
                    <motion.p
                        className="text-lg text-muted-foreground max-w-3xl mx-auto leading-relaxed"
                        initial={{ opacity: 0, y: 30 }}
                        animate={isInView ? { opacity: 1, y: 0 } : {}}
                        transition={{ delay: 0.3 }}
                    >
                        Discover the comprehensive suite of tools and features designed to accelerate your personal and professional development.
                    </motion.p>
                </div>

                {/* Features Grid */}
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {features.map((feature, index) => (
                        <motion.div
                            key={feature.title}
                            className="card-ultra-glass group"
                            initial={{ opacity: 0, y: 30 }}
                            animate={isInView ? { opacity: 1, y: 0 } : {}}
                            transition={{ delay: 0.4 + index * 0.1 }}
                            whileHover={{ y: -8 }}
                        >
                            <div className={`w-16 h-16 bg-gradient-to-br ${feature.gradient} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                                <feature.icon className="w-8 h-8 text-white" />
                            </div>
                            <h3 className="text-xl font-bold text-foreground mb-4">
                                {feature.title}
                            </h3>
                            <p className="text-muted-foreground leading-relaxed">
                                {feature.description}
                            </p>
                        </motion.div>
                    ))}
                </div>

                {/* CTA Section */}
                <motion.div
                    className="text-center mt-32"
                    initial={{ opacity: 0, y: 30 }}
                    animate={isInView ? { opacity: 1, y: 0 } : {}}
                    transition={{ delay: 0.8 }}
                >
                    <h3 className="text-3xl font-bold text-foreground mb-6">
                        Ready for Real-world communication scenarios?
                    </h3>
                    <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
                        Join thousands of users who are already experiencing the power of AI-driven personal development.
                    </p>
                    <motion.button
                        className="btn-ultra-modern text-lg px-8 py-4"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => navigate('/practice')}
                    >
                        Get Started Today
                    </motion.button>
                </motion.div>
            </div>
        </section>
    );
};

export default FeaturesSection;

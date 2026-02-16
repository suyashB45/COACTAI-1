import { motion } from 'framer-motion';
import { Mic, MessageSquare, TrendingUp, Sparkles } from 'lucide-react';

const HowItWorksSection = () => {
    const steps = [
        {
            icon: Mic,
            title: "Speak Naturally",
            description: "Choose a scenario and start a real-time voice conversation with our AI. No typing needed – just talk like you would to a real person.",
            color: "blue"
        },
        {
            icon: MessageSquare,
            title: "Get Real-time Feedback",
            description: "Receive instant, adaptive responses. The AI embodies the role perfectly, challenging you and helping you practice difficult conversations.",
            color: "purple"
        },
        {
            icon: TrendingUp,
            title: "Analyze & Grow",
            description: "After every session, get a detailed report with insights on your tone, clarity, and effectiveness, plus specific tips for improvement.",
            color: "pink"
        }
    ];

    return (
        <section className="relative py-12 overflow-hidden bg-background" id="how-it-works">
            {/* Background Gradients */}
            <div className="absolute inset-0 pointer-events-none">
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl" />
                <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
            </div>

            <div className="container mx-auto px-6 relative z-10">
                <div className="text-center mb-16">
                    <motion.div
                        className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/5 border border-blue-400/15 rounded-full text-sm font-bold text-slate-950 dark:text-blue-300 mb-6"
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                    >
                        <Sparkles className="w-4 h-4" />
                        Simple Process
                    </motion.div>

                    <motion.h2
                        className="text-4xl md:text-5xl font-bold text-foreground mb-6"
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.1 }}
                    >
                        Master Any Conversation in <span className="text-ultra-gradient">3 Steps</span>
                    </motion.h2>

                    <motion.p
                        className="text-lg text-muted-foreground max-w-2xl mx-auto"
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.2 }}
                    >
                        Our AI-powered platform makes it easy to practice, learn, and improve your communication skills on your own schedule.
                    </motion.p>
                </div>

                <div className="grid md:grid-cols-3 gap-8 relative">
                    {/* Connecting Line (Desktop) */}
                    <div className="hidden md:block absolute top-12 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 transform translate-y-8" />

                    {steps.map((step, index) => (
                        <motion.div
                            key={index}
                            className="relative"
                            initial={{ opacity: 0, y: 30 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.2 }}
                        >
                            <div className={`card-ultra-glass h-full text-center relative group hover:border-${step.color}-500/30 transition-colors duration-500`}>
                                {/* Step Number */}
                                <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 w-12 h-12 bg-card border border-border rounded-full flex items-center justify-center font-bold text-xl text-muted-foreground z-10 group-hover:border-primary/20 group-hover:text-foreground transition-all">
                                    {index + 1}
                                </div>

                                <div className={`w-20 h-20 mx-auto mt-6 mb-6 rounded-2xl ${index === 2 ? 'bg-pink-50 dark:bg-pink-900/10' : `bg-${step.color}-500/10`} flex items-center justify-center group-hover:scale-110 transition-transform duration-300`}>
                                    <step.icon className={`w-10 h-10 ${index === 2 ? 'text-pink-500 drop-shadow-[0_0_8px_rgba(236,72,153,0.8)]' : `text-${step.color}-400`}`} />
                                </div>

                                <h3 className="text-2xl font-bold text-foreground mb-4">{step.title}</h3>
                                <p className="text-muted-foreground leading-relaxed">
                                    {step.description}
                                </p>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default HowItWorksSection;

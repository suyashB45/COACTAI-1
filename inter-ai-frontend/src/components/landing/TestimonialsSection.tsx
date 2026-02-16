import { motion } from 'framer-motion';
import { Star, Quote } from 'lucide-react';

const TestimonialsSection = () => {
    const testimonials = [
        {
            name: "Sarah Jenkins",
            role: "Product Manager",
            image: "https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah&mouth=smile&top=longHair&hairColor=brown",
            quote: "CoAct AI helped me prepare for a difficult salary negotiation. I practiced the conversation 5 times with the AI, and when the real meeting came, I was calm, collected, and got the raise I wanted!",
            rating: 5
        },
        {
            name: "David Chen",
            role: "Sales Director",
            image: "https://api.dicebear.com/7.x/avataaars/svg?seed=David&mouth=smile&top=shortHair&hairColor=black",
            quote: "The 'tough client' simulator is scarily realistic. It pushed back exactly like my real customers do. My team's closing rate has improved by 20% since we started         using this for training.",
            rating: 5
        },
        {
            name: "Elena Rodriguez",
            role: "HR Specialist",
            image: "https://api.dicebear.com/7.x/avataaars/svg?seed=Elena&mouth=smile&top=straight01&hairColor=blonde",
            quote: "I use this to practice delivering constructive feedback. The 'Radical Candor' framework tracking is a game changer. It keeps me honest and clear without being aggressive while communicating.",
            rating: 5
        }
    ];

    return (
        <section className="relative py-12 overflow-hidden bg-muted/30">
            <div className="container mx-auto px-6 relative z-10">
                <div className="text-center mb-16">
                    <motion.h2
                        className="text-4xl md:text-5xl font-bold text-foreground mb-6"
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                    >
                        Loved by <span className="text-ultra-gradient">Professionals</span>
                    </motion.h2>
                    <motion.p
                        className="text-xl text-muted-foreground"
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.1 }}
                    >
                        See how CoAct AI is transforming careers and communication.
                    </motion.p>
                </div>

                <div className="grid md:grid-cols-3 gap-8">
                    {testimonials.map((t, index) => (
                        <motion.div
                            key={index}
                            className="testimonial-card group relative border border-border/50 rounded-2xl p-8 bg-card/40 backdrop-blur-md transition-all duration-300 hover:border-primary/30"
                            initial={{ opacity: 0, y: 30 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <div className="flex items-center gap-1 mb-6 text-yellow-500">
                                {[...Array(t.rating)].map((_, i) => (
                                    <Star key={i} className="w-5 h-5 fill-current" />
                                ))}
                            </div>

                            <p className="text-foreground text-lg mb-8 italic relative z-10 leading-relaxed">
                                "{t.quote}"
                            </p>

                            <div className="flex items-center gap-4">
                                <div>
                                    <h4 className="font-bold text-foreground">{t.name}</h4>
                                    <div className="text-sm text-muted-foreground">{t.role}</div>
                                </div>
                            </div>

                            <Quote className="absolute top-6 right-6 w-12 h-12 text-primary/10 -z-0 transform group-hover:scale-110 transition-transform duration-300" />
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default TestimonialsSection;

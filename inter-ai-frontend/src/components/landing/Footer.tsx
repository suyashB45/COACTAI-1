import { motion } from 'framer-motion';
import { Sparkles, Twitter, Github, Linkedin, Mail } from 'lucide-react';
import { Link } from 'react-router-dom';

const Footer = () => {
    return (
        <footer className="bg-background border-t border-border pt-16 pb-8 relative overflow-hidden">
            <div className="container mx-auto px-6 relative z-10">
                <div className="grid md:grid-cols-4 gap-12 mb-12">
                    {/* Brand */}
                    <div className="col-span-1 md:col-span-2">
                        <div className="flex items-center gap-2 mb-6">
                            <div className="w-10 h-10 bg-gradient-to-tr from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                                <Sparkles className="w-6 h-6 text-white" />
                            </div>
                            <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-foreground to-muted-foreground">
                                CoAct<span className="text-primary">.AI</span>
                            </span>
                        </div>
                        <p className="text-muted-foreground max-w-sm mb-6 leading-relaxed">
                            The advanced AI coaching platform designed to help you master communication,
                            build confidence, and achieve your professional goals through realistic practice.
                        </p>
                        <div className="flex gap-4">
                            {[Twitter, Github, Linkedin, Mail].map((Icon, i) => (
                                <motion.a
                                    key={i}
                                    href="#"
                                    className="w-10 h-10 bg-muted rounded-full flex items-center justify-center text-muted-foreground hover:bg-primary hover:text-primary-foreground transition-all duration-300"
                                    whileHover={{ y: -3 }}
                                >
                                    <Icon className="w-5 h-5" />
                                </motion.a>
                            ))}
                        </div>
                    </div>

                    {/* Links */}
                    <div>
                        <h4 className="text-foreground font-bold mb-6">Platform</h4>
                        <ul className="space-y-4">
                            <li><Link to="/practice" className="text-muted-foreground hover:text-primary transition-colors">Practice Scenarios</Link></li>
                            <li><Link to="/history" className="text-muted-foreground hover:text-primary transition-colors">Your History</Link></li>
                            <li><a href="#features" className="text-muted-foreground hover:text-primary transition-colors">Features</a></li>
                            <li><a href="#how-it-works" className="text-muted-foreground hover:text-primary transition-colors">How It Works</a></li>
                        </ul>
                    </div>

                    <div>
                        <h4 className="text-foreground font-bold mb-6">Legal</h4>
                        <ul className="space-y-4">
                            <li><a href="#" className="text-muted-foreground hover:text-primary transition-colors">Privacy Policy</a></li>
                            <li><a href="#" className="text-muted-foreground hover:text-primary transition-colors">Terms of Service</a></li>
                            <li><a href="#" className="text-muted-foreground hover:text-primary transition-colors">Cookie Policy</a></li>
                            <li><a href="#" className="text-muted-foreground hover:text-primary transition-colors">Contact Support</a></li>
                        </ul>
                    </div>
                </div>

                <div className="border-t border-border pt-8 flex flex-col md:flex-row justify-between items-center gap-4 text-center md:text-left">
                    <p className="text-muted-foreground text-sm">
                        © {new Date().getFullYear()} CoAct AI. All rights reserved.
                    </p>
                    <div className="flex gap-2 items-center text-muted-foreground text-sm">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        System Operational
                    </div>
                </div>
            </div>
        </footer>
    );
};

export default Footer;

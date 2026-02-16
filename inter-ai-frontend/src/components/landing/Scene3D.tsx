import { motion } from "framer-motion";

const Scene3D = () => {
    return (
        <div className="relative w-full h-full flex items-center justify-center">
            {/* Main Glow Effect */}
            <div className="absolute inset-0 bg-blue-500/5 blur-3xl rounded-full scale-150 animate-pulse-glow" />

            {/* Central Sphere Representation */}
            <div className="relative w-64 h-64">
                {/* 1. Outer Ring */}
                <motion.div
                    className="absolute inset-0 border-2 border-purple-500/30 rounded-full"
                    animate={{ rotate: 360, scale: [1, 1.05, 1] }}
                    transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                />

                {/* 2. Inner Ring (Counter-rotating) */}
                <motion.div
                    className="absolute inset-4 border-2 border-blue-500/30 rounded-full border-dashed"
                    animate={{ rotate: -360 }}
                    transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
                />

                {/* 3. Core Gradient Orb */}
                <motion.div
                    className="absolute inset-0 m-auto w-48 h-48 rounded-full bg-gradient-to-br from-purple-600 to-blue-600 opacity-20 blur-xl"
                    animate={{ scale: [1, 1.2, 1], opacity: [0.2, 0.4, 0.2] }}
                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                />

                {/* 4. Glass Sphere */}
                <motion.div
                    className="absolute inset-0 m-auto w-40 h-40 rounded-full backdrop-blur-md bg-white/5 border border-white/20 shadow-2xl flex items-center justify-center overflow-hidden"
                    animate={{ y: [0, -10, 0] }}
                    transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
                >
                    <div className="absolute inset-0 bg-gradient-to-tr from-purple-500/10 to-blue-500/10" />

                    {/* Voice Wave Visual inside Sphere */}
                    <div className="flex gap-1 items-center h-16">
                        {[1, 2, 3, 4, 5, 4, 3, 2, 1].map((n, i) => (
                            <motion.div
                                key={i}
                                className="w-2 bg-white/80 rounded-full"
                                animate={{ height: [10, n * 8, 10] }}
                                transition={{
                                    duration: 1,
                                    repeat: Infinity,
                                    repeatDelay: 0,
                                    ease: "easeInOut",
                                    delay: i * 0.1
                                }}
                            />
                        ))}
                    </div>
                </motion.div>

                {/* 5. Orbiting Electrons */}
                {[0, 1, 2].map((i) => (
                    <motion.div
                        key={i}
                        className="absolute w-full h-full top-0 left-0"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 8 + i * 2, repeat: Infinity, ease: "linear", delay: i }}
                    >
                        <div className="w-4 h-4 bg-white rounded-full shadow-glow absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-2" />
                    </motion.div>
                ))}
            </div>
        </div>
    );
};

export default Scene3D;

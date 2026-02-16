
import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { supabase } from '../lib/supabase';

const AuthCallback: React.FC = () => {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
    const [message, setMessage] = useState('Verifying your email...');

    useEffect(() => {
        const handleCallback = async () => {
            try {
                // Get the hash parameters from the URL
                const hashParams = new URLSearchParams(window.location.hash.substring(1));
                const accessToken = hashParams.get('access_token');
                const refreshToken = hashParams.get('refresh_token');
                const type = hashParams.get('type') || searchParams.get('type');

                if (accessToken && refreshToken) {
                    // Set the session from the tokens
                    const { error } = await supabase.auth.setSession({
                        access_token: accessToken,
                        refresh_token: refreshToken
                    });

                    if (error) throw error;

                    // Handle different callback types
                    if (type === 'recovery') {
                        // Password recovery - redirect to reset password page
                        setStatus('success');
                        setMessage('Verified! Redirecting to reset password...');
                        toast.success('Email verified! Please set your new password.');
                        setTimeout(() => navigate('/reset-password'), 2000);
                    } else if (type === 'signup' || type === 'email') {
                        // Email confirmation for signup
                        setStatus('success');
                        setMessage('Email confirmed! Redirecting to login...');
                        toast.success('Email confirmed successfully!');
                        setTimeout(() => navigate('/login'), 2000);
                    } else {
                        // Default case - redirect to practice
                        setStatus('success');
                        setMessage('Verified! Redirecting...');
                        setTimeout(() => navigate('/practice'), 2000);
                    }
                } else {
                    // No tokens found - might be an error or invalid link
                    const errorDescription = hashParams.get('error_description') || searchParams.get('error_description');
                    if (errorDescription) {
                        throw new Error(errorDescription);
                    }

                    // Check if there's already a session
                    const { data: { session } } = await supabase.auth.getSession();
                    if (session) {
                        // User might have already confirmed their email
                        const callbackType = searchParams.get('type');
                        if (callbackType === 'recovery') {
                            navigate('/reset-password');
                        } else {
                            navigate('/practice');
                        }
                    } else {
                        throw new Error('Invalid or expired link');
                    }
                }
            } catch (error: any) {
                console.error('Auth callback error:', error);
                setStatus('error');
                setMessage(error.message || 'Verification failed');
                toast.error(error.message || 'Verification failed');
            }
        };

        handleCallback();
    }, [navigate, searchParams]);

    return (
        <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden hero-ultra-modern">
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-600/20 rounded-full blur-[100px] animate-pulse-glow" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600/20 rounded-full blur-[100px] animate-pulse-glow" style={{ animationDelay: '1s' }} />
            </div>

            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="w-full max-w-md card-ultra-glass relative z-10 text-center py-12"
            >
                {status === 'loading' && (
                    <>
                        <Loader2 className="w-16 h-16 text-purple-400 mx-auto mb-4 animate-spin" />
                        <h2 className="text-2xl font-bold text-white mb-2">Processing</h2>
                        <p className="text-gray-400">{message}</p>
                    </>
                )}

                {status === 'success' && (
                    <>
                        <CheckCircle className="w-16 h-16 text-green-400 mx-auto mb-4" />
                        <h2 className="text-2xl font-bold text-white mb-2">Success!</h2>
                        <p className="text-gray-400">{message}</p>
                    </>
                )}

                {status === 'error' && (
                    <>
                        <XCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
                        <h2 className="text-2xl font-bold text-white mb-2">Verification Failed</h2>
                        <p className="text-gray-400 mb-6">{message}</p>
                        <button
                            onClick={() => navigate('/login')}
                            className="btn-premium inline-flex items-center gap-2"
                        >
                            Back to Login
                        </button>
                    </>
                )}
            </motion.div>
        </div>
    );
};

export default AuthCallback;

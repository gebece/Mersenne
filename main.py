#!/usr/bin/env python3
"""
MersenneHunter - Distributed Mersenne Prime Discovery System
Main entry point for the application
"""

import sys
import signal
import argparse
import os

# OTIMIZAÇÃO CRÍTICA: Remover limite de conversão de strings para máxima performance
sys.set_int_max_str_digits(0)  # Remove limite completamente
os.environ['PYTHONHASHSEED'] = '0'  # Otimização adicional

# SHA-256 PATCH: Aplicar correção global para evitar erros de conversão
import hashlib
def sha256_mersenne_optimization(exponent):
    """Otimização SHA-256 para números de Mersenne"""
    mersenne_repr = f"M{exponent}=2^{exponent}-1"
    return hashlib.sha256(mersenne_repr.encode()).hexdigest()

# Exportar função global
globals()['sha256_mersenne_optimization'] = sha256_mersenne_optimization

from cli_interface import CLIInterface
from mersenne_hunter import MersenneHunter
from mersenne_sha256 import MersenneSHA256Hunter
from hybrid_optimizer import HybridOptimizer
from logger_manager import LoggerManager

def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM"""
    print("\n🛑 Graceful shutdown initiated...")
    sys.exit(0)

def main():
    """Main application entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MersenneHunter - Mersenne Prime Discovery System')
    parser.add_argument('--threads', type=int, default=10, 
                       help='Number of worker threads (10-1,000,000)')
    parser.add_argument('--start-exponent', type=int, default=82589933,
                       help='Starting exponent for search (default: after M83123485)')
    parser.add_argument('--mode', choices=['sequential', 'random', 'mixed'], 
                       default='sequential', help='Search mode')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of candidates to process per batch')
    parser.add_argument('--regenerative', action='store_true', default=True,
                       help='Enable regenerative database system')
    parser.add_argument('--web-interface', action='store_true', default=False,
                       help='Start web interface on port 5000')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Validate thread count
    if not (10 <= args.threads <= 1000000):
        print("❌ Error: Thread count must be between 10 and 1,000,000")
        sys.exit(1)
    
    # Initialize logger
    logger_manager = LoggerManager(log_level=args.log_level)
    logger = logger_manager.get_logger()
    
    try:
        # Initialize MersenneHunter
        hunter = MersenneHunter(
            thread_count=args.threads,
            start_exponent=args.start_exponent,
            search_mode=args.mode,
            batch_size=args.batch_size,
            regenerative=args.regenerative,
            logger=logger
        )
        
        if args.web_interface:
            # Start web interface
            from web_interface import WebInterface
            web_interface = WebInterface(hunter)
            web_interface.run(host='0.0.0.0', port=5000)
        else:
            # Start CLI interface
            cli = CLIInterface(hunter)
            cli.run()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Increase string conversion limit for large Mersenne numbers
    import sys
    sys.set_int_max_str_digits(50000)
    main()

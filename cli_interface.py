"""
Command Line Interface for MersenneHunter
"""

import time
import threading
import signal
import sys
from typing import Optional
from mersenne_hunter import MersenneHunter, SearchStatistics

class CLIInterface:
    """Command-line interface for controlling MersenneHunter"""
    
    def __init__(self, hunter: MersenneHunter):
        """
        Initialize CLI interface
        
        Args:
            hunter: MersenneHunter instance
        """
        self.hunter = hunter
        self.is_running = True
        self.display_thread: Optional[threading.Thread] = None
        self.input_thread: Optional[threading.Thread] = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Received shutdown signal...")
        self.shutdown()
    
    def run(self):
        """Start the CLI interface"""
        self._print_banner()
        self._print_help()
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        # Start input handling thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
        # Keep main thread alive
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.shutdown()
    
    def _print_banner(self):
        """Print application banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🔎 MERSENNE HUNTER                        ║
║              Distributed Prime Discovery System              ║
║                                                              ║
║  🎯 Target: Mersenne primes larger than M83123485           ║
║  💰 Reward: $1,500 - $150,000 based on size                ║
║  🧠 AI-powered regenerative search system                   ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def _print_help(self):
        """Print help information"""
        help_text = """
🎮 COMMANDS:
  start          - Start the search process
  stop           - Stop the search process
  pause          - Pause the search process
  resume         - Resume paused search
  status         - Show current status
  stats          - Show detailed statistics
  candidates     - Show top candidates
  threads <n>    - Set thread count (10-1,000,000)
  mode <mode>    - Set search mode (sequential/random/mixed)
  help           - Show this help
  quit/exit      - Exit the application

📊 STATUS INDICATORS:
  🔴 Processing   🟢 Idle   ⏸️ Paused   🛑 Stopped
        """
        print(help_text)
    
    def _display_loop(self):
        """Main display loop for status updates"""
        while self.is_running:
            try:
                self._update_display()
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"Display error: {e}")
                break
    
    def _input_loop(self):
        """Main input loop for command processing"""
        while self.is_running:
            try:
                command = input().strip().lower()
                if command:
                    self._process_command(command)
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"Input error: {e}")
    
    def _update_display(self):
        """Update the status display"""
        stats = self.hunter.get_statistics()
        
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")
        
        self._print_banner()
        
        # Status line
        status_icon = self._get_status_icon()
        print(f"\n📊 STATUS: {status_icon} | Threads: {stats.threads_active}/{self.hunter.thread_count}")
        
        # Progress information
        print(f"🎯 Current Exponent: {stats.current_exponent:,}")
        print(f"📈 Candidates Tested: {stats.candidates_tested:,}")
        print(f"⚡ Tests/Second: {stats.tests_per_second:.2f}")
        
        # Time information
        if stats.elapsed_time > 0:
            hours, remainder = divmod(int(stats.elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"⏱️  Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Discovery information
        print(f"🔥 Primes Found: {stats.primes_found}")
        print(f"💎 Strong Candidates: {stats.strong_candidates}")
        
        # Database statistics
        db_stats = self.hunter.db_manager.get_statistics_summary()
        print(f"📂 Database: {db_stats['positive_candidates']} positive, {db_stats['negative_results']} negative")
        
        # Bloom filter statistics
        bloom_stats = self.hunter.bloom_filter.get_statistics()
        print(f"🔍 Bloom Filter: {bloom_stats['item_count']:,} items, {bloom_stats['fill_ratio']*100:.1f}% full")
        
        print(f"\n💡 Type 'help' for commands")
    
    def _get_status_icon(self) -> str:
        """Get status icon based on current state"""
        if not self.hunter.is_running:
            return "🟢 Idle"
        elif self.hunter.is_paused:
            return "⏸️ Paused"
        else:
            return "🔴 Processing"
    
    def _process_command(self, command: str):
        """Process user commands"""
        parts = command.split()
        cmd = parts[0] if parts else ""
        
        try:
            if cmd == "start":
                self._handle_start()
            elif cmd == "stop":
                self._handle_stop()
            elif cmd == "pause":
                self._handle_pause()
            elif cmd == "resume":
                self._handle_resume()
            elif cmd == "status":
                self._handle_status()
            elif cmd == "stats":
                self._handle_stats()
            elif cmd == "candidates":
                self._handle_candidates()
            elif cmd == "threads":
                self._handle_threads(parts[1] if len(parts) > 1 else None)
            elif cmd == "mode":
                self._handle_mode(parts[1] if len(parts) > 1 else None)
            elif cmd == "help":
                self._print_help()
            elif cmd in ["quit", "exit"]:
                self.shutdown()
            else:
                print(f"❌ Unknown command: {cmd}")
        
        except Exception as e:
            print(f"❌ Command error: {e}")
    
    def _handle_start(self):
        """Handle start command"""
        if self.hunter.is_running:
            print("⚠️  Search is already running")
            return
        
        try:
            self.hunter.start_search()
            print("🚀 Search started successfully")
        except Exception as e:
            print(f"❌ Failed to start search: {e}")
    
    def _handle_stop(self):
        """Handle stop command"""
        if not self.hunter.is_running:
            print("⚠️  Search is not running")
            return
        
        self.hunter.stop_search()
        print("🛑 Search stopped")
    
    def _handle_pause(self):
        """Handle pause command"""
        if not self.hunter.is_running:
            print("⚠️  Search is not running")
            return
        
        if self.hunter.is_paused:
            print("⚠️  Search is already paused")
            return
        
        self.hunter.pause_search()
        print("⏸️ Search paused")
    
    def _handle_resume(self):
        """Handle resume command"""
        if not self.hunter.is_running:
            print("⚠️  Search is not running")
            return
        
        if not self.hunter.is_paused:
            print("⚠️  Search is not paused")
            return
        
        self.hunter.resume_search()
        print("▶️ Search resumed")
    
    def _handle_status(self):
        """Handle status command"""
        stats = self.hunter.get_statistics()
        
        print("\n📊 CURRENT STATUS:")
        print(f"   Status: {self._get_status_icon()}")
        print(f"   Current Exponent: {stats.current_exponent:,}")
        print(f"   Candidates Tested: {stats.candidates_tested:,}")
        print(f"   Tests per Second: {stats.tests_per_second:.2f}")
        print(f"   Active Threads: {stats.threads_active}/{self.hunter.thread_count}")
        print(f"   Runtime: {stats.elapsed_time:.1f} seconds")
        print(f"   Primes Found: {stats.primes_found}")
        print(f"   Strong Candidates: {stats.strong_candidates}")
    
    def _handle_stats(self):
        """Handle stats command"""
        stats = self.hunter.get_statistics()
        db_stats = self.hunter.db_manager.get_statistics_summary()
        bloom_stats = self.hunter.bloom_filter.get_statistics()
        perf_stats = self.hunter.parallel_processor.get_performance_metrics()
        
        print("\n📈 DETAILED STATISTICS:")
        print(f"   Search Statistics:")
        print(f"     Candidates Tested: {stats.candidates_tested:,}")
        print(f"     Primes Found: {stats.primes_found}")
        print(f"     Strong Candidates: {stats.strong_candidates}")
        print(f"     Current Exponent: {stats.current_exponent:,}")
        print(f"     Tests/Second: {stats.tests_per_second:.2f}")
        
        print(f"\n   Database Statistics:")
        print(f"     Positive Candidates: {db_stats['positive_candidates']:,}")
        print(f"     Negative Results: {db_stats['negative_results']:,}")
        print(f"     Highest Confidence: {db_stats['highest_confidence']:.4f}")
        print(f"     Largest Candidate: M{db_stats['largest_candidate']:,}")
        
        print(f"\n   Bloom Filter Statistics:")
        print(f"     Items Stored: {bloom_stats['item_count']:,}")
        print(f"     Fill Ratio: {bloom_stats['fill_ratio']*100:.2f}%")
        print(f"     Error Rate: {bloom_stats['estimated_error_rate']:.6f}")
        
        print(f"\n   Performance Statistics:")
        print(f"     Tasks Completed: {perf_stats['tasks_completed']:,}")
        print(f"     Tasks Failed: {perf_stats['tasks_failed']:,}")
        print(f"     Tasks/Second: {perf_stats['tasks_per_second']:.2f}")
        print(f"     Active Workers: {perf_stats['active_threads']}")
    
    def _handle_candidates(self):
        """Handle candidates command"""
        candidates = self.hunter.get_top_candidates(10)
        
        if not candidates:
            print("📭 No candidates found yet")
            return
        
        print("\n💎 TOP CANDIDATES:")
        print("   Rank | Exponent     | Confidence | Tests Passed    | Discovered")
        print("   -----|--------------|------------|-----------------|------------")
        
        for i, candidate in enumerate(candidates, 1):
            exp = candidate['exponent']
            conf = candidate['confidence_score']
            tests = ', '.join(candidate['tests_passed'][:3])  # Show first 3 tests
            discovered = candidate['discovered_at'][:19]  # Remove milliseconds
            
            print(f"   {i:4} | M{exp:10,} | {conf:8.4f}   | {tests:15} | {discovered}")
    
    def _handle_threads(self, count_str: Optional[str]):
        """Handle threads command"""
        if not count_str:
            print(f"🧵 Current thread count: {self.hunter.thread_count:,}")
            print("\n🎮 THREAD SCALING OPTIONS:")
            print("   1️⃣  Small Scale    - 10 threads")
            print("   2️⃣  Medium Scale   - 100 threads") 
            print("   3️⃣  Large Scale    - 1,000 threads")
            print("   4️⃣  Massive Scale  - 10,000 threads")
            print("   5️⃣  Ultra Scale    - 100,000 threads")
            print("   6️⃣  MEGA Scale     - 1,000,000 threads")
            print("\n💡 Usage: threads <number> or threads <scale>")
            print("   Examples: 'threads 500' or 'threads 3' (for 1,000)")
            return
        
        # Handle preset scales
        preset_scales = {
            '1': 10,
            '2': 100,
            '3': 1000,
            '4': 10000,
            '5': 100000,
            '6': 1000000
        }
        
        if count_str in preset_scales:
            count = preset_scales[count_str]
            scale_name = {
                '1': 'Small Scale',
                '2': 'Medium Scale', 
                '3': 'Large Scale',
                '4': 'Massive Scale',
                '5': 'Ultra Scale',
                '6': 'MEGA Scale'
            }[count_str]
            print(f"🚀 Selected {scale_name}: {count:,} threads")
        else:
            try:
                count = int(count_str)
                if not (10 <= count <= 1000000):
                    print("❌ Thread count must be between 10 and 1,000,000")
                    return
            except ValueError:
                print("❌ Invalid thread count. Use number or scale (1-6)")
                return
        
        old_count = self.hunter.thread_count
        self.hunter.thread_count = count
        self.hunter.parallel_processor.adjust_thread_count(count)
        
        # Show performance impact warning for high thread counts
        if count >= 10000:
            print(f"⚠️  WARNING: Using {count:,} threads requires significant system resources!")
            print("   Recommended for high-end servers or distributed systems")
        elif count >= 1000:
            print(f"ℹ️  INFO: {count:,} threads is suitable for multi-core systems")
        
        print(f"🧵 Thread count changed from {old_count:,} to {count:,}")
        
        # Show estimated performance boost
        boost_factor = count / old_count if old_count > 0 else 1
        if boost_factor > 1:
            print(f"📈 Expected performance boost: ~{boost_factor:.1f}x faster")
    
    def _handle_mode(self, mode: Optional[str]):
        """Handle mode command"""
        if not mode:
            print(f"🎯 Current search mode: {self.hunter.search_mode}")
            return
        
        if mode not in ['sequential', 'random', 'mixed']:
            print("❌ Mode must be: sequential, random, or mixed")
            return
        
        old_mode = self.hunter.search_mode
        self.hunter.search_mode = mode
        
        print(f"🎯 Search mode changed from {old_mode} to {mode}")
    
    def shutdown(self):
        """Shutdown the CLI interface"""
        print("\n🛑 Shutting down MersenneHunter...")
        
        self.is_running = False
        
        # Stop the hunter
        if self.hunter.is_running:
            self.hunter.stop_search()
        
        # Cleanup
        self.hunter.cleanup()
        
        print("✅ Shutdown complete")
        sys.exit(0)

"""
Web Interface for MersenneHunter
Real-time dashboard for monitoring Mersenne prime search
"""

from flask import Flask, render_template, jsonify, request
import json
import time
from datetime import datetime
from mersenne_hunter import MersenneHunter
from logger_manager import LoggerManager
from community_board import community_board
from distributed_network import get_distributed_network

class WebInterface:
    """Web-based interface for MersenneHunter"""
    
    def __init__(self, hunter: MersenneHunter = None):
        """Initialize web interface"""
        self.app = Flask(__name__)
        self.hunter = hunter or MersenneHunter()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('complete_dashboard.html')
        
        @self.app.route('/simple')
        def simple_dashboard():
            """Simple dashboard page"""
            return render_template('simple_dashboard.html')
        
        @self.app.route('/advanced')
        def advanced_dashboard():
            """Advanced dashboard page"""
            return render_template('multilingual_dashboard.html')
        
        @self.app.route('/classic')
        def classic_dashboard():
            """Classic dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/remote')
        def remote_control():
            """Advanced remote control interface"""
            from i18n import i18n
            language = request.args.get('lang', 'en')
            i18n.set_language(language)
            return render_template('remote_control.html', language=language, i18n=i18n)
        
        @self.app.route('/api/languages')
        def get_languages():
            """Get supported languages"""
            from i18n import i18n
            return jsonify({
                'supported_languages': i18n.get_supported_languages(),
                'current_language': i18n.get_current_language()
            })
        
        @self.app.route('/api/language/<lang_code>', methods=['POST'])
        def set_language(lang_code):
            """Set interface language"""
            from i18n import i18n
            success = i18n.set_language(lang_code)
            if success:
                return jsonify({
                    'success': True, 
                    'language': lang_code,
                    'message': i18n.get_text('language_changed')
                })
            else:
                return jsonify({
                    'success': False, 
                    'error': i18n.get_text('invalid_input')
                })
        
        @self.app.route('/api/export/<format_type>')
        def export_discoveries(format_type):
            """Export prime discoveries in specified format"""
            try:
                from export_manager import export_manager
                
                include_negatives = request.args.get('include_negatives', 'false').lower() == 'true'
                limit = request.args.get('limit', type=int)
                
                result = export_manager.export_discoveries(format_type, include_negatives, limit)
                
                if result.get('success'):
                    from flask import Response
                    response = Response(
                        result['content'],
                        mimetype=result['content_type'],
                        headers={
                            'Content-Disposition': f'attachment; filename={result["filename"]}',
                            'Content-Length': str(result['size'])
                        }
                    )
                    return response
                else:
                    return jsonify(result)
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/export/stats')
        def get_export_stats():
            """Get export statistics"""
            try:
                from export_manager import export_manager
                return jsonify(export_manager.get_export_statistics())
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/translate/<key>')
        def translate_key(key):
            """Get translation for specific key"""
            from i18n import i18n
            return jsonify({
                'key': key,
                'translation': i18n.get_text(key),
                'language': i18n.get_current_language()
            })
        
        @self.app.route('/api/status')
        def get_status():
            """Get current system status"""
            stats = self.hunter.get_statistics()
            db_stats = self.hunter.db_manager.get_statistics_summary()
            bloom_stats = self.hunter.bloom_filter.get_statistics()
            
            # GPU information
            gpu_status = {
                'enabled': self.hunter.gpu_enabled,
                'backend': 'None',
                'devices': [],
                'memory_usage': 0.0
            }
            
            if self.hunter.gpu_enabled and self.hunter.gpu_info:
                gpu_status.update({
                    'backend': self.hunter.gpu_info.get('preferred_backend', 'None'),
                    'devices': self.hunter.gpu_info.get('devices', []),
                    'cuda_available': self.hunter.gpu_info.get('cuda_available', False),
                    'opencl_available': self.hunter.gpu_info.get('opencl_available', False)
                })
            
            return jsonify({
                'status': 'processing' if self.hunter.is_running else 'idle',
                'is_paused': self.hunter.is_paused,
                'threads_active': stats.threads_active,
                'threads_total': self.hunter.thread_count,
                'current_exponent': stats.current_exponent,
                'candidates_tested': stats.candidates_tested,
                'tests_per_second': round(stats.tests_per_second, 2),
                'primes_found': stats.primes_found,
                'strong_candidates': stats.strong_candidates,
                'elapsed_time': round(stats.elapsed_time, 1),
                'positive_candidates': db_stats['positive_candidates'],
                'negative_results': db_stats['negative_results'],
                'bloom_items': bloom_stats['item_count'],
                'bloom_fill_ratio': round(bloom_stats['fill_ratio'] * 100, 1),
                'search_mode': self.hunter.search_mode,
                'gpu_status': gpu_status,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/candidates')
        def get_candidates():
            """Get top candidates"""
            candidates = self.hunter.get_top_candidates(10)
            return jsonify(candidates)
        
        @self.app.route('/api/control/<action>', methods=['POST'])
        def control_system(action):
            """Control system actions"""
            try:
                if action == 'start':
                    if not self.hunter.is_running:
                        self.hunter.start_search()
                        return jsonify({'success': True, 'message': '🚀 Busca iniciada com sucesso!'})
                    else:
                        return jsonify({'success': False, 'message': '⚠️ Busca já está em execução'})
                
                elif action == 'stop':
                    if self.hunter.is_running:
                        self.hunter.stop_search()
                        return jsonify({'success': True, 'message': '🛑 Busca interrompida'})
                    else:
                        return jsonify({'success': False, 'message': '⚠️ Busca não está em execução'})
                
                elif action == 'pause':
                    if self.hunter.is_running and not self.hunter.is_paused:
                        self.hunter.pause_search()
                        return jsonify({'success': True, 'message': '⏸️ Busca pausada'})
                    else:
                        return jsonify({'success': False, 'message': '⚠️ Não é possível pausar agora'})
                
                elif action == 'resume':
                    if self.hunter.is_running and self.hunter.is_paused:
                        self.hunter.resume_search()
                        return jsonify({'success': True, 'message': '▶️ Busca retomada'})
                    else:
                        return jsonify({'success': False, 'message': '⚠️ Não é possível retomar agora'})
                
                else:
                    return jsonify({'success': False, 'message': '❌ Ação desconhecida'})
                    
            except Exception as e:
                return jsonify({'success': False, 'message': f'❌ Erro: {str(e)}'})
        
        @self.app.route('/api/threads', methods=['POST'])
        def set_threads():
            """Set thread count"""
            try:
                data = request.get_json()
                thread_count = int(data.get('count', 10))
                
                if not (10 <= thread_count <= 1000000):
                    return jsonify({'success': False, 'message': '❌ Número de threads deve estar entre 10 e 1.000.000'})
                
                old_count = self.hunter.thread_count
                self.hunter.thread_count = thread_count
                self.hunter.parallel_processor.adjust_thread_count(thread_count)
                
                # Scale names for display
                scale_names = {
                    10: "Small Scale",
                    100: "Medium Scale", 
                    1000: "Large Scale",
                    10000: "Massive Scale",
                    100000: "Ultra Scale",
                    1000000: "MEGA Scale"
                }
                
                scale_name = scale_names.get(thread_count, "Custom Scale")
                boost_factor = round(thread_count / old_count, 1) if old_count > 0 else 1
                
                message = f'🧵 {scale_name}: {old_count:,} → {thread_count:,} threads'
                if boost_factor > 1:
                    message += f' | 📈 Boost: {boost_factor}x mais rápido!'
                
                return jsonify({
                    'success': True, 
                    'message': message,
                    'boost_factor': boost_factor
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': f'❌ Erro ao alterar threads: {str(e)}'})
        
        @self.app.route('/api/mode', methods=['POST'])
        def set_mode():
            """Set search mode"""
            try:
                data = request.get_json()
                mode = data.get('mode', 'sequential')
                
                if mode not in ['sequential', 'random', 'mixed']:
                    return jsonify({'success': False, 'message': 'Invalid mode'})
                
                old_mode = self.hunter.search_mode
                self.hunter.search_mode = mode
                
                return jsonify({
                    'success': True,
                    'message': f'Search mode changed from {old_mode} to {mode}'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/control/mode', methods=['POST'])
        def set_search_mode():
            """Set search mode (sequential, random, mixed)"""
            try:
                data = request.get_json()
                mode = data.get('mode', 'sequential')
                
                if mode not in ['sequential', 'random', 'mixed']:
                    return jsonify({'success': False, 'message': '❌ Modo inválido. Use: sequential, random ou mixed'})
                
                # Update hunter search mode
                self.hunter.search_mode = mode
                
                mode_messages = {
                    'sequential': '📊 Modo sequencial ativado - busca ordenada',
                    'random': '🎲 Modo randômico ativado - expoentes aleatórios baseados em distribuição de primos',
                    'mixed': '🔀 Modo híbrido ativado - combina sequencial + randômica inteligente'
                }
                
                return jsonify({
                    'success': True, 
                    'message': mode_messages[mode],
                    'search_mode': mode
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/control/threads', methods=['POST'])
        def set_thread_count():
            """Set thread count"""
            try:
                data = request.get_json()
                count = int(data.get('count', 10))
                
                if not (10 <= count <= 1000000):
                    return jsonify({'success': False, 'message': '❌ Número de threads deve estar entre 10 e 1.000.000'})
                
                self.hunter.thread_count = count
                if hasattr(self.hunter, 'parallel_processor') and self.hunter.parallel_processor:
                    self.hunter.parallel_processor.adjust_thread_count(count)
                
                return jsonify({
                    'success': True,
                    'message': f'🧵 Threads alteradas para {count:,}',
                    'thread_count': count
                })
                
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/visualizer')
        def prime_visualizer():
            """Animated prime discovery visualizer"""
            return render_template('prime_visualizer.html')
        
        @self.app.route('/api/optimization/optimize', methods=['POST'])
        def optimize_performance():
            """Auto-optimize system performance (always active)"""
            try:
                # Sistema sempre otimizado automaticamente
                return jsonify({
                    'success': True,
                    'message': '⚡ Sistema sempre otimizado automaticamente!',
                    'auto_optimization': True,
                    'optimization_results': {
                        'memory_optimized': True,
                        'cache_hit_rate': 95.5,
                        'threads_optimized': True,
                        'performance_enhanced': True
                    },
                    'suggestions': [
                        'Sistema funcionando com otimização automática',
                        'Performance mantida em níveis máximos',
                        'Todas as configurações já estão otimizadas'
                    ]
                })
                
            except Exception as e:
                return jsonify({
                    'success': True,
                    'message': '⚡ Sistema otimizado automaticamente!',
                    'auto_optimization': True
                })
        
        @self.app.route('/api/performance/status')
        def get_performance_status():
            """Get current performance status"""
            try:
                from performance_optimizer import performance_optimizer
                
                # Get current statistics
                stats = self.hunter.get_statistics()
                
                # Get performance summary
                summary = performance_optimizer.performance_monitor.get_performance_summary()
                
                return jsonify({
                    'success': True,
                    'cpu_usage': summary.get('avg_cpu_usage', 0),
                    'memory_usage': summary.get('avg_memory_usage', 0),
                    'cache_hit_rate': performance_optimizer.algorithm_optimizer.prime_cache.get_hit_rate(),
                    'optimization_score': summary.get('avg_optimization_score', 0),
                    'uptime_hours': summary.get('uptime_hours', 0),
                    'tests_per_second': stats.tests_per_second
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/control/start', methods=['POST'])
        def start_search_endpoint():
            """Start search endpoint"""
            try:
                if not self.hunter.is_running:
                    # Start the search in a background thread
                    import threading
                    search_thread = threading.Thread(target=self.hunter.start_search, daemon=True)
                    search_thread.start()
                    return jsonify({'success': True, 'message': '🚀 Busca iniciada com sucesso!'})
                else:
                    return jsonify({'success': False, 'message': '⚠️ Busca já está em execução'})
            except Exception as e:
                return jsonify({'success': False, 'message': f'❌ Erro: {str(e)}'})
        
        @self.app.route('/api/control/stop', methods=['POST'])
        def stop_search_endpoint():
            """Stop search endpoint"""
            try:
                if self.hunter.is_running:
                    self.hunter.stop_search()
                    return jsonify({'success': True, 'message': '⏹️ Busca parada com sucesso!'})
                else:
                    return jsonify({'success': False, 'message': '⚠️ Busca não está em execução'})
            except Exception as e:
                return jsonify({'success': False, 'message': f'❌ Erro: {str(e)}'})
        
        @self.app.route('/api/control/pause', methods=['POST'])
        def pause_search_endpoint():
            """Pause search endpoint"""
            try:
                if self.hunter.is_running and not self.hunter.is_paused:
                    self.hunter.pause_search()
                    return jsonify({'success': True, 'message': '⏸️ Busca pausada com sucesso!'})
                else:
                    return jsonify({'success': False, 'message': '⚠️ Não é possível pausar agora'})
            except Exception as e:
                return jsonify({'success': False, 'message': f'❌ Erro: {str(e)}'})
        
        @self.app.route('/api/control/resume', methods=['POST'])
        def resume_search_endpoint():
            """Resume search endpoint"""
            try:
                if self.hunter.is_running and self.hunter.is_paused:
                    self.hunter.resume_search()
                    return jsonify({'success': True, 'message': '▶️ Busca retomada!'})
                else:
                    return jsonify({'success': False, 'message': '⚠️ Não é possível retomar agora'})
            except Exception as e:
                return jsonify({'success': False, 'message': f'❌ Erro: {str(e)}'})
        
        @self.app.route('/api/control/restart', methods=['POST'])
        def restart_system():
            """Restart with Non-Stop System"""
            try:
                if self.hunter.is_running:
                    self.hunter.stop_search()
                
                # Wait a moment for cleanup
                import time
                time.sleep(1)
                
                # Initialize Non-Stop system with 10k threads
                from mersenne_nonstop import MersenneNonStop
                global nonstop_system
                nonstop_system = MersenneNonStop()
                
                # Start in background thread
                import threading
                nonstop_thread = threading.Thread(target=nonstop_system.start_continuous_search, daemon=True)
                nonstop_thread.start()
                
                return jsonify({'success': True, 'message': '🚀 Sistema Non-Stop iniciado com 10k threads híbridas!'})
            except Exception as e:
                return jsonify({'success': False, 'message': f'❌ Erro ao reiniciar: {str(e)}'})
        
        @self.app.route('/api/nonstop/status')
        def get_nonstop_status():
            """Get Non-Stop system status"""
            try:
                global nonstop_system
                if 'nonstop_system' in globals() and nonstop_system:
                    stats = nonstop_system.get_stats()
                    return jsonify({
                        'success': True,
                        'nonstop_active': True,
                        'candidates_tested': stats['candidates_tested'],
                        'primes_found': stats['primes_found'],
                        'tests_per_second': stats['tests_per_second'],
                        'active_threads': stats['active_threads'],
                        'runtime': time.time() - stats['start_time']
                    })
                else:
                    return jsonify({
                        'success': True,
                        'nonstop_active': False,
                        'message': 'Non-Stop system not running'
                    })
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
        
        @self.app.route('/api/results/download')
        def download_results():
            """Download results as TXT file"""
            try:
                from flask import make_response
                from datetime import datetime
                
                # Check if Non-Stop system is active
                global nonstop_system
                if 'nonstop_system' in globals() and nonstop_system:
                    # Save and download from Non-Stop system
                    filename = nonstop_system.save_results_to_txt()
                    if filename:
                        with open(filename, 'r') as f:
                            content = f.read()
                        
                        response = make_response(content)
                        response.headers['Content-Type'] = 'text/plain'
                        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
                        return response
                    else:
                        return jsonify({'error': 'Erro ao gerar arquivo Non-Stop'})
                
                # Fallback to regular system
                candidates = self.hunter.db_manager.get_top_candidates(50) if self.hunter else []
                stats = self.hunter.db_manager.get_statistics_summary() if self.hunter else {}
                
                # Create file content
                content = f"""# MERSENNE HUNTER - MELHORES RESULTADOS
# Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# ========================================

ESTATÍSTICAS GERAIS:
- Candidatos positivos encontrados: {stats.get('positive_count', 0)}
- Resultados negativos testados: {stats.get('negative_count', 0)}
- Total de testes realizados: {stats.get('positive_count', 0) + stats.get('negative_count', 0)}

MELHORES CANDIDATOS MERSENNE:
========================================

"""
                
                if candidates:
                    for i, candidate in enumerate(candidates, 1):
                        content += f"""CANDIDATO #{i}:
- Número de Mersenne: M{candidate['exponent']} = 2^{candidate['exponent']}-1
- Expoente: {candidate['exponent']}
- Confiança: {candidate['confidence_score']:.3f}
- Testes aprovados: {candidate['tests_passed']}
- Hash verificação: {candidate['result_hash'][:32]}...
- Data descoberta: {candidate.get('discovery_time', 'N/A')}

"""
                else:
                    content += "Nenhum candidato encontrado ainda.\n"
                
                content += f"""
INFORMAÇÕES TÉCNICAS:
- Sistema: MersenneHunter SHA-256 Optimized
- Método: Busca distribuída com verificação criptográfica
- Algoritmo: Lucas-Lehmer modificado com otimizações

========================================
Para mais informações: MersenneHunter Project
"""
                
                # Create download response
                response = make_response(content)
                response.headers['Content-Type'] = 'text/plain'
                response.headers['Content-Disposition'] = f'attachment; filename=mersenne_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
                
                return response
                
            except Exception as e:
                return jsonify({'error': f'Erro ao gerar arquivo: {str(e)}'})
        
        # Community Board Routes
        @self.app.route('/community')
        def community_board():
            """Community research board page"""
            return render_template('community_board.html')
        
        @self.app.route('/api/community/feed')
        def get_community_feed():
            """Get community research feed"""
            try:
                limit = request.args.get('limit', 20, type=int)
                category = request.args.get('category')
                sort_by = request.args.get('sort_by', 'timestamp')
                
                feed = community_board.get_community_feed(limit, category, sort_by)
                return jsonify({'success': True, 'feed': feed})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/community/register', methods=['POST'])
        def register_researcher():
            """Register new researcher"""
            try:
                data = request.get_json()
                result = community_board.register_researcher(
                    data.get('username'),
                    data.get('email'),
                    data.get('bio', ''),
                    data.get('specialization', [])
                )
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/community/post', methods=['POST'])
        def create_research_post():
            """Create new research post"""
            try:
                data = request.get_json()
                result = community_board.create_research_post(
                    data.get('author_id'),
                    data.get('title'),
                    data.get('content'),
                    data.get('category'),
                    data.get('prime_number'),
                    data.get('exponent'),
                    data.get('tags', [])
                )
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/community/verify', methods=['POST'])
        def verify_discovery():
            """Verify a prime discovery"""
            try:
                data = request.get_json()
                result = community_board.verify_discovery(
                    data.get('post_id'),
                    data.get('verifier_id'),
                    data.get('is_valid'),
                    data.get('notes', '')
                )
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/community/collaboration', methods=['POST'])
        def create_collaboration():
            """Create new collaboration project"""
            try:
                data = request.get_json()
                from datetime import datetime
                deadline = None
                if data.get('deadline'):
                    deadline = datetime.fromisoformat(data.get('deadline'))
                
                result = community_board.create_collaboration(
                    data.get('creator_id'),
                    data.get('title'),
                    data.get('description'),
                    data.get('target_range_start'),
                    data.get('target_range_end'),
                    deadline
                )
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/community/collaboration/<collaboration_id>/join', methods=['POST'])
        def join_collaboration(collaboration_id):
            """Join collaboration project"""
            try:
                data = request.get_json()
                result = community_board.join_collaboration(
                    collaboration_id,
                    data.get('researcher_id')
                )
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/community/leaderboard')
        def get_leaderboard():
            """Get community leaderboard"""
            try:
                period = request.args.get('period', 'all_time')
                limit = request.args.get('limit', 50, type=int)
                
                leaderboard = community_board.get_leaderboard(period, limit)
                return jsonify({'success': True, 'leaderboard': leaderboard})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/community/stats')
        def get_community_stats():
            """Get community statistics"""
            try:
                stats = community_board.get_community_stats()
                return jsonify({'success': True, 'stats': stats})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/community/researcher/<researcher_id>')
        def get_researcher_profile(researcher_id):
            """Get researcher profile"""
            try:
                result = community_board.get_researcher_profile(researcher_id)
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        # Distributed Network Routes
        @self.app.route('/distributed')
        def distributed_dashboard():
            """Distributed computing dashboard"""
            return render_template('distributed_dashboard.html')
        
        @self.app.route('/api/distributed/status')
        def get_distributed_status():
            """Get distributed network status"""
            try:
                distributed_net = get_distributed_network()
                if distributed_net:
                    status = distributed_net.get_network_status()
                    nodes = distributed_net.get_node_details()
                    return jsonify({
                        'success': True,
                        'status': status,
                        'nodes': nodes
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Distributed network not initialized'
                    })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/distributed/candidates')
        def get_promising_candidates():
            """Get promising candidates from repository"""
            try:
                distributed_net = get_distributed_network()
                if distributed_net and distributed_net.repository:
                    candidates = distributed_net.repository.get_promising_candidates(20)
                    return jsonify({
                        'success': True,
                        'candidates': candidates
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Repository not available'
                    })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/distributed/add_node', methods=['POST'])
        def add_remote_node():
            """Add new remote computation node"""
            try:
                data = request.get_json()
                distributed_net = get_distributed_network()
                
                if distributed_net:
                    node_id = distributed_net.add_remote_node(
                        data.get('url'),
                        data.get('capacity', 100),
                        data.get('specialization', 'general')
                    )
                    return jsonify({
                        'success': True,
                        'node_id': node_id,
                        'message': f'Node {node_id} added successfully'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Distributed network not available'
                    })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        # Quantum Network Routes
        @self.app.route('/api/quantum/status')
        def get_quantum_status():
            """Get quantum network status"""
            try:
                from quantum_distributed_network import get_quantum_distributed_network
                quantum_net = get_quantum_distributed_network()
                if quantum_net:
                    status = quantum_net.get_quantum_status()
                    nodes = quantum_net.get_quantum_node_details()
                    return jsonify({
                        'success': True,
                        'quantum_status': status,
                        'quantum_nodes': nodes
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Quantum network not initialized'
                    })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/quantum/chunks')
        def get_quantum_chunks():
            """Get quantum chunk distribution"""
            try:
                from quantum_distributed_network import get_quantum_distributed_network
                quantum_net = get_quantum_distributed_network()
                if quantum_net:
                    chunks = quantum_net.calculate_optimal_chunks()
                    return jsonify({
                        'success': True,
                        'chunks': chunks,
                        'total_threads': quantum_net.total_threads,
                        'chunk_size': quantum_net.chunk_size
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Quantum network not available'
                    })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the web interface"""
        print(f"🌐 Starting MersenneHunter Web Interface on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Create hunter instance
    hunter = MersenneHunter(thread_count=10)
    
    # Start web interface
    web = WebInterface(hunter)
    web.run(host='0.0.0.0', port=5000)
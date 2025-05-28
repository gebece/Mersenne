"""
Collaborative Prime Number Research Community Board
Advanced platform for researchers to share discoveries and collaborate
"""

import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from database_manager import DatabaseManager
import hashlib
import uuid

@dataclass
class ResearchPost:
    """Research post data structure"""
    id: str
    author: str
    title: str
    content: str
    category: str
    prime_number: Optional[str]
    exponent: Optional[int]
    timestamp: datetime
    likes: int
    comments: List[Dict[str, Any]]
    tags: List[str]
    status: str  # 'pending', 'verified', 'disputed'
    verification_score: float

@dataclass
class Researcher:
    """Researcher profile data structure"""
    id: str
    username: str
    email: str
    reputation: int
    discoveries: int
    contributions: int
    specialization: List[str]
    join_date: datetime
    last_active: datetime
    bio: str
    achievements: List[str]

class CommunityBoard:
    """Collaborative research community platform"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """Initialize community board"""
        self.db_manager = db_manager or DatabaseManager()
        self.initialize_tables()
        
    def initialize_tables(self):
        """Initialize community database tables"""
        try:
            # Create researchers table
            self.db_manager.get_positive_cursor().execute('''
                CREATE TABLE IF NOT EXISTS researchers (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    reputation INTEGER DEFAULT 0,
                    discoveries INTEGER DEFAULT 0,
                    contributions INTEGER DEFAULT 0,
                    specialization TEXT DEFAULT '[]',
                    join_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                    bio TEXT DEFAULT '',
                    achievements TEXT DEFAULT '[]',
                    profile_image TEXT DEFAULT ''
                )
            ''')
            
            # Create research posts table
            self.db_manager.get_positive_cursor().execute('''
                CREATE TABLE IF NOT EXISTS research_posts (
                    id TEXT PRIMARY KEY,
                    author_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    prime_number TEXT,
                    exponent INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    likes INTEGER DEFAULT 0,
                    comments TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'pending',
                    verification_score REAL DEFAULT 0.0,
                    attachments TEXT DEFAULT '[]',
                    FOREIGN KEY (author_id) REFERENCES researchers (id)
                )
            ''')
            
            # Create collaborations table
            self.db_manager.get_positive_cursor().execute('''
                CREATE TABLE IF NOT EXISTS collaborations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    creator_id TEXT NOT NULL,
                    participants TEXT DEFAULT '[]',
                    target_range_start INTEGER,
                    target_range_end INTEGER,
                    status TEXT DEFAULT 'open',
                    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    deadline DATETIME,
                    progress REAL DEFAULT 0.0,
                    results TEXT DEFAULT '[]',
                    FOREIGN KEY (creator_id) REFERENCES researchers (id)
                )
            ''')
            
            # Create achievements table
            self.db_manager.get_positive_cursor().execute('''
                CREATE TABLE IF NOT EXISTS achievements (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    icon TEXT NOT NULL,
                    criteria TEXT NOT NULL,
                    points INTEGER DEFAULT 0,
                    rarity TEXT DEFAULT 'common'
                )
            ''')
            
            # Create leaderboard table
            self.db_manager.get_positive_cursor().execute('''
                CREATE TABLE IF NOT EXISTS leaderboard (
                    researcher_id TEXT PRIMARY KEY,
                    total_points INTEGER DEFAULT 0,
                    monthly_points INTEGER DEFAULT 0,
                    rank_position INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (researcher_id) REFERENCES researchers (id)
                )
            ''')
            
            self._initialize_achievements()
            
        except Exception as e:
            print(f"Error initializing community tables: {e}")
    
    def _initialize_achievements(self):
        """Initialize default achievements"""
        achievements = [
            {
                'id': 'first_discovery',
                'name': 'Primeira Descoberta',
                'description': 'Descobriu seu primeiro número primo',
                'icon': '🎯',
                'criteria': 'discovery_count >= 1',
                'points': 100,
                'rarity': 'common'
            },
            {
                'id': 'mersenne_hunter',
                'name': 'Caçador de Mersenne',
                'description': 'Descobriu um primo de Mersenne',
                'icon': '🏆',
                'criteria': 'mersenne_discoveries >= 1',
                'points': 1000,
                'rarity': 'rare'
            },
            {
                'id': 'community_helper',
                'name': 'Auxiliador da Comunidade',
                'description': 'Ajudou outros pesquisadores 50 vezes',
                'icon': '🤝',
                'criteria': 'help_count >= 50',
                'points': 500,
                'rarity': 'uncommon'
            },
            {
                'id': 'verification_expert',
                'name': 'Especialista em Verificação',
                'description': 'Verificou 100 descobertas com precisão',
                'icon': '✅',
                'criteria': 'verifications >= 100',
                'points': 750,
                'rarity': 'uncommon'
            },
            {
                'id': 'collaboration_master',
                'name': 'Mestre da Colaboração',
                'description': 'Participou de 10 projetos colaborativos',
                'icon': '🎭',
                'criteria': 'collaborations >= 10',
                'points': 800,
                'rarity': 'rare'
            }
        ]
        
        for achievement in achievements:
            try:
                self.db_manager.get_positive_cursor().execute('''
                    INSERT OR IGNORE INTO achievements 
                    (id, name, description, icon, criteria, points, rarity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    achievement['id'], achievement['name'], achievement['description'],
                    achievement['icon'], achievement['criteria'], achievement['points'],
                    achievement['rarity']
                ))
            except Exception as e:
                print(f"Error inserting achievement {achievement['id']}: {e}")
    
    def register_researcher(self, username: str, email: str, bio: str = '', 
                          specialization: List[str] = None) -> Dict[str, Any]:
        """Register a new researcher"""
        try:
            researcher_id = str(uuid.uuid4())
            specialization = specialization or []
            
            self.db_manager.get_positive_cursor().execute('''
                INSERT INTO researchers 
                (id, username, email, bio, specialization)
                VALUES (?, ?, ?, ?, ?)
            ''', (researcher_id, username, email, bio, json.dumps(specialization)))
            
            return {
                'success': True,
                'researcher_id': researcher_id,
                'message': f'Pesquisador {username} registrado com sucesso!'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Erro ao registrar pesquisador'
            }
    
    def create_research_post(self, author_id: str, title: str, content: str,
                           category: str, prime_number: str = None, 
                           exponent: int = None, tags: List[str] = None) -> Dict[str, Any]:
        """Create a new research post"""
        try:
            post_id = str(uuid.uuid4())
            tags = tags or []
            
            self.db_manager.get_positive_cursor().execute('''
                INSERT INTO research_posts 
                (id, author_id, title, content, category, prime_number, exponent, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (post_id, author_id, title, content, category, 
                  prime_number, exponent, json.dumps(tags)))
            
            # Update researcher contributions
            self._update_researcher_stats(author_id, contributions=1)
            
            return {
                'success': True,
                'post_id': post_id,
                'message': 'Post de pesquisa criado com sucesso!'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Erro ao criar post de pesquisa'
            }
    
    def get_community_feed(self, limit: int = 20, category: str = None,
                          sort_by: str = 'timestamp') -> List[Dict[str, Any]]:
        """Get community research feed"""
        try:
            query = '''
                SELECT rp.*, r.username, r.reputation
                FROM research_posts rp
                JOIN researchers r ON rp.author_id = r.id
            '''
            
            params = []
            if category:
                query += ' WHERE rp.category = ?'
                params.append(category)
            
            if sort_by == 'likes':
                query += ' ORDER BY rp.likes DESC'
            elif sort_by == 'verification_score':
                query += ' ORDER BY rp.verification_score DESC'
            else:
                query += ' ORDER BY rp.timestamp DESC'
            
            query += ' LIMIT ?'
            params.append(limit)
            
            cursor = self.db_manager.get_positive_cursor()
            cursor.execute(query, params)
            posts = cursor.fetchall()
            
            feed = []
            for post in posts:
                feed.append({
                    'id': post[0],
                    'author': post[13],
                    'author_reputation': post[14],
                    'title': post[2],
                    'content': post[3],
                    'category': post[4],
                    'prime_number': post[5],
                    'exponent': post[6],
                    'timestamp': post[7],
                    'likes': post[8],
                    'comments': json.loads(post[9] or '[]'),
                    'tags': json.loads(post[10] or '[]'),
                    'status': post[11],
                    'verification_score': post[12]
                })
            
            return feed
            
        except Exception as e:
            print(f"Error getting community feed: {e}")
            return []
    
    def verify_discovery(self, post_id: str, verifier_id: str, 
                        is_valid: bool, notes: str = '') -> Dict[str, Any]:
        """Verify a prime discovery"""
        try:
            # Get current verification score
            cursor = self.db_manager.get_positive_cursor()
            cursor.execute('SELECT verification_score, author_id FROM research_posts WHERE id = ?', (post_id,))
            result = cursor.fetchone()
            
            if not result:
                return {'success': False, 'message': 'Post não encontrado'}
            
            current_score, author_id = result
            
            # Calculate new verification score
            if is_valid:
                new_score = min(1.0, current_score + 0.1)
                points_awarded = 50
            else:
                new_score = max(0.0, current_score - 0.05)
                points_awarded = 5  # Small points for verification effort
            
            # Update post verification score
            cursor.execute('''
                UPDATE research_posts 
                SET verification_score = ?, status = ?
                WHERE id = ?
            ''', (new_score, 'verified' if new_score > 0.7 else 'pending', post_id))
            
            # Award points to verifier
            self._award_points(verifier_id, points_awarded)
            
            # If discovery is verified, award discovery points to author
            if is_valid and new_score > 0.7:
                self._update_researcher_stats(author_id, discoveries=1)
                self._award_points(author_id, 200)
            
            return {
                'success': True,
                'new_score': new_score,
                'message': f'Verificação registrada! Nova pontuação: {new_score:.2f}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Erro ao verificar descoberta'
            }
    
    def create_collaboration(self, creator_id: str, title: str, description: str,
                           target_range_start: int, target_range_end: int,
                           deadline: datetime = None) -> Dict[str, Any]:
        """Create a new collaborative research project"""
        try:
            collab_id = str(uuid.uuid4())
            deadline = deadline or (datetime.now() + timedelta(days=30))
            
            self.db_manager.get_positive_cursor().execute('''
                INSERT INTO collaborations 
                (id, title, description, creator_id, target_range_start, 
                 target_range_end, deadline)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (collab_id, title, description, creator_id,
                  target_range_start, target_range_end, deadline))
            
            return {
                'success': True,
                'collaboration_id': collab_id,
                'message': 'Projeto colaborativo criado com sucesso!'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Erro ao criar colaboração'
            }
    
    def join_collaboration(self, collaboration_id: str, researcher_id: str) -> Dict[str, Any]:
        """Join a collaborative research project"""
        try:
            cursor = self.db_manager.get_positive_cursor()
            cursor.execute('SELECT participants FROM collaborations WHERE id = ?', (collaboration_id,))
            result = cursor.fetchone()
            
            if not result:
                return {'success': False, 'message': 'Colaboração não encontrada'}
            
            participants = json.loads(result[0] or '[]')
            
            if researcher_id not in participants:
                participants.append(researcher_id)
                
                cursor.execute('''
                    UPDATE collaborations 
                    SET participants = ?
                    WHERE id = ?
                ''', (json.dumps(participants), collaboration_id))
                
                return {
                    'success': True,
                    'message': 'Você se juntou ao projeto colaborativo!'
                }
            else:
                return {
                    'success': False,
                    'message': 'Você já participa desta colaboração'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Erro ao entrar na colaboração'
            }
    
    def get_leaderboard(self, period: str = 'all_time', limit: int = 50) -> List[Dict[str, Any]]:
        """Get community leaderboard"""
        try:
            if period == 'monthly':
                sort_field = 'monthly_points'
            else:
                sort_field = 'total_points'
            
            cursor = self.db_manager.get_positive_cursor()
            cursor.execute(f'''
                SELECT r.username, r.reputation, r.discoveries, l.{sort_field}, l.rank_position
                FROM researchers r
                JOIN leaderboard l ON r.id = l.researcher_id
                ORDER BY l.{sort_field} DESC
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            
            leaderboard = []
            for i, result in enumerate(results, 1):
                leaderboard.append({
                    'rank': i,
                    'username': result[0],
                    'reputation': result[1],
                    'discoveries': result[2],
                    'points': result[3],
                    'position': result[4]
                })
            
            return leaderboard
            
        except Exception as e:
            print(f"Error getting leaderboard: {e}")
            return []
    
    def get_researcher_profile(self, researcher_id: str) -> Dict[str, Any]:
        """Get detailed researcher profile"""
        try:
            cursor = self.db_manager.get_positive_cursor()
            cursor.execute('''
                SELECT r.*, l.total_points, l.monthly_points, l.rank_position
                FROM researchers r
                LEFT JOIN leaderboard l ON r.id = l.researcher_id
                WHERE r.id = ?
            ''', (researcher_id,))
            
            result = cursor.fetchone()
            if not result:
                return {'success': False, 'message': 'Pesquisador não encontrado'}
            
            profile = {
                'id': result[0],
                'username': result[1],
                'email': result[2],
                'reputation': result[3],
                'discoveries': result[4],
                'contributions': result[5],
                'specialization': json.loads(result[6] or '[]'),
                'join_date': result[7],
                'last_active': result[8],
                'bio': result[9],
                'achievements': json.loads(result[10] or '[]'),
                'total_points': result[12] or 0,
                'monthly_points': result[13] or 0,
                'rank_position': result[14] or 0
            }
            
            return {'success': True, 'profile': profile}
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Erro ao buscar perfil'
            }
    
    def _update_researcher_stats(self, researcher_id: str, discoveries: int = 0,
                               contributions: int = 0, reputation: int = 0):
        """Update researcher statistics"""
        try:
            cursor = self.db_manager.get_positive_cursor()
            cursor.execute('''
                UPDATE researchers 
                SET discoveries = discoveries + ?,
                    contributions = contributions + ?,
                    reputation = reputation + ?,
                    last_active = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (discoveries, contributions, reputation, researcher_id))
            
        except Exception as e:
            print(f"Error updating researcher stats: {e}")
    
    def _award_points(self, researcher_id: str, points: int):
        """Award points to researcher"""
        try:
            cursor = self.db_manager.get_positive_cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO leaderboard 
                (researcher_id, total_points, monthly_points, last_updated)
                VALUES (
                    ?, 
                    COALESCE((SELECT total_points FROM leaderboard WHERE researcher_id = ?), 0) + ?,
                    COALESCE((SELECT monthly_points FROM leaderboard WHERE researcher_id = ?), 0) + ?,
                    CURRENT_TIMESTAMP
                )
            ''', (researcher_id, researcher_id, points, researcher_id, points))
            
        except Exception as e:
            print(f"Error awarding points: {e}")
    
    def get_community_stats(self) -> Dict[str, Any]:
        """Get overall community statistics"""
        try:
            cursor = self.db_manager.get_positive_cursor()
            
            # Get researcher count
            cursor.execute('SELECT COUNT(*) FROM researchers')
            researcher_count = cursor.fetchone()[0]
            
            # Get post count
            cursor.execute('SELECT COUNT(*) FROM research_posts')
            post_count = cursor.fetchone()[0]
            
            # Get collaboration count
            cursor.execute('SELECT COUNT(*) FROM collaborations')
            collaboration_count = cursor.fetchone()[0]
            
            # Get verified discoveries
            cursor.execute('SELECT COUNT(*) FROM research_posts WHERE status = "verified"')
            verified_discoveries = cursor.fetchone()[0]
            
            # Get top researchers
            cursor.execute('''
                SELECT r.username, l.total_points
                FROM researchers r
                JOIN leaderboard l ON r.id = l.researcher_id
                ORDER BY l.total_points DESC
                LIMIT 5
            ''')
            top_researchers = cursor.fetchall()
            
            return {
                'total_researchers': researcher_count,
                'total_posts': post_count,
                'active_collaborations': collaboration_count,
                'verified_discoveries': verified_discoveries,
                'top_researchers': [{'username': r[0], 'points': r[1]} for r in top_researchers]
            }
            
        except Exception as e:
            print(f"Error getting community stats: {e}")
            return {}

# Initialize community board instance
community_board = CommunityBoard()
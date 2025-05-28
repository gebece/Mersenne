# 🚀 MersenneHunter Installation Guide

Complete installation instructions for the MersenneHunter prime discovery platform.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **RAM**: 4GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

### Recommended for Optimal Performance
- **RAM**: 32GB+ for large-scale operations
- **CPU**: 16+ cores for maximum thread utilization
- **GPU**: NVIDIA with CUDA support (optional)
- **Storage**: SSD for database operations

## 🛠️ Quick Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/mersenne-hunter.git
cd mersenne-hunter
```

### 2. Install Dependencies
```bash
# Core installation
pip install Flask numpy numba psutil

# OR install all dependencies from list
pip install -r dependencies.txt
```

### 3. Run Basic System
```bash
python main.py --web-interface --threads 1000
```

Access the web interface at: `http://localhost:5000`

## 🎯 Advanced Installation Options

### GPU Acceleration (NVIDIA)
```bash
# Install CUDA support
pip install cupy-cuda12x

# Verify GPU detection
python -c "import cupy; print('GPU Available:', cupy.cuda.is_available())"
```

### Quantum Computing Support
```bash
# Install Qiskit
pip install qiskit qiskit-aer qiskit-algorithms

# Test quantum backend
python -c "from qiskit import Aer; print('Quantum Available:', len(Aer.backends()) > 0)"
```

### High Performance Package
```bash
# Install all performance packages
pip install cupy-cuda12x qiskit qiskit-aer numba aiohttp zstd
```

## 🌟 Installation Verification

### Test Basic Functionality
```bash
# Test core system
python -c "from mersenne_hunter import MersenneHunter; print('✅ Core system working')"

# Test web interface
python main.py --web-interface --threads 10 &
curl http://localhost:5000/api/status
```

### Performance Test
```bash
# Run performance benchmark
python main.py --benchmark --threads 100 --duration 60
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Optional optimizations
export PYTHONOPTIMIZE=1
export NUMBA_CACHE_DIR=/tmp/numba_cache
export CUDA_VISIBLE_DEVICES=0
```

### Database Configuration
```bash
# Custom database paths
export POSITIVE_DB_PATH="./custom_positives.db"
export NEGATIVE_DB_PATH="./custom_negatives.db"
```

## 🎮 Usage Examples

### Web Interface Mode
```bash
# Standard web interface
python main.py --web-interface --threads 5000 --mode hybrid

# High-performance mode
python main.py --web-interface --threads 10000 --mode hybrid --gpu
```

### Non-Stop Mode
```bash
# Continuous 24/7 operation
python mersenne_nonstop.py

# With custom configuration
python mersenne_nonstop.py --start-exponent 85000000
```

### CLI Mode
```bash
# Command line interface
python main.py --cli --threads 1000 --batch-size 50

# Specific range search
python main.py --cli --start 82000000 --end 83000000 --threads 2000
```

## 🐛 Troubleshooting

### Common Issues

#### Python Version Error
```bash
# Check Python version
python --version
# Should be 3.11+

# Install correct version if needed
# Ubuntu/Debian: sudo apt install python3.11
# macOS: brew install python@3.11
# Windows: Download from python.org
```

#### Memory Issues
```bash
# Reduce thread count
python main.py --threads 100

# Monitor memory usage
python main.py --monitor-memory
```

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi

# Verify CuPy installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

#### Database Permissions
```bash
# Fix database permissions
chmod 755 *.db
chown $USER:$USER *.db
```

## 📁 Directory Structure
```
mersenne-hunter/
├── main.py                    # Main entry point
├── mersenne_hunter.py         # Core search engine
├── mersenne_nonstop.py        # Non-stop system
├── web_interface.py           # Web dashboard
├── templates/                 # HTML templates
├── static/                    # CSS/JS files
├── logs/                      # Log files
├── *.db                       # Database files
└── README.md                  # Documentation
```

## 🔗 Integration with External Tools

### Jupyter Notebook
```python
# Install Jupyter support
pip install jupyter

# Create notebook
jupyter notebook mersenne_analysis.ipynb
```

### Docker Support
```dockerfile
# Create Dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r dependencies.txt
EXPOSE 5000
CMD ["python", "main.py", "--web-interface"]
```

## 📞 Support

If you encounter issues during installation:

1. **Check System Requirements**: Ensure all requirements are met
2. **Update Dependencies**: Run `pip install --upgrade -r dependencies.txt`
3. **Review Logs**: Check the `logs/` directory for error messages
4. **GitHub Issues**: Report problems at the GitHub repository
5. **Community**: Join discussions for community support

## 🎯 Next Steps

After successful installation:

1. **Web Interface**: Open `http://localhost:5000` to access the dashboard
2. **Start Search**: Click "▶️ Iniciar" to begin prime discovery
3. **Monitor Progress**: Watch real-time statistics and performance metrics
4. **Download Results**: Use the "📥 Baixar Resultados" button to export findings
5. **Non-Stop Mode**: Click "🔄 Reiniciar Sistema" to activate 10k thread mode

Happy prime hunting! 🎯
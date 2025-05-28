# 🤝 Contributing to MersenneHunter

Thank you for your interest in contributing to MersenneHunter! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Use the GitHub Issues tracker
- Include system information (OS, Python version, etc.)
- Provide steps to reproduce the issue
- Include relevant log files

### ✨ Feature Requests
- Describe the feature clearly
- Explain the use case and benefits
- Consider implementation complexity
- Check if similar features already exist

### 🔧 Code Contributions
- Fork the repository
- Create a feature branch
- Write clean, documented code
- Include tests for new features
- Follow the existing code style

## 🚀 Development Setup

### Local Environment
```bash
# Clone your fork
git clone https://github.com/yourusername/mersenne-hunter.git
cd mersenne-hunter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r dependencies.txt
```

### Testing
```bash
# Run basic tests
python -m pytest tests/

# Performance tests
python main.py --benchmark --duration 30

# Integration tests
python tests/test_integration.py
```

## 📝 Code Style Guidelines

### Python Standards
- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation
- Document all public functions
- Include docstrings with parameters and return values
- Update README.md for new features
- Add comments for complex algorithms

### Example Function Documentation
```python
def calculate_mersenne_candidate(exponent: int, confidence_threshold: float = 0.9) -> Dict[str, Any]:
    """
    Calculate Mersenne number candidate with confidence scoring.
    
    Args:
        exponent: Prime exponent for Mersenne number calculation
        confidence_threshold: Minimum confidence score (0.0 to 1.0)
        
    Returns:
        Dictionary containing candidate information and confidence score
        
    Raises:
        ValueError: If exponent is not prime or threshold invalid
    """
```

## 🧪 Testing Guidelines

### Unit Tests
- Test individual functions in isolation
- Use meaningful test names
- Cover edge cases and error conditions
- Aim for high code coverage

### Performance Tests
- Benchmark critical algorithms
- Test with various thread counts
- Monitor memory usage
- Validate scalability

### Integration Tests
- Test complete workflows
- Verify database operations
- Check web interface functionality
- Test distributed computing features

## 🔄 Pull Request Process

### Before Submitting
1. Ensure all tests pass
2. Update documentation
3. Add changelog entry
4. Verify code style compliance
5. Test on multiple Python versions

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog updated
```

## 🎨 Design Principles

### Performance First
- Optimize for speed and efficiency
- Minimize memory usage
- Use appropriate data structures
- Consider thread safety

### Mathematical Accuracy
- Implement proven algorithms
- Validate mathematical correctness
- Handle edge cases properly
- Maintain numerical precision

### User Experience
- Keep interfaces intuitive
- Provide clear error messages
- Offer comprehensive monitoring
- Enable easy configuration

## 🌟 Feature Development Guidelines

### New Search Algorithms
1. Research mathematical foundation
2. Implement with proper testing
3. Compare performance with existing methods
4. Document algorithm complexity
5. Add configuration options

### Web Interface Enhancements
1. Maintain responsive design
2. Ensure cross-browser compatibility
3. Add proper error handling
4. Include accessibility features
5. Test with various screen sizes

### Performance Optimizations
1. Profile code to identify bottlenecks
2. Use appropriate optimization techniques
3. Maintain code readability
4. Validate performance improvements
5. Consider memory vs speed tradeoffs

## 📊 Benchmarking Standards

### Performance Metrics
- Tests per second
- Memory usage
- CPU utilization
- Thread efficiency
- Cache hit rates

### Benchmark Environment
- Document hardware specifications
- Use consistent test conditions
- Run multiple iterations
- Report statistical significance
- Compare with baseline performance

## 🛡️ Security Considerations

### Code Security
- Validate all inputs
- Handle errors gracefully
- Avoid information leakage
- Use secure communication
- Follow security best practices

### Data Protection
- Protect sensitive results
- Implement access controls
- Use encryption for network communication
- Audit system access
- Regular security reviews

## 📚 Learning Resources

### Mathematical Background
- Number Theory fundamentals
- Primality testing algorithms
- Computational complexity
- Distributed algorithms
- Performance optimization

### Technical Skills
- Python advanced features
- Multi-threading programming
- Database optimization
- Web development
- GPU programming (optional)

## 🎉 Recognition

Contributors will be:
- Listed in the project contributors
- Mentioned in release notes
- Credited in documentation
- Invited to project discussions
- Recognized for significant contributions

## 📞 Getting Help

### Community Support
- GitHub Discussions for questions
- Issues tracker for bugs
- Pull request reviews
- Code review feedback
- Mentoring for new contributors

### Contact Information
- Project maintainers
- Core development team
- Community moderators
- Technical advisors

Thank you for contributing to advancing mathematical research! 🧮✨
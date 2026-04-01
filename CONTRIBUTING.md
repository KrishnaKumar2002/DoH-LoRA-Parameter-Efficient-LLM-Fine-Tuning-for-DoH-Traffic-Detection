# Contributing to DoH-LoRA

Thank you for interest in contributing to the DoH-LoRA project! Here's how you can help:

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/DoH-LoRA.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Install dev dependencies: `pip install -r requirements.txt`

## Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) standards
- Use type hints for function signatures
- Write docstrings for all functions and classes (Google style)

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Testing

- Write tests for new features in `tests/`
- Run tests: `pytest tests/`
- Aim for >80% code coverage for new code

### Documentation

- Update README.md if adding features
- Add docstrings to all public functions
- Update CHANGELOG.md with your changes

## Commit Guidelines

- Use descriptive commit messages
- Reference issues: `Fixes #123`
- Format: `type(scope): description`
  - type: feat, fix, docs, refactor, test, chore
  - scope: the module/component affected
  - Example: `feat(model): add gradient accumulation support`

## Pull Request Process

1. Update documentation and tests
2. Ensure all tests pass: `pytest tests/`
3. Run code quality checks (black, flake8, mypy)
4. Create PR with clear description
5. Address review feedback

## Reporting Bugs

Include:
- Python version
- CUDA version (if applicable)
- Error traceback
- Minimal reproducible example
- Expected vs actual behavior

## Questions?

Open a [Discussion](https://github.com/KrishnaKumar2002/DoH-LoRA-Parameter-Efficient-LLM-Fine-Tuning-for-DoH-Traffic-Detection/discussions) on GitHub!

---

Happy contributing! 🚀

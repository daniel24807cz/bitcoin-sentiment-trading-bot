# Contributing

Contributions are welcome! To keep the codebase clean and reliable, please follow these guidelines:

1. Create a new branch from `main` (`git checkout -b feature/your-feature-name`).
2. Keep functions focused and include docstrings and type annotations.
3. Add corresponding unit tests in `test_bot.py` (with mocked external APIs) for every new feature or bug fix.
4. Make sure all tests pass before opening a Pull Request:
   ```bash
   python -m unittest test_bot.py -v
   ```
5. Never commit files containing API keys (`.env`) or local backups (`backup/`).

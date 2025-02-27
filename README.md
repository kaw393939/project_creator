# Python Project Generator

A bash script to quickly generate well-structured Python project templates with modern development tooling.

## Overview

This script creates a complete Python project structure with your choice of:
- Basic Python package
- Command-line interface (CLI) application
- Streamlit web application

The generator sets up a comprehensive development environment including:
- Proper package structure following best practices
- Virtual environment
- Testing framework
- Code quality tools
- Docker configuration (optional)
- Project documentation

## Installation

1. Download the script:
   ```bash
   curl -o generate_python_project.sh https://raw.githubusercontent.com/yourusername/python-project-generator/main/generate_python_project.sh
   ```

2. Make it executable:
   ```bash
   chmod +x generate_python_project.sh
   ```

3. Run the script:
   ```bash
   ./generate_python_project.sh
   ```

## Quick Start

Running the script will guide you through an interactive setup process with sensible defaults:

```bash
./generate_python_project.sh
```

The script will prompt you for:
- Project name
- Package name (for imports)
- Initial version
- Project description
- Author name & email
- Project type (basic package, CLI, or Streamlit)
- Python version
- Whether to include various tools (pytest, Docker, Black, Flake8)

## Project Types Explained

### Basic Python Package

A standard Python package structure suitable for libraries and modules you intend to import into other projects.

**When to use**: When creating reusable code, libraries, or modules.

### Command-line Interface (CLI)

Sets up a Python application with argument parsing and entry points for command-line usage.

**When to use**: When creating tools and utilities meant to be run from the terminal.

**Example usage** after installation:
```bash
# After installing the package:
your_package_name --name "World"

# Or through the run script:
./run_cli.sh --name "World"
```

### Streamlit Application

Creates a web application using the Streamlit framework, great for data visualization, dashboards, and simple web apps.

**When to use**: When you want to create interactive data visualizations, dashboards, or simple web applications without extensive web development knowledge.

**Example usage**:
```bash
# Run the Streamlit app:
./run_streamlit.sh

# Access in browser at http://localhost:8501
```

## Included Technologies & Tools

### Testing with pytest

[pytest](https://docs.pytest.org/) is a powerful testing framework for Python.

**What it does**: Allows you to write simple tests for your code.

**How to use it**:
1. Tests are automatically created in the `tests/` directory
2. Run tests with:
   ```bash
   ./run_tests.sh
   ```
   Or:
   ```bash
   python -m pytest
   ```

3. Run tests with coverage report:
   ```bash
   python -m pytest --cov=your_package_name
   ```

**Example test** (already created for you):
```python
# tests/unit/test_hello.py
def test_hello_default():
    result = hello()
    assert result == "Hello, World!"
```

### Code Formatting with Black

[Black](https://black.readthedocs.io/) is an automatic code formatter for Python.

**What it does**: Automatically formats your code to a consistent style, eliminating debates about formatting.

**How to use it**:
1. Format a specific file:
   ```bash
   black src/your_package/main.py
   ```

2. Format the entire project:
   ```bash
   black .
   ```

3. Check if files are properly formatted without changing them:
   ```bash
   black --check .
   ```

### Code Linting with Flake8

[Flake8](https://flake8.pycqa.org/) is a code linter that checks for stylistic and logical errors.

**What it does**: Identifies potential errors, style issues, and complexity problems in your code.

**How to use it**:
1. Check a specific file:
   ```bash
   flake8 src/your_package/main.py
   ```

2. Check the entire project:
   ```bash
   flake8
   ```

### Docker Integration

[Docker](https://www.docker.com/) is a platform for developing, shipping, and running applications in containers.

**What it does**: Packages your application with its dependencies into a standardized unit for software development.

**How to use it**:
1. Build the Docker image:
   ```bash
   docker build -t your_project_name .
   ```

2. Run the Docker container:
   ```bash
   # For a basic package or CLI application:
   docker run your_project_name
   
   # For Streamlit apps, exposing the port:
   docker run -p 8501:8501 your_project_name
   ```

3. Access Streamlit applications at `http://localhost:8501` when running in Docker.

### Virtual Environment

Python virtual environments isolate dependencies for different projects.

**What it does**: Creates a separate environment for your project's dependencies, preventing conflicts with other projects.

**How to use it**:
1. Activate the environment:
   ```bash
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Install your package in development mode:
   ```bash
   pip install -e .
   ```

4. Deactivate when done:
   ```bash
   deactivate
   ```

## Streamlit Crash Course

If you selected the Streamlit project type, here's a quick guide to what's included:

Streamlit turns Python scripts into shareable web apps with minimal effort. Your generated project includes:

- Main application in `src/your_package/main.py`
- Example multi-page app setup with data analysis page
- Interactive controls (sliders, dropdowns, etc.)
- Data visualization examples

To understand the code:

1. **Layout**: Components flow from top to bottom
   ```python
   st.title("My App")        # Creates a title
   st.header("Data Section") # Creates a header
   
   # Creates two columns
   col1, col2 = st.columns(2)
   with col1:
       st.write("Column 1 content")
   with col2:
       st.write("Column 2 content")
   ```

2. **Interactive components**:
   ```python
   name = st.text_input("Enter your name")
   age = st.slider("Your age", 0, 100, 25)
   submitted = st.button("Submit")
   
   if submitted:
       st.write(f"Hello {name}, you are {age} years old!")
   ```

3. **Data visualization**:
   ```python
   import pandas as pd
   import numpy as np
   
   data = pd.DataFrame({
       'x': np.random.randn(100),
       'y': np.random.randn(100)
   })
   
   st.scatter_chart(data)  # Creates a scatter plot
   ```

## CLI Application Crash Course

If you selected the CLI project type, here's a quick guide to what's included:

The template uses Python's `argparse` module to handle command-line arguments:

1. **Adding new commands**:
   Edit `src/your_package/main.py` to add new arguments:
   ```python
   parser.add_argument("--count", type=int, default=1, 
                      help="Number of times to repeat")
   ```

2. **Adding new functionality**:
   Create new functions in `main.py` and call them from the `main()` function.

3. **Testing your CLI**:
   ```bash
   # Using the run script:
   ./run_cli.sh --help
   
   # Or directly:
   python -m your_package.main --help
   ```

## Project Structure Explained

```
your_project_name/
├── .flake8                 # Flake8 configuration
├── .gitignore              # Git ignore rules
├── .coveragerc             # Coverage configuration
├── Dockerfile              # Docker configuration
├── README.md               # Project documentation
├── pyproject.toml          # Black configuration
├── requirements-dev.txt    # Development dependencies
├── requirements.txt        # Core dependencies
├── run_tests.sh            # Script to run tests
├── setup.py                # Package installation
├── src/                    # Source code
│   └── your_package/       # Your package
│       ├── __init__.py
│       ├── __version__.py  # Version information
│       └── main.py         # Main module
└── tests/                  # Tests
    ├── __init__.py
    ├── conftest.py         # Test fixtures
    ├── integration/        # Integration tests
    └── unit/               # Unit tests
```

## Best Practices

The project generator includes several Python best practices:

1. **Package Structure**: Follows the `src` layout which avoids import issues during development.

2. **Semantic Versioning**: Uses [SemVer](https://semver.org/) for version numbers (MAJOR.MINOR.PATCH).

3. **Documentation**: Includes docstrings in all functions and modules.

4. **Type Hints**: Uses Python type hints for better code understanding and editor support.

5. **Testing**: Separates unit and integration tests.

6. **Continuous Integration**: Ready to integrate with CI systems.

## Common Tasks

### Adding a new dependency

1. Add to `requirements.txt`:
   ```
   new_package>=1.0.0
   ```

2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update `setup.py` to include the new dependency in `install_requires`.

### Creating a release

1. Update version in `src/your_package/__version__.py`

2. Build the package:
   ```bash
   pip install build
   python -m build
   ```

3. Package will be created in the `dist/` directory.

### Running code quality checks

```bash
# Format code with Black
black .

# Run linting
flake8

# Run tests
pytest
```

## Troubleshooting

### Virtual Environment Issues

If you have issues with the virtual environment:

```bash
# Remove and recreate it
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

### Import Errors

If you get import errors during development:

```bash
# Make sure you've installed the package in development mode
pip install -e .

# Check your PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### Streamlit Not Running

If Streamlit app fails to start:

```bash
# Make sure Streamlit is installed
pip install streamlit

# Run with verbose logging
streamlit run app.py --logger.level=debug
```

## Contributing to This Generator

Feel free to enhance this generator:

1. Add new features or templates
2. Improve documentation or examples
3. Fix bugs or issues

Submit pull requests to make this tool even better!

## License

This project is open-sourced under the MIT License.

## Credits

Created with ❤️ to simplify Python project setup and promote best practices.

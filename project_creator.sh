#!/bin/bash
set -e

echo "====================================================="
echo "  Python Project Generator - Interactive Setup"
echo "====================================================="
echo ""

# Function to prompt for input with default value
prompt_with_default() {
  local prompt="$1"
  local default="$2"
  local response
  
  read -p "$prompt [$default]: " response
  echo "${response:-$default}"
}

# Function to prompt for yes/no with default
prompt_yes_no() {
  local prompt="$1"
  local default="$2"
  local response
  
  if [[ "$default" == "Y" ]]; then
    read -p "$prompt [Y/n]: " response
    response=${response:-Y}
  else
    read -p "$prompt [y/N]: " response
    response=${response:-N}
  fi
  
  if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Y"
  else
    echo "N"
  fi
}

# Check if directory exists and handle it
check_existing_project() {
  if [ -d "$PROJECT_NAME" ]; then
    read -p "Directory '$PROJECT_NAME' already exists. Overwrite? [y/N]: " OVERWRITE
    if [[ ! "$OVERWRITE" =~ ^[Yy]$ ]]; then
      echo "Aborted."
      exit 1
    fi
    # Backup the existing directory
    BACKUP_DIR="${PROJECT_NAME}_backup_$(date +%Y%m%d%H%M%S)"
    echo "Backing up existing directory to $BACKUP_DIR"
    mv "$PROJECT_NAME" "$BACKUP_DIR"
  fi
}

# Function to check if commands exist
check_command() {
  local cmd="$1"
  if ! command -v "$cmd" &> /dev/null; then
    echo "Error: $cmd is not installed or not in PATH"
    return 1
  fi
  return 0
}

# Check essential commands
check_command python3 || exit 1

# Prompt for project details
PROJECT_NAME=$(prompt_with_default "Project name" "python_starter")
# Validate project name
if [[ ! "$PROJECT_NAME" =~ ^[a-zA-Z][a-zA-Z0-9_-]*$ ]]; then
  echo "Invalid project name. Project name must start with a letter and contain only letters, numbers, underscores, and hyphens."
  exit 1
fi

check_existing_project
PACKAGE_NAME=$(prompt_with_default "Package name (for imports)" "${PROJECT_NAME//-/_}")
VERSION=$(prompt_with_default "Initial version" "0.1.0")
DESCRIPTION=$(prompt_with_default "Project description" "A Python project")
AUTHOR=$(prompt_with_default "Author name" "Your Name")
EMAIL=$(prompt_with_default "Author email" "email@example.com")

# Additional features
USE_PYTEST=$(prompt_yes_no "Use pytest for testing" "Y")
USE_DOCKER=$(prompt_yes_no "Set up Docker" "Y")
USE_BLACK=$(prompt_yes_no "Use Black for code formatting" "Y")
USE_FLAKE8=$(prompt_yes_no "Use Flake8 for linting" "Y")

# Project type options
echo "Project type:"
echo "1) Basic Python package"
echo "2) Command-line interface (CLI)"
echo "3) Streamlit application"
read -p "Select project type [1]: " PROJECT_TYPE
PROJECT_TYPE=${PROJECT_TYPE:-1}

CREATE_CLI="N"
CREATE_STREAMLIT="N"

case $PROJECT_TYPE in
  2) CREATE_CLI="Y" ;;
  3) CREATE_STREAMLIT="Y" ;;
  *) echo "Selected basic Python package" ;;
esac

# Python version
PY_VERSION=$(prompt_with_default "Python version" "3.10")

echo ""
echo "Creating project: $PROJECT_NAME"
echo ""

# Create project directory
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# pytest
.pytest_cache/
.coverage
htmlcov/

# Docker
.dockerignore

# IDE
.idea/
.vscode/
*.swp
*.swo

# MacOS
.DS_Store
EOF

# Create virtual environment
python3 -m venv venv
echo "Virtual environment created"

# Define function to activate venv and run a command
run_in_venv() {
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        "$@"
        deactivate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        "$@"
        deactivate
    else
        echo "Warning: Could not activate virtual environment"
        "$@"
    fi
}

# Create source directory
mkdir -p src/$PACKAGE_NAME
touch src/$PACKAGE_NAME/__init__.py

# Create version file
cat > src/$PACKAGE_NAME/__version__.py << EOF
"""Version information."""

__version__ = "$VERSION"
EOF

# Create main module based on project type
if [[ "$CREATE_STREAMLIT" == "Y" ]]; then
  # Create config.toml for Streamlit
  mkdir -p .streamlit
  cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
EOF

  # Create main Streamlit module
  cat > src/$PACKAGE_NAME/main.py << EOF
"""Streamlit application module."""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import importlib.metadata
from pathlib import Path

# Get version in a way that works with Streamlit
try:
    # First try direct import if package is installed
    from $PACKAGE_NAME.__version__ import __version__
except ImportError:
    try:
        # Try to get version from metadata if installed
        __version__ = importlib.metadata.version("$PROJECT_NAME")
    except importlib.metadata.PackageNotFoundError:
        # Fallback version if not installed
        __version__ = "$VERSION"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_sample_data() -> pd.DataFrame:
    """Generate sample data for demonstration.
    
    Returns:
        pd.DataFrame: Sample dataframe with random data
    """
    logger.info("Generating sample data")
    np.random.seed(42)  # For reproducible results
    date_range = pd.date_range(start="2023-01-01", periods=20, freq="D")
    
    data = {
        "date": date_range,
        "value_a": np.random.randn(20).cumsum(),
        "value_b": np.random.randn(20).cumsum(),
        "category": np.random.choice(["A", "B", "C"], size=20)
    }
    
    return pd.DataFrame(data)


def create_sidebar() -> dict:
    """Create and process sidebar elements.
    
    Returns:
        dict: Dictionary of sidebar settings
    """
    st.sidebar.header("Settings")
    
    settings = {
        "chart_type": st.sidebar.selectbox(
            "Chart Type",
            options=["Line", "Bar", "Area"],
            index=0
        ),
        "show_raw_data": st.sidebar.checkbox("Show raw data", value=False),
        "smooth_factor": st.sidebar.slider("Smoothing factor", 0, 10, 0),
    }
    
    st.sidebar.header("About")
    st.sidebar.markdown(f"**${PROJECT_NAME}** v{__version__}")
    st.sidebar.markdown("Created with Streamlit")
    
    return settings


def display_chart(data: pd.DataFrame, settings: dict) -> None:
    """Display chart based on settings.
    
    Args:
        data: DataFrame containing the data
        settings: Dictionary of chart settings
    """
    # Apply smoothing if requested
    if settings["smooth_factor"] > 0:
        data = data.copy()
        for col in ["value_a", "value_b"]:
            data[col] = data[col].rolling(
                window=settings["smooth_factor"], 
                min_periods=1
            ).mean()
    
    # Display the appropriate chart type
    if settings["chart_type"] == "Line":
        st.line_chart(data.set_index("date")[["value_a", "value_b"]])
    elif settings["chart_type"] == "Bar":
        st.bar_chart(data.set_index("date")[["value_a", "value_b"]])
    else:  # Area chart
        st.area_chart(data.set_index("date")[["value_a", "value_b"]])
    
    # Show raw data if requested
    if settings["show_raw_data"]:
        st.subheader("Raw Data")
        st.dataframe(data)


def main() -> None:
    """Main Streamlit application."""
    logger.info(f"Starting Streamlit app v{__version__}")
    
    st.set_page_config(
        page_title="${PROJECT_NAME}",
        page_icon="✨",
        layout="wide",
    )
    
    st.title("${PROJECT_NAME}")
    st.caption(f"v{__version__}")
    
    # Create sidebar and get settings
    settings = create_sidebar()
    
    # Main content area
    st.header("Data Visualization")
    
    # Generate sample data
    data = generate_sample_data()
    
    # Display visualizations
    display_chart(data, settings)
    
    # Additional information
    with st.expander("About this application"):
        st.markdown("""
        This is a sample Streamlit application that demonstrates:
        
        - Data visualization with different chart types
        - Interactive controls in the sidebar
        - Expandable sections
        - Responsive layout
        
        Feel free to modify this template for your own needs.
        """)


if __name__ == "__main__":
    main()
EOF

  # Create entry point script for running streamlit directly
  cat > app.py << EOF
"""Streamlit application entry point."""
import sys
import os
from pathlib import Path

# Add src directory to Python path to ensure proper imports
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Now import and run the main function
import streamlit as st
from ${PACKAGE_NAME}.main import main

if __name__ == "__main__":
    main()
EOF

  # Create pages directory for multi-page Streamlit apps
  mkdir -p src/$PACKAGE_NAME/pages
  touch src/$PACKAGE_NAME/pages/__init__.py
  
  # Create sample pages
  cat > src/$PACKAGE_NAME/pages/data_analysis.py << EOF
"""Data analysis page for Streamlit multi-page app."""
import streamlit as st
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Data Analysis", page_icon="📊")

st.title("Data Analysis")
st.markdown("Perform basic analysis on your data.")

# Generate some sample data
@st.cache_data
def get_data() -> pd.DataFrame:
    """Generate sample data with caching.
    
    Returns:
        pd.DataFrame: Sample dataframe
    """
    logger.info("Generating cached data for analysis page")
    np.random.seed(42)
    data = {
        "category": np.random.choice(["A", "B", "C", "D"], size=100),
        "value1": np.random.randn(100),
        "value2": np.random.randn(100) * 2 + 1,
        "date": pd.date_range(start="2023-01-01", periods=100)
    }
    return pd.DataFrame(data)

data = get_data()

# Allow file upload for real data
uploaded_file = st.file_uploader("Upload your own CSV data", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        logger.info(f"User uploaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        logger.error(f"File upload error: {e}")

# Display data statistics
st.subheader("Data Overview")
st.dataframe(data.head())

col1, col2 = st.columns(2)
with col1:
    st.subheader("Summary Statistics")
    st.dataframe(data.describe())

with col2:
    if "category" in data.columns:
        st.subheader("Category Distribution")
        cat_counts = data["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        st.bar_chart(cat_counts.set_index("Category"))

# Advanced analysis
st.subheader("Advanced Analysis")
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()

if len(numeric_cols) >= 2:
    selected_cols = st.multiselect(
        "Select columns for correlation analysis",
        options=numeric_cols,
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    )
    
    if len(selected_cols) >= 2:
        st.subheader("Correlation Matrix")
        corr = data[selected_cols].corr()
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
        
        st.subheader("Scatter Plot")
        x_col = st.selectbox("X axis", options=selected_cols, index=0)
        y_col = st.selectbox("Y axis", options=selected_cols, index=min(1, len(selected_cols)-1))
        
        scatter_data = pd.DataFrame({
            "x": data[x_col],
            "y": data[y_col]
        })
        st.scatter_chart(scatter_data.set_index("x"))
EOF

  # Create run_streamlit.sh script
  cat > run_streamlit.sh << EOF
#!/bin/bash
# Run the Streamlit application

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
  source venv/Scripts/activate
fi

# Ensure src package is in PYTHONPATH
export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src

# Run streamlit application
streamlit run app.py "\$@"
EOF
  chmod +x run_streamlit.sh

elif [[ "$CREATE_CLI" == "Y" ]]; then
  cat > src/$PACKAGE_NAME/main.py << EOF
"""Main module for the CLI application."""
import argparse
import sys
import logging
from typing import List, Optional
from .__version__ import __version__

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def hello(name: str = "World") -> str:
    """Return a greeting message.
    
    Args:
        name: Name to greet. Defaults to "World".
    
    Returns:
        str: Greeting message
    """
    logger.debug(f"Generating greeting for {name}")
    return f"Hello, {name}!"


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.
    
    Args:
        args: Command line arguments. Defaults to None (sys.argv[1:]).
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="$DESCRIPTION")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("--name", default="World", help="Name to greet")
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Increase verbosity"
    )
    
    # Parse args
    parsed_args = parser.parse_args(args)
    
    # Set log level based on verbosity
    if parsed_args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif parsed_args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    
    return parsed_args


def main(args: Optional[List[str]] = None) -> int:
    """Entry point for the application.
    
    Args:
        args: Command line arguments. Defaults to None (sys.argv[1:]).
        
    Returns:
        int: Exit code
    """
    parsed_args = parse_args(args)
    logger.info(f"Starting $PACKAGE_NAME v{__version__}")
    
    message = hello(parsed_args.name)
    print(message)
    
    logger.info("Command completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
EOF

  # Create entry point script
  mkdir -p bin
  cat > bin/$PACKAGE_NAME << EOF
#!/usr/bin/env python3
"""Command-line entry point."""

from $PACKAGE_NAME.main import main
import sys

if __name__ == "__main__":
    sys.exit(main())
EOF
  chmod +x bin/$PACKAGE_NAME

else
  cat > src/$PACKAGE_NAME/main.py << EOF
"""Main module for the application."""
import sys
import os
import logging
from typing import Dict, Any, Optional, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import version dynamically
from .__version__ import __version__


def hello(name: str = "World") -> str:
    """Return a greeting message.
    
    Args:
        name: Name to greet. Defaults to "World".
    
    Returns:
        str: Greeting message
    """
    logger.info(f"Greeting {name}")
    return f"Hello, {name}!"


def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process a list of data dictionaries.
    
    Args:
        data: List of data dictionaries to process
        
    Returns:
        Dict[str, Any]: Results of the processing
    """
    if not data:
        logger.warning("Empty data provided")
        return {"status": "empty", "count": 0, "results": []}
    
    logger.info(f"Processing {len(data)} data items")
    
    # Example processing logic
    total = sum(item.get("value", 0) for item in data)
    average = total / len(data) if data else 0
    
    return {
        "status": "success",
        "count": len(data),
        "total": total,
        "average": average,
        "results": data
    }


def main() -> None:
    """Entry point for the application."""
    logger.info(f"Running {__name__} version {__version__}")
    message = hello()
    print(message)
    
    # Example data processing
    sample_data = [
        {"id": 1, "name": "Item 1", "value": 10},
        {"id": 2, "name": "Item 2", "value": 20},
        {"id": 3, "name": "Item 3", "value": 30},
    ]
    
    results = process_data(sample_data)
    print(f"Processed {results['count']} items")
    print(f"Total value: {results['total']}")
    print(f"Average value: {results['average']}")


if __name__ == "__main__":
    main()
EOF

  # Create run script
  cat > run.py << EOF
#!/usr/bin/env python3
"""Run the application."""

from src.$PACKAGE_NAME.main import main

if __name__ == "__main__":
    main()
EOF
  chmod +x run.py
fi

# Create tests directory
if [[ "$USE_PYTEST" == "Y" ]]; then
  mkdir -p tests
  mkdir -p tests/unit
  mkdir -p tests/integration
  touch tests/__init__.py
  touch tests/unit/__init__.py
  touch tests/integration/__init__.py
  
  # Create conftest.py for shared pytest fixtures
  cat > tests/conftest.py << EOF
"""Shared fixtures for pytest."""
import pytest
import sys
import os
from pathlib import Path

# Add the src directory to the path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

@pytest.fixture
def sample_data():
    """Fixture providing sample data for tests."""
    return [
        {"id": 1, "name": "Item 1", "value": 10},
        {"id": 2, "name": "Item 2", "value": 20},
        {"id": 3, "name": "Item 3", "value": 30},
    ]
EOF

  # Create test file based on project type
  if [[ "$CREATE_CLI" == "Y" ]]; then
    # Unit tests for CLI application
    cat > tests/unit/test_hello.py << EOF
"""Unit tests for hello function."""
import pytest
from $PACKAGE_NAME.main import hello

def test_hello_default():
    """Test hello function with default parameter."""
    result = hello()
    assert result == "Hello, World!"

def test_hello_with_name():
    """Test hello function with a specific name."""
    result = hello("Python")
    assert result == "Hello, Python!"
EOF

    cat > tests/unit/test_args.py << EOF
"""Unit tests for argument parsing."""
import pytest
from $PACKAGE_NAME.main import parse_args

def test_parse_args_default():
    """Test argument parsing with default values."""
    args = parse_args([])
    assert args.name == "World"

def test_parse_args_custom():
    """Test argument parsing with custom values."""
    args = parse_args(["--name", "CLI"])
    assert args.name == "CLI"

def test_parse_args_verbose():
    """Test verbose flag parsing."""
    args = parse_args(["-v"])
    assert args.verbose == 1
    
    args = parse_args(["-vv"])
    assert args.verbose == 2
EOF

    cat > tests/integration/test_cli.py << EOF
"""Integration tests for CLI."""
import pytest
import subprocess
import sys
import os
from unittest.mock import patch
from $PACKAGE_NAME.main import main

def test_main_return_code():
    """Test main function return code."""
    with patch('sys.stdout'):
        result = main(["--name", "Tester"])
        assert result == 0

@pytest.mark.integration
def test_executable():
    """Test the command-line executable (requires installation)."""
    # Skip if not installed
    if not os.path.exists("bin/$PACKAGE_NAME"):
        pytest.skip("Executable not found - package may not be installed")
    
    # Run the command
    result = subprocess.run(
        ["./bin/$PACKAGE_NAME", "--name", "CLI Test"],
        capture_output=True,
        text=True
    )
    
    # Check results
    assert result.returncode == 0
    assert "Hello, CLI Test!" in result.stdout
EOF

  elif [[ "$CREATE_STREAMLIT" == "Y" ]]; then
    # Unit tests for Streamlit application
    cat > tests/unit/test_data_generation.py << EOF
"""Unit tests for data generation functions."""
import pytest
from $PACKAGE_NAME.main import generate_sample_data

def test_generate_sample_data():
    """Test sample data generation."""
    data = generate_sample_data()
    
    # Check that data has the expected structure
    assert len(data) == 20
    assert "date" in data.columns
    assert "value_a" in data.columns
    assert "value_b" in data.columns
    assert "category" in data.columns
    
    # Check data types
    assert data["date"].dtype.kind == 'M'  # datetime
    assert data["value_a"].dtype.kind == 'f'  # float
    assert data["value_b"].dtype.kind == 'f'  # float
    assert data["category"].dtype.kind in ['O', 'U']  # object or unicode
EOF

    cat > tests/integration/test_streamlit.py << EOF
"""Integration tests for Streamlit application."""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import importlib

# This is a basic test to ensure the module can be imported
def test_import_main():
    """Test that the main module can be imported."""
    try:
        from $PACKAGE_NAME import main
        assert main is not None
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

# Since Streamlit apps are difficult to test directly,
# we can test the components/functions without the Streamlit context
def test_main_function_imports():
    """Test that main function can be imported."""
    try:
        from $PACKAGE_NAME.main import main
        assert main is not None
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
EOF

  else
    # Unit tests for basic Python package
    cat > tests/unit/test_hello.py << EOF
"""Unit tests for hello function."""
import pytest
from $PACKAGE_NAME.main import hello

def test_hello_default():
    """Test hello function with default parameter."""
    result = hello()
    assert result == "Hello, World!"

def test_hello_with_name():
    """Test hello function with a specific name."""
    result = hello("Python")
    assert result == "Hello, Python!"
EOF

    cat > tests/unit/test_data_processing.py << EOF
"""Unit tests for data processing functions."""
import pytest
from $PACKAGE_NAME.main import process_data

def test_process_data_empty():
    """Test processing empty data."""
    result = process_data([])
    assert result["status"] == "empty"
    assert result["count"] == 0

def test_process_data(sample_data):
    """Test processing sample data using fixture."""
    result = process_data(sample_data)
    
    assert result["status"] == "success"
    assert result["count"] == 3
    assert result["total"] == 60  # 10 + 20 + 30
    assert result["average"] == 20  # 60 / 3
    assert len(result["results"]) == 3
EOF

    cat > tests/integration/test_main.py << EOF
"""Integration tests for main module."""
import pytest
import sys
from unittest.mock import patch
from io import StringIO
from $PACKAGE_NAME.main import main

def test_main_function():
    """Test the main function output."""
    # Capture stdout
    captured_output = StringIO()
    sys.stdout = captured_output
    
    # Call main
    main()
    
    # Reset stdout
    sys.stdout = sys.__stdout__
    
    # Check output
    output = captured_output.getvalue()
    assert "Hello, World!" in output
    assert "Processed 3 items" in output
    assert "Total value: 60" in output
    assert "Average value: 20" in output
EOF
  fi

  # Create pytest.ini with more complete configuration
  cat > pytest.ini << EOF
[pytest]
pythonpath = src
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
addopts = --strict-markers -v
EOF

  # Add coverage configuration
  cat > .coveragerc << EOF
[run]
source = $PACKAGE_NAME
omit = 
    */tests/*
    */venv/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    pass
    raise ImportError
omit =
    tests/*
EOF
fi

# Create README.md first (needed by setup.py)
cat > README.md << EOF
# $PROJECT_NAME

$DESCRIPTION

## Development Setup

1. Clone the repository
2. Create and activate virtual environment:
   \`\`\`
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   \`\`\`
3. Install development dependencies:
   \`\`\`
   pip install -r requirements-dev.txt
   pip install -e .
   \`\`\`

EOF

if [[ "$CREATE_STREAMLIT" == "Y" ]]; then
  cat >> README.md << EOF
## Running the Streamlit Application

Run the Streamlit application:
\`\`\`
./run_streamlit.sh
\`\`\`

Or directly with the streamlit command:
\`\`\`
streamlit run app.py
\`\`\`

EOF
elif [[ "$CREATE_CLI" == "Y" ]]; then
  cat >> README.md << EOF
## Running the CLI Application

Run the CLI application:
\`\`\`
./bin/$PACKAGE_NAME --name YourName
\`\`\`

Or after installation:
\`\`\`
$PACKAGE_NAME --name YourName
\`\`\`

EOF
else
  cat >> README.md << EOF
## Running the Application

Run the basic application:
\`\`\`
python run.py
\`\`\`

EOF
fi

if [[ "$USE_PYTEST" == "Y" ]]; then
  cat >> README.md << EOF
## Running Tests

Run tests with pytest:
\`\`\`
pytest
\`\`\`

Or with coverage:
\`\`\`
pytest --cov=$PACKAGE_NAME
\`\`\`

EOF
fi

if [[ "$USE_DOCKER" == "Y" ]]; then
  cat >> README.md << EOF
## Docker

Build and run with Docker:
\`\`\`
docker build -t $PROJECT_NAME .
docker run $PROJECT_NAME
\`\`\`

EOF
fi

# Create setup.py (now README.md exists)
cat > setup.py << EOF
"""Setup script for the package."""

from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read the version from the package
about = {}
with open(os.path.join("src", "$PACKAGE_NAME", "__version__.py"), encoding="utf-8") as f:
    exec(f.read(), about)

# Define package dependencies based on project type
install_requires = []
EOF

if [[ "$CREATE_STREAMLIT" == "Y" ]]; then
  cat >> setup.py << EOF
# Streamlit project dependencies
install_requires.extend([
    "streamlit>=1.30.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
])
EOF
fi

cat >> setup.py << EOF

setup(
    name="$PROJECT_NAME",
    version=about["__version__"],
    description="$DESCRIPTION",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="$AUTHOR",
    author_email="$EMAIL",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=$PY_VERSION",
    install_requires=install_requires,
EOF

if [[ "$CREATE_CLI" == "Y" ]]; then
  cat >> setup.py << EOF
    entry_points={
        "console_scripts": [
            "$PACKAGE_NAME=$PACKAGE_NAME.main:main",
        ],
    },
EOF
fi

cat >> setup.py << EOF
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: ${PY_VERSION/\./}",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
EOF

# Create requirements files
cat > requirements.txt << EOF
# Core dependencies
EOF

if [[ "$CREATE_STREAMLIT" == "Y" ]]; then
  cat >> requirements.txt << EOF
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
EOF
fi

# Create development requirements
cat > requirements-dev.txt << EOF
-r requirements.txt
# Development dependencies
EOF

# Add testing dependencies
if [[ "$USE_PYTEST" == "Y" ]]; then
  cat >> requirements-dev.txt << EOF
pytest>=7.4.0
pytest-cov>=4.1.0
EOF
fi

# Add code quality tools
if [[ "$USE_BLACK" == "Y" ]]; then
  cat >> requirements-dev.txt << EOF
black>=23.7.0
EOF
fi

if [[ "$USE_FLAKE8" == "Y" ]]; then
  cat >> requirements-dev.txt << EOF
flake8>=6.1.0
EOF
fi

# Add a README file for the project summary
cat >> README.md << EOF

## Project Summary

This project was created with the Python Project Generator script.
EOF

# Create Docker configuration
if [[ "$USE_DOCKER" == "Y" ]]; then
  # Create Dockerfile
  cat > Dockerfile << EOF
FROM python:$PY_VERSION-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .
EOF

  if [[ "$CREATE_STREAMLIT" == "Y" ]]; then
    cat >> Dockerfile << EOF

# Expose the Streamlit port
EXPOSE 8501

# Set entrypoint to run Streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
EOF
  elif [[ "$CREATE_CLI" == "Y" ]]; then
    cat >> Dockerfile << EOF

# Set entrypoint
ENTRYPOINT ["$PACKAGE_NAME"]
EOF
  else
    cat >> Dockerfile << EOF

# Run application
CMD ["python", "-m", "$PACKAGE_NAME.main"]
EOF
  fi

  # Create .dockerignore
  cat > .dockerignore << EOF
# Version Control
.git/
.gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
*.egg-info/

# Tests
.pytest_cache/
.coverage
htmlcov/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
Dockerfile
EOF
fi

# Create convenience scripts
cat > run_tests.sh << 'EOF'
#!/bin/bash
# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
  source venv/Scripts/activate
fi

# Run the tests with proper Python path
python -m pytest "$@"
EOF
chmod +x run_tests.sh

if [[ "$CREATE_CLI" == "Y" ]]; then
  cat > run_cli.sh << EOF
#!/bin/bash
# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
  source venv/Scripts/activate
fi

# Run the CLI application
python -m $PACKAGE_NAME.main "\$@"
EOF
  chmod +x run_cli.sh
elif [[ ! "$CREATE_STREAMLIT" == "Y" ]]; then
  cat > run_app.sh << EOF
#!/bin/bash
# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
  source venv/Scripts/activate
fi

# Run the application
python run.py
EOF
  chmod +x run_app.sh
fi

# Setup code quality tools
if [[ "$USE_FLAKE8" == "Y" ]]; then
  cat > .flake8 << 'EOF'
[flake8]
max-line-length = 88
extend-ignore = E203
exclude = .git,__pycache__,build,dist
EOF
fi

if [[ "$USE_BLACK" == "Y" ]]; then
  cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/'''
EOF
fi

# Install development dependencies using run_in_venv
run_in_venv pip install -r requirements-dev.txt
echo "Development dependencies installed in virtual environment"

# Now install the package in development mode
run_in_venv pip install -e .
echo "Package installed in development mode in virtual environment"

# Store the original directory and project info right at the start
ORIGINAL_DIR=$(pwd)
PROJECT_PATH="$ORIGINAL_DIR"

# Define the write_final_message function
write_final_message() {
  echo ""
  echo "======================================================"
  echo "  Project '$PROJECT_NAME' created successfully!"
  echo "======================================================"
  echo ""
  echo "Project location: $PROJECT_PATH"
  echo ""
  echo "Features enabled:"
  [[ "$USE_PYTEST" == "Y" ]] && echo "  - Testing with pytest"
  [[ "$CREATE_CLI" == "Y" ]] && echo "  - Command-line interface"
  [[ "$CREATE_STREAMLIT" == "Y" ]] && echo "  - Streamlit application"
  [[ "$USE_DOCKER" == "Y" ]] && echo "  - Docker configuration"
  [[ "$USE_BLACK" == "Y" ]] && echo "  - Code formatting with Black"
  [[ "$USE_FLAKE8" == "Y" ]] && echo "  - Code linting with Flake8"
  echo ""
  echo "Next steps:"
  echo "  cd $PROJECT_NAME"
  echo "  source venv/bin/activate"
  echo ""
}

# Write final message at the end
write_final_message

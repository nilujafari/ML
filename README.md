## Setup

### macOS / Linux

```bash
# Clone the repository
git clone <repo-url>
cd ML

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook main.ipynb
```

### Windows

```bat
:: Clone the repository
git clone <repo-url>
cd ML

:: Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt

:: Launch the notebook
jupyter notebook main.ipynb
```

## Dependencies

Key libraries used:

- `numpy` ? numerical computations
- `pandas` ? data loading and manipulation
- `matplotlib` ? data visualization
- `scikit-learn` ? preprocessing and KNN classifier
- `ipykernel` ? Jupyter notebook support

See `requirements.txt` for the full list with pinned versions.

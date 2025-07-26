# Music Genre Classification

A machine learning project that classifies music genres using audio features extracted from music samples. This project compares the performance of three different classification algorithms: k-Nearest Neighbors (k-NN), Decision Tree, and Logistic Regression.

## Research Paper

This project is accompanied by a detailed research paper that discusses the methodology, experiments, and results in depth.

**Paper:** [Music Genre Classification Analysis.pdf](./Music_Genre_Classification_Analysis.pdf)

The paper covers:
- Literature review of music genre classification techniques
- Detailed methodology and experimental setup
- Comprehensive results analysis and discussion
- Comparison with existing approaches
- Conclusions and future work

## Dataset

This project uses the **GTZAN Dataset** for music genre classification, which contains audio features extracted from music samples across 10 different genres including blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock.

**Dataset Source:** [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)

**Dataset files needed:**
- `features_3_sec.csv` - Features extracted from 3-second audio clips
- `features_30_sec.csv` - Features extracted from 30-second audio clips

**To download and set up the dataset:**

### Option 1: Direct Download
1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)
2. Click "Download" (requires free Kaggle account)
3. Extract the downloaded zip file
4. Create a `Data/` folder in your project directory
5. Copy `features_3_sec.csv` and `features_30_sec.csv` to the `Data/` folder

### Option 2: Using Kaggle CLI (Advanced)
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API key setup)
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification

# Extract and organize
unzip gtzan-dataset-music-genre-classification.zip
mkdir Data
mv features_*.csv Data/
```

```
MusicGenreClassification/
├── Data/
│   ├── features_3_sec.csv
│   └── features_30_sec.csv
├── main.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/MusicGenreClassification.git
cd MusicGenreClassification
```

2. Create a virtual environment:
```bash
python -m venv music_genre_venv
```

3. Activate the virtual environment:
```bash
# On macOS/Linux:
source music_genre_venv/bin/activate

# On Windows:
music_genre_venv\Scripts\activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

5. Download the dataset from Kaggle (see Dataset section above) and place it in the `Data/` folder

### Running the Project

```bash
python main.py
```

## Machine Learning Models

The project implements and compares three classification algorithms:

### 1. k-Nearest Neighbors (k-NN)
- **Parameters tuned**: Number of neighbors, weights, distance metrics
- **Best for**: Simple, interpretable classifications
- **Grid search**: Tests different k values, weighting schemes, and distance metrics

### 2. Decision Tree
- **Parameters tuned**: Max depth, min samples split/leaf, splitting criteria
- **Best for**: Feature importance analysis and interpretable rules
- **Grid search**: Optimizes tree structure to prevent overfitting

### 3. Logistic Regression
- **Parameters tuned**: Regularization strength, solvers, max iterations
- **Best for**: Linear decision boundaries and probability estimates
- **Grid search**: Finds optimal regularization and solver combinations

## Features

- **Data Exploration**: Dataset statistics, genre distribution, missing value analysis
- **Feature Analysis**: Correlation matrix visualization and PCA analysis
- **Data Preprocessing**: Feature scaling and label encoding
- **Model Training**: Grid search with cross-validation for hyperparameter tuning
- **Model Evaluation**: 
  - Accuracy scores
  - Classification reports
  - Confusion matrices
  - Per-class accuracy analysis
- **Cross-Validation**: 5-fold cross-validation for robust performance estimates
- **Visualization**: 
  - Feature correlation heatmaps
  - PCA scatter plots
  - Confusion matrix heatmaps

## Project Structure

```
├── main.py                 # Main script with all functions
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── Data/                  # Dataset folder (not in git)
│   ├── features_3_sec.csv
│   └── features_30_sec.csv
└── music_genre_venv/      # Virtual environment (not in git)
```

## Requirements

```
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

## Audio Features

The GTZAN dataset includes various audio features extracted from 30-second and 3-second audio clips such as:
- **Spectral features**: Spectral centroid, bandwidth, rolloff
- **MFCC**: Mel-frequency cepstral coefficients (13 coefficients)
- **Chroma features**: Pitch class profiles (12 features)
- **Tempo**: Beat tracking features
- **Zero crossing rate**: Measure of signal noisiness
- **RMS Energy**: Root mean square energy
- And statistical measures (mean, variance) for each feature

The dataset covers 10 music genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock.

## Expected Output

The program will display:
1. Dataset exploration statistics
2. Feature correlation visualizations
3. PCA analysis plots
4. Cross-validation results for all models
5. Best hyperparameters for each model
6. Test accuracy scores
7. Detailed classification reports
8. Confusion matrices
9. Final results summary table

## Customization

You can modify the project by:
- **Changing the dataset**: Update the file path in `main()` function
- **Adding new models**: Extend the comparison with SVM, Random Forest, etc.
- **Feature engineering**: Add new audio features or feature selection
- **Hyperparameter tuning**: Modify the parameter grids for different optimization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Notes

- The dataset is downloaded from Kaggle and not included in the repository
- Virtual environment folder is excluded from git tracking
- Make sure to activate the virtual environment before running the script
- The GTZAN dataset is a widely-used benchmark dataset in music information retrieval research

---

This project implements and compares multiple machine learning approaches for automatic music genre classification using the GTZAN dataset.
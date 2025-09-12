# LLM Evaluation Framework - Week 1

A comprehensive framework for evaluating Large Language Models (LLMs) on domain-specific tasks, starting with the HumAID humanitarian crisis dataset.

## 🎯 Project Overview

This project implements Week 1 of the Aleph Alpha LLM Evaluation track, focusing on:
- Dataset exploration and analysis
- Identifying real-world evaluation challenges
- Building foundation for custom evaluation pipelines

## 📁 Project Structure

```
llm-evaluation-framework/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup file
├── .gitignore               # Git ignore file
├── .env.example             # Environment variables template
│
├── config/                  # Configuration files
│   ├── __init__.py
│   ├── settings.py          # Global settings
│   └── datasets.yaml        # Dataset configurations
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── data/               # Data handling modules
│   │   ├── __init__.py
│   │   ├── downloader.py   # Dataset download logic
│   │   ├── explorer.py     # Dataset exploration
│   │   └── analyzer.py     # Statistical analysis
│   │
│   ├── visualization/      # Visualization modules
│   │   ├── __init__.py
│   │   └── plotter.py      # Plotting functions
│   │
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── logger.py       # Logging configuration
│       └── helpers.py      # Helper functions
│
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_downloader.py
│   └── test_analyzer.py
│
└── main.py                 # Main entry point
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-evaluation-framework.git
cd llm-evaluation-framework
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the complete exploration pipeline:
```bash
python main.py
```

Advanced options:
```bash
python main.py --dataset "QCRI/HumAID-events"  # Different dataset
python main.py --download-only                  # Only download
python main.py --analyze-only                   # Skip download
python main.py --verbose                        # Verbose output
```

## 📊 Features

- **Automatic dataset discovery**: Detects columns, data types, and splits
- **Statistical analysis**: Word counts, character lengths, distributions
- **Label analysis**: Class balance, distribution metrics
- **Visualizations**: Comprehensive plots and charts
- **Reports**: Text, JSON, and visual reports

## 📈 Output

After running, find results in `outputs/`:
- `reports/`: Analysis reports and statistics
- `visualizations/`: Generated plots
- `data/`: Downloaded datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

## 📄 License

MIT License

## 🙏 Acknowledgments

- Aleph Alpha for the evaluation framework design
- QCRI for the HumAID dataset
- Hugging Face for dataset hosting

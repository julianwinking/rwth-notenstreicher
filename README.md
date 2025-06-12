# 🎓 RWTH Notenstreicher

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rwth-notenstreicher.streamlit.app/)

A Streamlit web application for analyzing RWTH Aachen University transcripts to determine the optimal module exclusion ("Streichung") for GPA improvement.

## Overview

At RWTH Aachen, students can exclude one module grade when they complete their degree within the standard study period (Regelstudienzeit). This tool helps you determine which module exclusion would result in the maximum GPA improvement.

## Getting Started

### Prerequisites
Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application
Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Access Hosted Version
Try the live application at: https://rwth-notenstreicher.streamlit.app/

## Features

### Multiple Input Methods
- **PDF Upload**: Upload your RWTH transcript PDF for automatic parsing
- **Manual Entry**: Enter module data manually if PDF parsing fails
- **Sample Data**: Use built-in sample data to explore the tool

### Analysis Features
- Current GPA calculation
- Semester-wise GPA breakdown
- Module exclusion recommendations ranked by improvement potential

## Project Structure

```
rwth-notenstreicher/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   └── sample_modules.json    # Sample data for testing
└── src/
    ├── analysis.py            # GPA calculation and analysis functions
    └── importer.py            # PDF import and data parsing
```

## Legal Notice

This tool is for educational purposes only. Always verify your calculations manually and consult with your academic advisor before making decisions about module exclusions.

## Reference

- [RWTH Module Grade Exclusion Rules](https://www.rwth-aachen.de/cms/root/studium/im-studium/pruefungsangelegenheiten/pruefungen/~whcqh/modulnotenstreichung-bei-regelstudienzei/?lidx=1)

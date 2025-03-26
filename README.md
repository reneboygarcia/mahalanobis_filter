# Compressive Strength Analysis Using Mahalanobis Distance Filter

## Table of Contents
- [Compressive Strength Analysis Using Mahalanobis Distance Filter](#compressive-strength-analysis-using-mahalanobis-distance-filter)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Purpose](#purpose)
  - [Technical Background](#technical-background)
    - [Key Features](#key-features)
  - [Code Structure](#code-structure)
    - [Classes](#classes)
      - [`MahalanobisFilter`](#mahalanobisfilter)
    - [Functions](#functions)
      - [Data Processing](#data-processing)
      - [Visualization](#visualization)
  - [Interactive Visualization with Plotly and Dash](#interactive-visualization-with-plotly-and-dash)
    - [Plotly Express Implementation](#plotly-express-implementation)
    - [Dash Dashboard](#dash-dashboard)
  - [Usage Example](#usage-example)
  - [Installation](#installation)
  - [Best Practices](#best-practices)
  - [References](#references)

## Overview
This Jupyter notebook implements a statistical analysis tool for identifying outliers in paired compressive strength test data using the Mahalanobis distance method. The implementation is specifically designed for analyzing paired samples in materials testing, following standards such as ASTM C39, and provides robust statistical validation of test results through automated outlier detection.

![Combined Analysis Graph](/images/combined_graph.png)
*Figure 1: Combined visualization showing (a) original scatter plot with paired data points, (b) histogram of Mahalanobis distances with three-sigma threshold, and (c) filtered results highlighting inliers and outliers with confidence ellipse*


This video explains the theory behind Mahalanobis Distances

[![YouTube video](https://img.youtube.com/vi/-F1f5mefSi0/0.jpg)](https://www.youtube.com/watch?v=-F1f5mefSi0)

## Purpose
In materials testing, particularly with concrete specimens, we often need to evaluate whether paired test results are statistically consistent. This tool helps identify potential outliers that may indicate:
- Testing equipment issues
- Specimen preparation problems
- Material inconsistencies
- Data recording errors

## Technical Background
The Mahalanobis distance is a multi-dimensional generalization of measuring how many standard deviations away a point is from the mean of a distribution. Unlike simple threshold methods, it accounts for the correlation between variables and the shape of the data distribution.

### Key Features
- Implements a robust Mahalanobis distance-based outlier detection
- Uses a three-sigma threshold (99.7% confidence interval)
- Provides visualization tools for data analysis
- Handles paired sample data typical in materials testing
- Interactive visualizations with Plotly Express
- Web-based dashboard with Dash for real-time analysis

## Code Structure

### Classes
#### `MahalanobisFilter`
- **Purpose**: Core class for outlier detection
- **Key Methods**:
  - `fit()`: Calculates mean and inverse covariance matrix
  - `mahalanobis_distance()`: Computes distance for individual points
  - `filter()`: Separates data into inliers and outliers

### Functions
#### Data Processing
- `process_data()`: Handles data preparation and cleaning
  - Extracts paired samples
  - Removes null values
  - Creates a structured DataFrame

#### Visualization
- `plot_ellipse()`: Creates statistical confidence ellipses
- `create_plots()`: Generates three analysis plots:
  1. Original scatter plot of paired data
  2. Histogram of Mahalanobis distances
  3. Filtered results with three-sigma ellipse and regression line

## Interactive Visualization with Plotly and Dash

### Plotly Express Implementation
The project now includes a Jupyter notebook (`mahalanobis_filter_plotly.ipynb`) that demonstrates how to use Plotly Express for creating interactive visualizations:

- **2D Scatter Plots**: Interactive scatter plots with hover information showing Mahalanobis distances and sample details
- **3D Visualization**: 3D scatter plot for visualizing all three samples simultaneously
- **Custom Styling**: Enhanced visual appearance with custom color schemes and layouts
- **Interactive Features**: Zoom, pan, hover tooltips, and data exploration capabilities

### Dash Dashboard
A web-based interactive dashboard (`mahalanobis_filter_dash.py`) has been implemented using Dash:

- **Real-time Analysis**: Adjust parameters and see results immediately
- **Sample Selection**: Choose different sample pairs for analysis
- **Confidence Level Control**: Adjust the sigma value for the confidence ellipse
- **Statistics Panel**: View summary statistics about the data and outliers
- **Outlier Table**: Detailed information about detected outliers
- **Responsive Design**: Modern UI with responsive layout

To run the dashboard:
```bash
python mahalanobis_filter_dash.py
```
Then open your web browser and navigate to: http://127.0.0.1:8050/

## Usage Example
- Pandas
- Matplotlib
- SciPy
- Plotly
- Dash

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Best Practices
1. Always inspect the visualizations to confirm the automated filtering makes practical sense
2. Consider the context of your testing program when interpreting outliers
3. Document any outliers removed from analysis for quality control purposes
4. Regularly validate the filter parameters against your specific testing requirements
5. Use the interactive dashboard for exploring different confidence levels and their impact on outlier detection

## References
- [ChatGPT Reference](https://chatgpt.com/share/67262cf6-0d04-8007-8e75-56fe155a0fbd)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
- [Dash Documentation](https://dash.plotly.com/)
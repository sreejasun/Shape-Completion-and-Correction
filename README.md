# Shape Completion and Correction

## Overview

This Streamlit app allows users to upload a CSV file containing path coordinates, apply refined hybrid smoothing techniques to correct and smooth the paths, and then download the corrected CSV file. The app provides side-by-side comparisons of the original and smoothed data.

## Features

- **CSV Upload**: Upload a CSV file containing path coordinates.
- **Refined Hybrid Smoothing**: Apply a hybrid smoothing method to correct and smooth path data.
- **Visual Comparison**: View side-by-side plots of the original and smoothed data.
- **Download**: Download the corrected CSV file.

## Requirements

To run this app, you need to have the following Python libraries installed:

- `streamlit`
- `pandas`
- `numpy`
- `svgwrite`
- `cairosvg`
- `matplotlib`
- `scipy`

You can install these libraries using pip:

```bash
pip install streamlit pandas numpy svgwrite cairosvg matplotlib scipy
```
## Usage
- Upload CSV File: Click the "Upload your CSV file" button and select a CSV file from your local machine.
- View Comparison: The app will display a side-by-side comparison of the original and smoothed data.
- Download Corrected CSV: Click the "Download corrected CSV" link to download the corrected path data.

## CSV File Format
The input CSV file should have the following columns:
- Shape_ID: Identifier for each path.
- Point_ID: Identifier for each point within the path.
- X: X coordinate of the point.
- Y: Y coordinate of the point.

## Code Explanation
- **read_csv**: Reads the input CSV file and parses path coordinates.
- **write_csv**: Writes the smoothed path coordinates to a new CSV file.
- **plot**: Plots the paths on a matplotlib axis.
- **refined_hybrid_smoothing**: Applies smoothing based on linearity of the paths.
- **plot_side_by_side_comparison**: Creates a side-by-side comparison plot of the original and smoothed data.
- **main**: The main function that sets up the Streamlit app.

import streamlit as st
import pandas as pd
import numpy as np
import svgwrite
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from io import BytesIO
import base64
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import euclidean

# Function to read CSV
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    path_XYs = []
    
    for path_id in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == path_id]
        XYs = []
        
        for point_id in np.unique(npXYs[:, 1]):
            XY = npXYs[npXYs[:, 1] == point_id][:, 2:]
            XYs.append(XY)
        
        path_XYs.append(XYs)
    
    return path_XYs

# Function to write CSV
def write_csv(output_csv, path_XYs):
    with open(output_csv, 'w') as f:
        for i, paths in enumerate(path_XYs):
            for j, XY in enumerate(paths):
                for k, (x, y) in enumerate(XY):
                    f.write(f"{i},{j},{x:.16E},{y:.16E}\n")

# Function to plot shapes
def plot(paths_XYs, colors, ax):
    for i, XYs in enumerate(paths_XYs):
        c = colors[i % len(colors)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')


# Refined hybrid smoothing method
def refined_hybrid_smoothing(df, linearity_threshold=0.05, smoothing_factor=1):
    df_smoothed = df.copy()

    for shape_id in df['Shape_ID'].unique():
        shape_data = df[df['Shape_ID'] == shape_id]
        x = shape_data['X'].values
        y = shape_data['Y'].values

        # Calculate the distance between consecutive points
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_distance = np.sum(distances)
        straight_line_distance = euclidean((x[0], y[0]), (x[-1], y[-1]))

        # Determine if the shape is linear based on the ratio of the straight-line distance to the total path length
        linearity_ratio = straight_line_distance / total_distance

        if linearity_ratio > (1 - linearity_threshold):  # Treat as linear
            # Straighten the line more aggressively
            x_straight = np.linspace(x[0], x[-1], len(x))
            y_straight = np.linspace(y[0], y[-1], len(y))
            df_smoothed.loc[df['Shape_ID'] == shape_id, 'X'] = x_straight
            df_smoothed.loc[df['Shape_ID'] == shape_id, 'Y'] = y_straight
        else:  # Treat as non-linear
            # Apply gentler spline smoothing to non-linear curves
            tck, u = splprep([x, y], s=smoothing_factor)
            x_smooth, y_smooth = splev(u, tck)
            df_smoothed.loc[df['Shape_ID'] == shape_id, 'X'] = x_smooth
            df_smoothed.loc[df['Shape_ID'] == shape_id, 'Y'] = y_smooth

    return df_smoothed

# Function to plot side by side comparison
def plot_side_by_side_comparison(original_df, smoothed_df):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    axs[0].set_title("Original Data")
    plot_original = []
    for shape_id in original_df['Shape_ID'].unique():
        shape_data = original_df[original_df['Shape_ID'] == shape_id]
        points = shape_data[['X', 'Y']].values
        plot_original.append([points])
    plot(plot_original, ['red', 'green', 'blue', 'orange', 'purple'], axs[0])
    
    axs[1].set_title("Smoothed Data")
    plot_smoothed = []
    for shape_id in smoothed_df['Shape_ID'].unique():
        shape_data = smoothed_df[smoothed_df['Shape_ID'] == shape_id]
        points = shape_data[['X', 'Y']].values
        plot_smoothed.append([points])
    plot(plot_smoothed, ['red', 'green', 'blue', 'orange', 'purple'], axs[1])
    
    return fig

# Streamlit app
def main():
    st.title("Shape Completion and Correction")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read and process the CSV file
        path_XYs = read_csv(uploaded_file)
        
        # Convert to DataFrame for smoothing
        data = []
        for i, paths in enumerate(path_XYs):
            for j, XY in enumerate(paths):
                for point in XY:
                    data.append([i, j, point[0], point[1]])
        df = pd.DataFrame(data, columns=['Shape_ID', 'Point_ID', 'X', 'Y'])
        
        # Apply the refined hybrid smoothing method
        df_refined_hybrid_smoothed = refined_hybrid_smoothing(df, linearity_threshold=0.05, smoothing_factor=1)
        
        # Plot the comparison
        fig = plot_side_by_side_comparison(df, df_refined_hybrid_smoothed)
        st.pyplot(fig)
        
        # Convert smoothed DataFrame back to path_XYs format
        smoothed_path_XYs = []
        for shape_id in df_refined_hybrid_smoothed['Shape_ID'].unique():
            shape_data = df_refined_hybrid_smoothed[df_refined_hybrid_smoothed['Shape_ID'] == shape_id]
            points = shape_data[['X', 'Y']].values
            smoothed_path_XYs.append([points])
        
        # Save and provide download link for the corrected CSV
        output_csv = "corrected_output.csv"
        write_csv(output_csv, smoothed_path_XYs)
        
        with open(output_csv, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{output_csv}">Download corrected CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()

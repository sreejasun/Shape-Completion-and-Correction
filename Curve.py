import streamlit as st
import pandas as pd
import numpy as np
import svgwrite
import cairosvg
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from io import BytesIO
import base64

# Your existing functions here
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

def write_csv(output_csv, path_XYs):
    with open(output_csv, 'w') as f:
        for i, paths in enumerate(path_XYs):
            for j, XY in enumerate(paths):
                for k, (x, y) in enumerate(XY):
                    f.write(f"{i},{j},{x:.16E},{y:.16E}\n")

def plot(paths_XYs, colors, ax):
    for i, XYs in enumerate(paths_XYs):
        c = colors[i % len(colors)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W = max(W, np.max(XY[:, 0]))
            H = max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges', size=(W, H))
    group = dwg.g()
    for path in paths_XYs:
        for XY in path:
            path_data = []
            path_data.append(f"M {XY[0, 0]} {XY[0, 1]}")
            for j in range(1, len(XY)):
                path_data.append(f"L {XY[j, 0]} {XY[j, 1]}")
            if not np.allclose(XY[0], XY[-1]):
                path_data.append("Z")
            group.add(dwg.path(d=" ".join(path_data), fill='none', stroke='black', stroke_width=2))
    dwg.add(group)
    dwg.save()

def svg_to_png(svg_path):
    png_path = svg_path.replace('.svg', '.png')
    cairosvg.svg2png(url=svg_path, write_to=png_path, background_color='white')
    return png_path

def fit_circle(XY):
    def objective(params):
        cx, cy, r = params
        return np.sqrt((XY[:, 0] - cx) ** 2 + (XY[:, 1] - cy) ** 2) - r

    if len(XY) < 3:
        raise ValueError("At least 3 points are required to fit a circle.")

    centroid = np.mean(XY, axis=0)
    initial_guess = [centroid[0], centroid[1], np.mean(np.sqrt(np.sum((XY - centroid) ** 2, axis=1)))]
    result = leastsq(objective, initial_guess)
    cx, cy, r = result[0]
    return cx, cy, r

def detect_circle(XY):
    try:
        cx, cy, r = fit_circle(XY)
        distances = np.sqrt((XY[:, 0] - cx) ** 2 + (XY[:, 1] - cy) ** 2)
        return np.allclose(np.mean(distances), r, atol=0.1)
    except ValueError:
        return False

def detect_straight_line(XY):
    if len(XY) < 2:
        return False
    diffs = np.diff(XY, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.allclose(np.ptp(angles), 0)

def detect_rectangle(XY):
    if len(XY) != 4:
        return False
    distances = [np.linalg.norm(XY[i] - XY[(i + 1) % 4]) for i in range(4)]
    angles = [np.arccos(np.dot((XY[i] - XY[(i + 1) % 4]), (XY[(i + 2) % 4] - XY[(i + 1) % 4])) /
                        (np.linalg.norm(XY[i] - XY[(i + 1) % 4]) * np.linalg.norm(XY[(i + 2) % 4] - XY[(i + 1) % 4]))) for i in range(4)]
    return all(np.allclose(distances[i], distances[(i + 2) % 4]) for i in range(4)) and all(np.allclose(angle, np.pi / 2) for angle in angles)

def detect_polygon(XY):
    num_vertices = len(XY)
    if num_vertices < 3:
        return False
    distances = [np.linalg.norm(XY[i] - XY[(i + 1) % num_vertices]) for i in range(num_vertices)]
    angles = [np.arccos(np.dot((XY[i] - XY[(i + 1) % num_vertices]), (XY[(i + 2) % num_vertices] - XY[(i + 1) % num_vertices])) /
                        (np.linalg.norm(XY[i] - XY[(i + 1) % num_vertices]) * np.linalg.norm(XY[(i + 2) % num_vertices] - XY[(i + 1) % num_vertices]))) for i in range(num_vertices)]
    return np.allclose(distances[0], distances[1]) and np.allclose(distances[1], distances[2]) and np.allclose(distances[2], distances[3]) and np.allclose(np.ptp(angles), 0)

def detect_star(XY):
    if len(XY) < 10:
        return False
    centroid = np.mean(XY, axis=0)
    distances = np.sqrt((XY[:, 0] - centroid[0]) ** 2 + (XY[:, 1] - centroid[1]) ** 2)
    return np.allclose(np.ptp(distances), np.mean(distances))

def complete_curve(XY, adjacent_curves):
    # Placeholder for curve completion logic
    if len(XY) < 2:
        return XY
    
    # Example: Interpolating missing segments using simple linear interpolation
    if len(XY) == 2:
        new_points = np.linspace(XY[0], XY[1], num=10)  # Linear interpolation for demonstration
        return new_points
    return XY

# Streamlit app
def main():
    st.title("Shape Completion and Correction")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read and process the CSV file
        path_XYs = read_csv(uploaded_file)
        
        results = []
        for paths in path_XYs:
            new_paths = []
            for XY in paths:
                if detect_straight_line(XY):
                    st.write("Straight line detected")
                elif detect_circle(XY):
                    st.write("Circle detected")
                elif detect_rectangle(XY):
                    st.write("Rectangle detected")
                elif detect_polygon(XY):
                    st.write("Polygon detected")
                elif detect_star(XY):
                    st.write("Star detected")
                else:
                    st.write("Unknown shape detected")
                
                # Apply curve completion (if necessary)
                completed_curve = complete_curve(XY, paths)  # For demonstration, we don't have adjacent curves logic
                new_paths.append(completed_curve)
            
            results.append(new_paths)
        
        # Plot the original and corrected figures
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
        axs[0].set_title("Original Image")
        plot(path_XYs, colors, axs[0])
        
        axs[1].set_title("Corrected Image")
        plot(results, colors, axs[1])
        st.pyplot(fig)
        
        # Save and provide download link for the corrected CSV
        output_csv = "corrected_output.csv"
        write_csv(output_csv, results)
        
        with open(output_csv, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{output_csv}">Download corrected CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Generate and display SVG and PNG images
        svg_path = output_csv.replace('.csv', '.svg')
        polylines2svg(results, svg_path)
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()


import pandas as pd
from scipy.spatial import KDTree
import laspy
import numpy as np
import os
import glob
import csv
import xml.etree.ElementTree as ET
from scipy.spatial import Delaunay

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
process_dir = os.path.join(base_dir, 'process')
reports_dir = os.path.join(base_dir, 'reports')

# Ensure process and reports directories exist
os.makedirs(process_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

def unit_conversion(value, from_unit, to_unit):
    conversion_factors = {
        ('meters', 'us_survey_feet'): 3.280833333,
        ('us_survey_feet', 'meters'): 1 / 3.280833333,
        ('meters', 'international_feet'): 3.280839895,
        ('international_feet', 'meters'): 1 / 3.280839895,
        ('us_survey_feet', 'international_feet'): 3.280833333 / 3.280839895,
        ('international_feet', 'us_survey_feet'): 3.280839895 / 3.280833333,
        ('meters', 'meters'): 1,
        ('us_survey_feet', 'us_survey_feet'): 1,
        ('international_feet', 'international_feet'): 1
    }
    return value * conversion_factors[(from_unit, to_unit)]

# Ask user for input units
input_unit = input("Enter the unit of the input data (meters/us_survey_feet/international_feet): ").strip().lower().replace(' ', '_')
output_unit = input("Enter the desired output unit (meters/us_survey_feet/international_feet): ").strip().lower().replace(' ', '_')

# Run Point Cloud processing
def process_point_cloud():
    # Find the CSV and LAZ files in the process directory
    csv_files = glob.glob(os.path.join(process_dir, '*.csv'))
    laz_files = glob.glob(os.path.join(process_dir, '*.laz'))

    if len(csv_files) != 1 or len(laz_files) != 1:
        raise FileNotFoundError("Exactly one CSV and one LAZ file must be in the process directory.")

    survey_points_csv = csv_files[0]
    point_cloud_laz = laz_files[0]

    # Load survey points from CSV without headers
    survey_points_df = pd.read_csv(survey_points_csv, header=None, names=['Point Number', 'Northing', 'Easting', 'Elevation', 'Description'])

    # Extracting 'Northing' (Y), 'Easting' (X), 'Elevation' (Z) columns
    survey_points = survey_points_df[['Northing', 'Easting', 'Elevation']].values

    # Load point cloud data from LAZ file
    point_cloud_file = laspy.read(point_cloud_laz)
    point_cloud = np.vstack((point_cloud_file.y, point_cloud_file.x, point_cloud_file.z)).T

    # Create a KDTree for the point cloud
    tree = KDTree(point_cloud)

    # Find the nearest point in the point cloud for each survey point
    distances, indices = tree.query(survey_points)

    # Get the nearest points
    nearest_points = point_cloud[indices]

    # Combine the survey points and their nearest point cloud points
    result_df = pd.DataFrame({
        'Point_Number': survey_points_df['Point Number'],
        'Survey_Northing': survey_points[:, 0],
        'Survey_Easting': survey_points[:, 1],
        'Survey_Elevation': survey_points[:, 2],
        'Nearest_Northing': nearest_points[:, 0],
        'Nearest_Easting': nearest_points[:, 1],
        'Nearest_Elevation': nearest_points[:, 2],
        'Distance': distances
    })

    # Save the results to a new CSV file in the reports directory
    pc_output_csv = os.path.join(reports_dir, 'nearest_point_cloud_points.csv')
    result_df.to_csv(pc_output_csv, index=False)

    # Calculate the RMSE for Northing, Easting, and Elevation
    def calculate_rmse(predicted, actual):
        return np.sqrt(((predicted - actual) ** 2).mean())

    rmse_northing = calculate_rmse(result_df['Survey_Northing'], result_df['Nearest_Northing'])
    rmse_easting = calculate_rmse(result_df['Survey_Easting'], result_df['Nearest_Easting'])
    rmse_elevation = calculate_rmse(result_df['Survey_Elevation'], result_df['Nearest_Elevation'])

    # Generate the report
    report = f"""
    RMSE Report
    ===========
    Point Cloud RMSE
    ----------------
    Northing RMSE: {rmse_northing:.4f}
    Easting RMSE: {rmse_easting:.4f}
    Elevation RMSE: {rmse_elevation:.4f}
    """

    # Save the report to a text file in the reports directory
    pc_report_file = os.path.join(reports_dir, 'PC_RMSE_report.txt')
    with open(pc_report_file, 'w') as f:
        f.write(report)

    # Return the path to the report
    return pc_report_file


# Run TIN processing
def process_tin():
    def read_csv(file_path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            return list(reader)

    def read_landxml_points(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        points = []

        ns = {'landxml': 'http://www.landxml.org/schema/LandXML-1.2'}

        for point in root.findall(".//landxml:Surface/landxml:Definition/landxml:Pnts/landxml:P", ns):
            coords = [float(x) for x in point.text.split()]
            points.append(coords)

        return np.array(points)

    def extract_tin_point(point, tin_points, tri):
        x, y = point[:2]
        simplex = tri.find_simplex(point[:2])
        if simplex == -1:
            return None
        b = tri.transform[simplex, :2].dot(np.array([x, y]) - tri.transform[simplex, 2])
        w = np.r_[b, 1 - b.sum()]
        tin_point = (w[:, np.newaxis] * tin_points[tri.simplices[simplex]]).sum(axis=0)
        return tin_point

    csv_file = next((os.path.join(process_dir, f) for f in os.listdir(process_dir) if f.endswith('.csv')), None)
    landxml_file = next((os.path.join(process_dir, f) for f in os.listdir(process_dir) if f.endswith('.xml')), None)

    if not csv_file or not landxml_file:
        print("Error: Could not find required input files in the process directory.")
        return

    survey_points = read_csv(csv_file)
    tin_points = read_landxml_points(landxml_file)

    # Create Delaunay triangulation
    tri = Delaunay(tin_points[:, :2])

    output_data = []
    differences = []

    for point in survey_points:
        point_number = point[0]
        survey_easting = float(point[1])
        survey_northing = float(point[2])
        survey_elevation = float(point[3])

        tin_point = extract_tin_point([survey_easting, survey_northing], tin_points, tri)

        if tin_point is not None:
            tin_easting, tin_northing, tin_elevation = tin_point
            diff = np.array([survey_easting - tin_easting,
                             survey_northing - tin_northing,
                             survey_elevation - tin_elevation])

            output_data.append({
                'Point Number': point_number,
                'Northing (survey)': survey_northing,
                'Easting (survey)': survey_easting,
                'Elevation (survey)': survey_elevation,
                'Northing (TIN)': tin_northing,
                'Easting (TIN)': tin_easting,
                'Elevation (TIN)': tin_elevation
            })
            differences.append(diff)
        else:
            print(f"Warning: Could not extract TIN point for survey point {point_number}")

    differences = np.array(differences)
    rmse = np.sqrt(np.mean(differences**2, axis=0))
    max_diff = np.max(np.abs(differences), axis=0)

    tin_output_csv = os.path.join(reports_dir, 'comparison_points.csv')

    with open(tin_output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_data[0].keys())
        writer.writeheader()
        writer.writerows(output_data)

    tin_report_file = os.path.join(reports_dir, 'TIN_RMSE_report.txt')

    with open(tin_report_file, 'w') as f:
        f.write(f"""
        RMSE Report
        ===========
        TIN RMSE
        --------
        Northing RMSE: {rmse[0]:.4f}
        Easting RMSE: {rmse[1]:.4f}
        Elevation RMSE: {rmse[2]:.4f}
        """)

    # Return the path to the report
    return tin_report_file


# Delete the process directory
def delete_process_files():
    for file in os.listdir(process_dir):
        file_path = os.path.join(process_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")

# Combine reports
def main():
    print("Processing Point Cloud...")
    pc_report = process_point_cloud()
    print(f"Point Cloud RMSE report saved to '{pc_report}'")

    print("Processing TIN...")
    tin_report = process_tin()
    print(f"TIN RMSE report saved to '{tin_report}'")

    # Combine reports
    final_report_path = os.path.join(reports_dir, 'final_combined_RMSE_report.txt')
    with open(final_report_path, 'w') as final_report:
        with open(pc_report, 'r') as f:
            final_report.write(f.read())
            final_report.write("\n\n")

        with open(tin_report, 'r') as f:
            final_report.write(f.read())

    print(f"Final combined RMSE report saved to '{final_report_path}'")
    
    # Delete the process files
    delete_process_files()

if __name__ == "__main__":
    main()
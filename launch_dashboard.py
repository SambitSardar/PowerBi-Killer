import os
import sys
import json
from pathlib import Path
from interactive_dashboard import InteractiveDashboard

def main():
    # Check if a file path was provided
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        # Look for JSON files in the current directory
        json_files = list(Path('.').glob('*.json'))
        
        if not json_files:
            print("Error: No JSON files found in the current directory.")
            print("Please provide a JSON file path as an argument:")
            print("python launch_dashboard.py path/to/your/data.json")
            return
        
        # Use the first JSON file found
        json_file_path = json_files[0]
    
    # Load the JSON data
    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        # Get the file name for the dashboard title
        file_path = Path(json_file_path)
        base_name = file_path.stem
        dashboard_title = f"{base_name} - Interactive Dashboard"
        
        print(f"\nLoading dashboard for: {json_file_path}")
        print(f"Data contains {len(json_data)} records")
        
        # Create and run the dashboard
        dashboard = InteractiveDashboard(json_data, title=dashboard_title)
        dashboard.run_dashboard(debug=False)
        
    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {json_file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
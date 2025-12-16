# Excel to CSV to JSON Converter with AI Analysis

This application allows users to convert Excel files (.xlsx, .xls, .xlsm, etc.) to CSV format and then to JSON format in one simple operation.

## Features

- Convert various Excel file formats to CSV
- Automatically convert the generated CSV to JSON
- AI-powered data analysis and visualization recommendations
- Data profiling with insights on data types, distributions, and correlations
- Sample visualizations to help understand your data
- Simple and intuitive graphical user interface
- Preserves data structure during conversion

## Requirements

To run this application, you need:

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - openpyxl (for Excel file handling)
  - tkinter (usually comes with Python)
  - scikit-learn (for AI analysis)
  - matplotlib (for visualizations)
  - numpy (for numerical operations)
  - pillow (for image handling)

## Installation

1. Ensure you have Python installed on your system
2. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

1. Run the application:

```
python excel_converter.py
```

2. Click on "Select Excel File" to browse and select your Excel file
3. Click "Convert" to process the file
4. The application will create two files in the same directory as the input file:
   - A CSV file with the same name as the input file
   - A JSON file with the same name as the input file
5. After conversion, click "Analyze with AI" to get data insights and visualization recommendations

## Supported Excel Formats

- .xlsx (Excel 2007+)
- .xls (Excel 97-2003)
- .xlsm (Excel with macros)
- .xlsb (Excel binary)
- .odf, .ods, .odt (OpenDocument formats)

## Notes

- Large Excel files may take longer to process
- The JSON output uses the 'records' orientation, which creates an array of objects where each object represents a row in the Excel file
- The AI analysis provides recommendations for the best visualization types based on your data characteristics
- Sample visualizations are generated to help you understand your data patterns
- 
![Test 1](https://github.com/user-attachments/assets/ad9b843c-69a1-4ba8-bb87-19dc2e4cdb65)


## AI Analysis Features

- **Data Profiling**: Analyzes data types, missing values, cardinality, and distributions
- **Visualization Recommendations**: Suggests appropriate chart types based on data characteristics
- **Sample Visualizations**: Generates example charts to demonstrate data patterns
- **Interactive Dashboard Guidance**: Provides insights on which metrics would work best for interactive dashboards

![Test 2](https://github.com/user-attachments/assets/b2adb2f7-4e63-456f-85d3-0f3604374dc7)


## OUTPUT

- ![Output](https://github.com/user-attachments/assets/dea0fc13-da52-42c0-b0d8-3e5fdc6a3016)

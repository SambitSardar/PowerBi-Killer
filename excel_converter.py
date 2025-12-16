import pandas as pd
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import re
# Import the interactive dashboard module
from interactive_dashboard import InteractiveDashboard

class ExcelConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Excel to CSV to JSON Converter with AI Analysis")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Configure styles
        self.root.configure(bg="#f0f0f0")
        
        # Create UI elements
        self.setup_ui()
        
        # Store JSON data and analyzer
        self.json_data = None
        self.analyzer = None
        self.dashboard = None
    
    def setup_ui(self):
        # Title label
        title_label = tk.Label(
            self.root, 
            text="Excel to CSV to JSON Converter with AI", 
            font=("Arial", 16, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack(pady=15)
        
        # Select file button
        select_btn = tk.Button(
            self.root, 
            text="Select Excel File", 
            command=self.select_file,
            width=20,
            font=("Arial", 10),
            bg="#4CAF50",
            fg="white",
            relief=tk.RAISED
        )
        select_btn.pack(pady=10)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg="#f0f0f0")
        status_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Status label
        self.status_label = tk.Label(
            status_frame, 
            text="Status: Ready", 
            font=("Arial", 10),
            bg="#f0f0f0",
            anchor="w"
        )
        self.status_label.pack(fill=tk.X)
        
        # File info label
        self.file_info = tk.Label(
            status_frame, 
            text="No file selected", 
            font=("Arial", 10),
            bg="#f0f0f0",
            anchor="w",
            justify=tk.LEFT,
            wraplength=460
        )
        self.file_info.pack(fill=tk.X, pady=5)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Convert button
        convert_btn = tk.Button(
            button_frame, 
            text="Convert", 
            command=self.convert_file,
            width=20,
            font=("Arial", 10),
            bg="#2196F3",
            fg="white",
            relief=tk.RAISED,
            state=tk.DISABLED
        )
        self.convert_btn = convert_btn
        convert_btn.pack(side=tk.LEFT, padx=10)
        
        # AI Analysis button
        analyze_btn = tk.Button(
            button_frame, 
            text="Analyze with AI", 
            command=self.analyze_data,
            width=20,
            font=("Arial", 10),
            bg="#9C27B0",
            fg="white",
            relief=tk.RAISED,
            state=tk.DISABLED
        )
        self.analyze_btn = analyze_btn
        analyze_btn.pack(side=tk.RIGHT, padx=10)
        
        # Interactive Dashboard button frame
        dashboard_frame = tk.Frame(self.root, bg="#f0f0f0")
        dashboard_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Interactive Dashboard button
        dashboard_btn = tk.Button(
            dashboard_frame, 
            text="Launch Interactive Dashboard", 
            command=self.launch_dashboard,
            width=30,
            font=("Arial", 10),
            bg="#FF9800",
            fg="white",
            relief=tk.RAISED,
            state=tk.DISABLED
        )
        self.dashboard_btn = dashboard_btn
        dashboard_btn.pack(pady=10)
        
        # Exit button
        exit_btn = tk.Button(
            self.root, 
            text="Exit", 
            command=self.root.destroy,
            width=10,
            font=("Arial", 10),
            bg="#f44336",
            fg="white"
        )
        exit_btn.pack(pady=10)
        
        # Store file path
        self.excel_file_path = None
    
    def select_file(self):
        filetypes = [
            ("Excel files", "*.xlsx;*.xls;*.xlsm;*.xlsb;*.odf;*.ods;*.odt"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=filetypes
        )
        
        if file_path:
            self.excel_file_path = file_path
            file_name = os.path.basename(file_path)
            self.file_info.config(text=f"Selected: {file_name}\nPath: {file_path}")
            self.status_label.config(text="Status: File selected")
            self.convert_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="Status: File selection cancelled")
    
    def convert_file(self):
        if not self.excel_file_path:
            messagebox.showerror("Error", "No file selected!")
            return
        
        try:
            # Update status
            self.status_label.config(text="Status: Converting...")
            self.root.update()
            
            # Get the file name without extension
            file_path = Path(self.excel_file_path)
            base_name = file_path.stem
            
            # Set fixed output directory
            # Use relative path to make it portable
            output_dir = Path(__file__).parent / "Csv & Json conversions"
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output paths
            csv_path = output_dir / f"{base_name}.csv"
            json_path = output_dir / f"{base_name}.json"
            
            # Step 1: Excel to CSV
            df = pd.read_excel(self.excel_file_path)
            df.to_csv(csv_path, index=False)
            
            # Step 2: CSV to JSON
            # Read the CSV back to ensure consistency
            csv_df = pd.read_csv(csv_path)
            
            # Convert to JSON
            json_data = csv_df.to_dict(orient='records')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)
            
            # Store JSON data for analysis
            self.json_data = json_data
            
            # Enable AI Analysis and Dashboard buttons
            self.analyze_btn.config(state=tk.NORMAL)
            self.dashboard_btn.config(state=tk.NORMAL)
            
            # Show success message
            message = f"Conversion successful!\n\nCSV saved at:\n{csv_path}\n\nJSON saved at:\n{json_path}\n\nYou can now use 'Analyze with AI' to get visualization recommendations."
            messagebox.showinfo("Success", message)
            
            # Update status
            self.status_label.config(text="Status: Conversion completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_label.config(text="Status: Conversion failed")
    
    def analyze_data(self):
        """Open a new window to analyze the JSON data and provide visualization recommendations"""
        if not self.json_data:
            messagebox.showerror("Error", "No data available for analysis. Please convert a file first.")
            return
        
        # Update status
        self.status_label.config(text="Status: Analyzing data...")
        self.root.update()
        
        # Create analyzer in a separate thread to avoid freezing UI
        def run_analysis():
            try:
                # Create analyzer
                self.analyzer = DataAnalyzer(self.json_data)
                
                # Run analysis
                analysis_results, recommendations = self.analyzer.analyze_data()
                
                # Show analysis window
                self.show_analysis_window(analysis_results, recommendations)
                
                # Update status
                self.status_label.config(text="Status: Analysis completed")
            except Exception as e:
                messagebox.showerror("Error", f"Analysis error: {str(e)}")
                self.status_label.config(text="Status: Analysis failed")
        
        # Run analysis in a separate thread
        threading.Thread(target=run_analysis).start()
    
    def show_analysis_window(self, analysis_results, recommendations):
        """Show a new window with analysis results and visualization recommendations"""
        # Create a new window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("AI Data Analysis Results")
        analysis_window.geometry("900x700")
        analysis_window.configure(bg="#f0f0f0")
        
        # Create a notebook (tabbed interface)
        notebook = ttk.Notebook(analysis_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Data Profile
        profile_tab = tk.Frame(notebook, bg="#f0f0f0")
        notebook.add(profile_tab, text="Data Profile")
        
        # Tab 2: Visualization Recommendations
        viz_tab = tk.Frame(notebook, bg="#f0f0f0")
        notebook.add(viz_tab, text="Visualization Recommendations")
        
        # Tab 3: Sample Visualizations
        sample_viz_tab = tk.Frame(notebook, bg="#f0f0f0")
        notebook.add(sample_viz_tab, text="Sample Visualizations")
        
        # Populate Data Profile tab
        self._populate_profile_tab(profile_tab, analysis_results)
        
        # Populate Visualization Recommendations tab
        self._populate_recommendations_tab(viz_tab, recommendations)
        
        # Populate Sample Visualizations tab
        self._populate_sample_viz_tab(sample_viz_tab)
    
    def _populate_profile_tab(self, tab, analysis_results):
        """Populate the data profile tab with analysis results"""
        # Create a scrolled text widget
        text_widget = scrolledtext.ScrolledText(tab, wrap=tk.WORD, width=80, height=30)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Insert header
        text_widget.insert(tk.END, "DATA PROFILE SUMMARY\n", "header")
        text_widget.insert(tk.END, "===================\n\n")
        
        # Configure tags for formatting
        text_widget.tag_configure("header", font=("Arial", 14, "bold"))
        text_widget.tag_configure("section", font=("Arial", 12, "bold"))
        text_widget.tag_configure("subsection", font=("Arial", 10, "bold"))
        text_widget.tag_configure("normal", font=("Arial", 10))
        
        # Data Types
        text_widget.insert(tk.END, "1. DATA TYPES\n", "section")
        data_types = analysis_results.get('data_types', {})
        for dtype, columns in data_types.items():
            text_widget.insert(tk.END, f"\n{dtype.capitalize()}: ", "subsection")
            text_widget.insert(tk.END, f"{', '.join(columns)}\n", "normal")
        
        # Missing Values
        text_widget.insert(tk.END, "\n\n2. MISSING VALUES\n", "section")
        missing_values = analysis_results.get('missing_values', {})
        if missing_values and len(missing_values.get('missing_count', {})) > 0:
            text_widget.insert(tk.END, "\nColumns with missing values:\n", "subsection")
            for col, count in missing_values.get('missing_count', {}).items():
                percent = missing_values.get('missing_percent', {}).get(col, 0)
                text_widget.insert(tk.END, f"{col}: {count} missing values ({percent:.2f}%)\n", "normal")
        else:
            text_widget.insert(tk.END, "\nNo missing values found.\n", "normal")
        
        # Numeric Distribution
        text_widget.insert(tk.END, "\n\n3. NUMERIC DISTRIBUTION\n", "section")
        numeric_dist = analysis_results.get('numeric_distribution', {})
        
        # Basic stats
        basic_stats = numeric_dist.get('basic_stats', {})
        if basic_stats:
            text_widget.insert(tk.END, "\nBasic Statistics:\n", "subsection")
            for col, stats in basic_stats.items():
                text_widget.insert(tk.END, f"\n{col}:\n", "subsection")
                for stat, value in stats.items():
                    text_widget.insert(tk.END, f"  {stat}: {value}\n", "normal")
        
        # Outliers
        outliers = numeric_dist.get('outliers', {})
        if outliers:
            text_widget.insert(tk.END, "\nOutliers Detected:\n", "subsection")
            for col, count in outliers.items():
                text_widget.insert(tk.END, f"{col}: {count} outliers\n", "normal")
        
        # Correlations
        text_widget.insert(tk.END, "\n\n4. CORRELATIONS\n", "section")
        correlations = analysis_results.get('correlations', {})
        high_corr = correlations.get('high_correlations', {})
        if high_corr:
            text_widget.insert(tk.END, "\nHigh Correlations (>0.7):\n", "subsection")
            for pair, corr in high_corr.items():
                text_widget.insert(tk.END, f"{pair}: {corr:.3f}\n", "normal")
        else:
            text_widget.insert(tk.END, "\nNo high correlations found.\n", "normal")
        
        # Time Series
        text_widget.insert(tk.END, "\n\n5. TIME SERIES DETECTION\n", "section")
        time_series = analysis_results.get('time_series', {})
        datetime_cols = time_series.get('potential_datetime_columns', [])
        if datetime_cols:
            text_widget.insert(tk.END, "\nPotential datetime columns:\n", "subsection")
            for col in datetime_cols:
                text_widget.insert(tk.END, f"{col}\n", "normal")
        else:
            text_widget.insert(tk.END, "\nNo datetime columns detected.\n", "normal")
        
        # Make text widget read-only
        text_widget.configure(state='disabled')
    
    def _populate_recommendations_tab(self, tab, recommendations):
        """Populate the visualization recommendations tab"""
        # Create a scrolled text widget
        text_widget = scrolledtext.ScrolledText(tab, wrap=tk.WORD, width=80, height=30)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Insert header
        text_widget.insert(tk.END, "VISUALIZATION RECOMMENDATIONS\n", "header")
        text_widget.insert(tk.END, "=============================\n\n")
        
        # Configure tags for formatting
        text_widget.tag_configure("header", font=("Arial", 14, "bold"))
        text_widget.tag_configure("chart_type", font=("Arial", 12, "bold"), foreground="#0066cc")
        text_widget.tag_configure("description", font=("Arial", 11, "bold"))
        text_widget.tag_configure("fields", font=("Arial", 10, "italic"), foreground="#006600")
        text_widget.tag_configure("reason", font=("Arial", 10))
        text_widget.tag_configure("separator", font=("Arial", 10))
        
        # Add recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                text_widget.insert(tk.END, f"{i}. {rec['chart_type']}\n", "chart_type")
                text_widget.insert(tk.END, f"Description: {rec['description']}\n", "description")
                text_widget.insert(tk.END, f"Fields: {', '.join(rec['fields'])}\n", "fields")
                text_widget.insert(tk.END, f"Reason: {rec['reason']}\n", "reason")
                text_widget.insert(tk.END, "\n" + "-"*50 + "\n\n", "separator")
        else:
            text_widget.insert(tk.END, "No visualization recommendations available.\n")
        
        # Make text widget read-only
        text_widget.configure(state='disabled')
    
    def _populate_sample_viz_tab(self, tab):
        """Populate the sample visualizations tab"""
        # Create a frame for the visualizations
        viz_frame = tk.Frame(tab, bg="#f0f0f0")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add a label
        label = tk.Label(
            viz_frame,
            text="Sample Visualizations Based on Your Data",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        )
        label.pack(pady=10)
        
        # Create a frame for the matplotlib canvas
        canvas_frame = tk.Frame(viz_frame, bg="#f0f0f0")
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Generate visualizations
        if self.analyzer:
            self.analyzer.generate_sample_visualizations(canvas_frame)
            
    def launch_dashboard(self):
        """Launch an interactive dashboard for the JSON data"""
        if not self.json_data:
            messagebox.showerror("Error", "No data available for dashboard. Please convert a file first.")
            return
        
        # Update status
        self.status_label.config(text="Status: Launching interactive dashboard...")
        self.root.update()
        
        try:
            # Create dashboard instance
            file_path = Path(self.excel_file_path)
            base_name = file_path.stem
            dashboard_title = f"{base_name} - Interactive Dashboard"
            
            self.dashboard = InteractiveDashboard(self.json_data, title=dashboard_title)
            
            # Show info message before launching dashboard
            messagebox.showinfo("Dashboard Launching", 
                              f"The interactive dashboard will now start.\n\n"
                              f"Access the dashboard at:\n"
                              f"http://127.0.0.1:8050\n\n"
                              f"The main application will be unresponsive until you close the dashboard.\n"
                              f"Press Ctrl+C in the terminal to stop the dashboard when finished.")
            
            # Run the dashboard (this will block until dashboard is closed)
            result = self.dashboard.run_dashboard(debug=False)
            
            # Update status after dashboard is closed
            self.status_label.config(text=f"Status: {result}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch dashboard: {str(e)}")
            self.status_label.config(text="Status: Dashboard launch failed")

class DataAnalyzer:
    def __init__(self, json_data):
        """Initialize the data analyzer with JSON data"""
        self.json_data = json_data
        self.df = pd.DataFrame(json_data)
        self.analysis_results = {}
        self.visualization_recommendations = []
    
    def analyze_data(self):
        """Perform comprehensive analysis on the data"""
        # Basic data profiling
        self._analyze_data_types()
        self._analyze_missing_values()
        self._analyze_cardinality()
        self._analyze_numeric_distribution()
        self._analyze_correlations()
        self._analyze_time_series()
        
        # Generate visualization recommendations
        self._generate_recommendations()
        
        return self.analysis_results, self.visualization_recommendations
    
    def _analyze_data_types(self):
        """Analyze the data types of each column"""
        type_counts = {}
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if np.issubdtype(dtype, np.number):
                if self.df[col].nunique() <= 10 and self.df[col].nunique() / len(self.df) < 0.05:
                    col_type = 'categorical_numeric'
                else:
                    col_type = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_type = 'datetime'
            elif self.df[col].nunique() <= 10 and self.df[col].nunique() / len(self.df) < 0.2:
                col_type = 'categorical'
            else:
                col_type = 'text'
                
            if col_type not in type_counts:
                type_counts[col_type] = []
            type_counts[col_type].append(col)
        
        self.analysis_results['data_types'] = type_counts
    
    def _analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_info = pd.DataFrame({
            'missing_count': missing_data,
            'missing_percent': missing_percent
        })
        self.analysis_results['missing_values'] = missing_info[missing_info['missing_count'] > 0].to_dict()
    
    def _analyze_cardinality(self):
        """Analyze cardinality (number of unique values) for each column"""
        cardinality = {}
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            unique_percent = (unique_count / len(self.df)) * 100
            cardinality[col] = {
                'unique_count': unique_count,
                'unique_percent': unique_percent
            }
        self.analysis_results['cardinality'] = cardinality
    
    def _analyze_numeric_distribution(self):
        """Analyze distribution of numeric columns"""
        numeric_stats = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Basic statistics
            stats = self.df[numeric_cols].describe().to_dict()
            numeric_stats['basic_stats'] = stats
            
            # Check for skewness
            skewness = {}
            for col in numeric_cols:
                try:
                    skew_value = self.df[col].skew()
                    if abs(skew_value) > 1:
                        skewness[col] = skew_value
                except:
                    pass
            numeric_stats['skewness'] = skewness
            
            # Check for outliers using IQR
            outliers = {}
            for col in numeric_cols:
                try:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_count = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                                    (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                    if outlier_count > 0:
                        outliers[col] = outlier_count
                except:
                    pass
            numeric_stats['outliers'] = outliers
        
        self.analysis_results['numeric_distribution'] = numeric_stats
    
    def _analyze_correlations(self):
        """Analyze correlations between numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                corr_matrix = self.df[numeric_cols].corr().abs()
                # Get pairs with high correlation (above 0.7)
                high_corr = {}
                for i in range(len(corr_matrix.columns)):
                    for j in range(i):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]
                            high_corr[f"{col1} - {col2}"] = corr_matrix.iloc[i, j]
                
                self.analysis_results['correlations'] = {
                    'high_correlations': high_corr
                }
            except:
                self.analysis_results['correlations'] = {
                    'error': 'Could not compute correlations'
                }
    
    def _analyze_time_series(self):
        """Detect and analyze potential time series data"""
        # Check for datetime columns
        datetime_cols = []
        for col in self.df.columns:
            # Check if column is already datetime
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                datetime_cols.append(col)
            else:
                # Try to convert to datetime
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    datetime_cols.append(col)
                except:
                    # Check if column name suggests datetime
                    if any(time_word in col.lower() for time_word in ['date', 'time', 'year', 'month', 'day']):
                        datetime_cols.append(col)
        
        if datetime_cols:
            self.analysis_results['time_series'] = {
                'potential_datetime_columns': datetime_cols
            }
    
    def _generate_recommendations(self):
        """Generate visualization recommendations based on data analysis"""
        recommendations = []
        data_types = self.analysis_results.get('data_types', {})
        
        # Get column lists by type
        numeric_cols = data_types.get('numeric', [])
        categorical_cols = data_types.get('categorical', []) + data_types.get('categorical_numeric', [])
        datetime_cols = data_types.get('datetime', [])
        
        # 1. Numeric distribution visualizations
        if numeric_cols:
            for col in numeric_cols[:3]:  # Limit to first 3 to avoid too many recommendations
                recommendations.append({
                    'chart_type': 'Histogram',
                    'description': f'Distribution of {col}',
                    'fields': [col],
                    'reason': 'To understand the distribution pattern of numeric data'
                })
                
                recommendations.append({
                    'chart_type': 'Box Plot',
                    'description': f'Box plot of {col}',
                    'fields': [col],
                    'reason': 'To identify outliers and quartile distribution'
                })
        
        # 2. Categorical visualizations
        if categorical_cols:
            for col in categorical_cols[:3]:  # Limit to first 3
                recommendations.append({
                    'chart_type': 'Bar Chart',
                    'description': f'Count of {col}',
                    'fields': [col],
                    'reason': 'To compare frequencies across different categories'
                })
                
                recommendations.append({
                    'chart_type': 'Pie Chart',
                    'description': f'Proportion of {col}',
                    'fields': [col],
                    'reason': 'To show proportion of each category in the whole'
                })
        
        # 3. Correlations between numeric variables
        if len(numeric_cols) >= 2:
            recommendations.append({
                'chart_type': 'Scatter Plot',
                'description': f'Relationship between {numeric_cols[0]} and {numeric_cols[1]}',
                'fields': [numeric_cols[0], numeric_cols[1]],
                'reason': 'To identify potential correlations between variables'
            })
            
            recommendations.append({
                'chart_type': 'Heatmap',
                'description': 'Correlation matrix of numeric variables',
                'fields': numeric_cols,
                'reason': 'To visualize correlations between all numeric variables'
            })
        
        # 4. Time series visualizations
        if datetime_cols and numeric_cols:
            recommendations.append({
                'chart_type': 'Line Chart',
                'description': f'Trend of {numeric_cols[0]} over {datetime_cols[0]}',
                'fields': [datetime_cols[0], numeric_cols[0]],
                'reason': 'To analyze trends over time'
            })
        
        # 5. Categorical and numeric combinations
        if categorical_cols and numeric_cols:
            recommendations.append({
                'chart_type': 'Grouped Bar Chart',
                'description': f'Average {numeric_cols[0]} by {categorical_cols[0]}',
                'fields': [categorical_cols[0], numeric_cols[0]],
                'reason': 'To compare a metric across different categories'
            })
        
        # 6. Dashboard-specific recommendations
        if len(numeric_cols) >= 1:
            recommendations.append({
                'chart_type': 'KPI Card',
                'description': f'Key metrics for {numeric_cols[0]}',
                'fields': [numeric_cols[0]],
                'reason': 'To highlight important metrics at a glance'
            })
        
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            recommendations.append({
                'chart_type': 'Filtered Dashboard',
                'description': f'Interactive dashboard with {categorical_cols[0]} as filter',
                'fields': [categorical_cols[0]] + numeric_cols[:2],
                'reason': 'To enable interactive exploration of data by category'
            })
        
        self.visualization_recommendations = recommendations
    
    def generate_sample_visualizations(self, canvas_frame):
        """Generate sample visualizations based on recommendations"""
        # Clear any existing widgets in the frame
        for widget in canvas_frame.winfo_children():
            widget.destroy()
        
        # Get data types for reference
        data_types = self.analysis_results.get('data_types', {})
        numeric_cols = data_types.get('numeric', [])
        categorical_cols = data_types.get('categorical', []) + data_types.get('categorical_numeric', [])
        datetime_cols = data_types.get('datetime', [])
        
        # Create visualizations based on data availability
        if not numeric_cols and not categorical_cols:
            label = tk.Label(canvas_frame, text="No suitable data for visualization found")
            label.pack(pady=20)
            return
        
        # Create a figure with subplots based on available data
        num_plots = min(4, len(self.visualization_recommendations))
        if num_plots == 0:
            return
        
        # Create figure with appropriate size
        fig = Figure(figsize=(10, 8))
        
        # Create subplots
        axes = []
        for i in range(num_plots):
            if i == 0:
                ax = fig.add_subplot(2, 2, i+1)
            else:
                ax = fig.add_subplot(2, 2, i+1)
            axes.append(ax)
        
        # Generate visualizations based on recommendations
        for i, (ax, rec) in enumerate(zip(axes, self.visualization_recommendations[:num_plots])):
            chart_type = rec['chart_type']
            fields = rec['fields']
            
            try:
                if chart_type == 'Histogram' and len(fields) == 1 and fields[0] in numeric_cols:
                    ax.hist(self.df[fields[0]].dropna(), bins=20, alpha=0.7)
                    ax.set_title(f'Histogram of {fields[0]}')
                    ax.set_xlabel(fields[0])
                    ax.set_ylabel('Frequency')
                
                elif chart_type == 'Bar Chart' and len(fields) == 1 and fields[0] in categorical_cols:
                    counts = self.df[fields[0]].value_counts().nlargest(10)
                    counts.plot(kind='bar', ax=ax)
                    ax.set_title(f'Top 10 {fields[0]} Categories')
                    ax.set_ylabel('Count')
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                elif chart_type == 'Scatter Plot' and len(fields) == 2 and all(f in numeric_cols for f in fields):
                    ax.scatter(self.df[fields[0]], self.df[fields[1]], alpha=0.5)
                    ax.set_title(f'Scatter Plot: {fields[0]} vs {fields[1]}')
                    ax.set_xlabel(fields[0])
                    ax.set_ylabel(fields[1])
                
                elif chart_type == 'Line Chart' and len(fields) == 2 and fields[0] in datetime_cols and fields[1] in numeric_cols:
                    # Try to convert to datetime if not already
                    try:
                        x = pd.to_datetime(self.df[fields[0]])
                        temp_df = pd.DataFrame({'date': x, 'value': self.df[fields[1]]})
                        temp_df.set_index('date').resample('D').mean()['value'].plot(ax=ax)
                        ax.set_title(f'Time Series: {fields[1]} over time')
                    except:
                        # Fallback if conversion fails
                        ax.plot(range(len(self.df)), self.df[fields[1]])
                        ax.set_title(f'Trend of {fields[1]}')
                
                elif chart_type == 'Box Plot' and len(fields) == 1 and fields[0] in numeric_cols:
                    ax.boxplot(self.df[fields[0]].dropna())
                    ax.set_title(f'Box Plot of {fields[0]}')
                    ax.set_ylabel(fields[0])
                    ax.set_xticks([1])
                    ax.set_xticklabels([fields[0]])
                
                elif chart_type == 'Pie Chart' and len(fields) == 1 and fields[0] in categorical_cols:
                    counts = self.df[fields[0]].value_counts().nlargest(5)
                    other_count = self.df[fields[0]].value_counts().sum() - counts.sum()
                    if other_count > 0:
                        counts['Other'] = other_count
                    counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
                    ax.set_title(f'Distribution of {fields[0]}')
                    ax.set_ylabel('')
                
                elif chart_type == 'Grouped Bar Chart' and len(fields) == 2 and fields[0] in categorical_cols and fields[1] in numeric_cols:
                    # Get top 5 categories
                    top_cats = self.df[fields[0]].value_counts().nlargest(5).index
                    filtered_df = self.df[self.df[fields[0]].isin(top_cats)]
                    grouped = filtered_df.groupby(fields[0])[fields[1]].mean().sort_values(ascending=False)
                    grouped.plot(kind='bar', ax=ax)
                    ax.set_title(f'Average {fields[1]} by {fields[0]}')
                    ax.set_ylabel(f'Average {fields[1]}')
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                else:
                    # Default visualization if specific chart type can't be created
                    if numeric_cols:
                        ax.hist(self.df[numeric_cols[0]].dropna(), bins=20)
                        ax.set_title(f'Distribution of {numeric_cols[0]}')
                    elif categorical_cols:
                        counts = self.df[categorical_cols[0]].value_counts().nlargest(10)
                        counts.plot(kind='bar', ax=ax)
                        ax.set_title(f'Top 10 {categorical_cols[0]} Categories')
            except Exception as e:
                ax.text(0.5, 0.5, f"Could not create visualization:\n{str(e)}", 
                       horizontalalignment='center', verticalalignment='center')
        
        # Adjust layout
        fig.tight_layout()
        
        # Create canvas and add to frame
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    app = ExcelConverter(root)
    root.mainloop()

if __name__ == "__main__":
    main()
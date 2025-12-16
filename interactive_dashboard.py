import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import threading
import webbrowser
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class InteractiveDashboard:
    def __init__(self, json_data, title="Interactive Data Dashboard"):
        """Initialize the dashboard with JSON data"""
        self.json_data = json_data
        self.df = pd.DataFrame(json_data)
        self.title = title
        self.app = None
        self.server_thread = None
        self.port = 8050
        self.classification_results = {}
        
    def preprocess_data(self):
        """Preprocess data for dashboard and classification"""
        # Store original data types for reference
        self.data_types = {}
        
        # Identify numeric and categorical columns
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        
        for col in self.df.columns:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_cols.append(col)
            # Check if column might be datetime
            elif self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col])
                    datetime_cols.append(col)
                    # Convert to datetime
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    categorical_cols.append(col)
            else:
                categorical_cols.append(col)
        
        self.data_types['numeric'] = numeric_cols
        self.data_types['categorical'] = categorical_cols
        self.data_types['datetime'] = datetime_cols
        
        # Handle missing values
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        
        return self.df
    
    def perform_classification(self):
        """Perform classification on the data using KMeans"""
        numeric_cols = self.data_types.get('numeric', [])
        
        if len(numeric_cols) < 2:
            self.classification_results['status'] = 'Not enough numeric columns for classification'
            return
        
        try:
            # Select numeric columns for clustering
            X = self.df[numeric_cols].copy()
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters (simplified method)
            max_clusters = min(5, len(self.df) // 10) if len(self.df) > 20 else 2
            
            # Apply KMeans
            kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to dataframe
            self.df['Cluster'] = clusters
            
            # Get cluster centers
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            
            # Calculate cluster statistics
            cluster_stats = {}
            for i in range(max_clusters):
                cluster_data = self.df[self.df['Cluster'] == i]
                cluster_stats[f'Cluster {i}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(self.df) * 100,
                    'features': {}
                }
                
                # Calculate statistics for each feature in the cluster
                for col in numeric_cols:
                    cluster_stats[f'Cluster {i}']['features'][col] = {
                        'mean': cluster_data[col].mean(),
                        'median': cluster_data[col].median(),
                        'std': cluster_data[col].std()
                    }
            
            # Try to apply PCA for visualization if we have enough features
            if len(numeric_cols) >= 3:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Add PCA components to dataframe
                self.df['PCA1'] = X_pca[:, 0]
                self.df['PCA2'] = X_pca[:, 1]
                
                # Calculate explained variance
                explained_variance = pca.explained_variance_ratio_
                
                self.classification_results['pca'] = {
                    'components': X_pca,
                    'explained_variance': explained_variance,
                    'feature_importance': dict(zip(numeric_cols, pca.components_[0]))
                }
            
            self.classification_results['status'] = 'success'
            self.classification_results['clusters'] = clusters
            self.classification_results['centers'] = centers
            self.classification_results['cluster_stats'] = cluster_stats
            self.classification_results['features_used'] = numeric_cols
            
        except Exception as e:
            self.classification_results['status'] = f'Classification failed: {str(e)}'
    
    def create_dashboard(self):
        """Create an interactive dashboard using Dash"""
        # Preprocess data
        self.preprocess_data()
        
        # Perform classification
        self.perform_classification()
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Set up the layout
        self.setup_layout()
        
        # Set up callbacks
        self.setup_callbacks()
        
        return self.app
    
    def setup_layout(self):
        """Set up the dashboard layout"""
        # Get column lists by type
        numeric_cols = self.data_types.get('numeric', [])
        categorical_cols = self.data_types.get('categorical', [])
        datetime_cols = self.data_types.get('datetime', [])
        
        # Create dropdown options
        numeric_options = [{'label': col, 'value': col} for col in numeric_cols]
        categorical_options = [{'label': col, 'value': col} for col in categorical_cols]
        datetime_options = [{'label': col, 'value': col} for col in datetime_cols]
        
        # Create the layout
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1(self.title, className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Filters section
            dbc.Row([
                dbc.Col([
                    html.H4("Filters"),
                    html.Div([
                        html.Label("Select Category:"),
                        dcc.Dropdown(
                            id="category-filter",
                            options=categorical_options,
                            value=categorical_cols[0] if categorical_cols else None,
                            multi=False
                        )
                    ]) if categorical_cols else html.Div("No categorical columns available")
                ], width=3),
                
                dbc.Col([
                    html.H4("Chart Controls"),
                    html.Div([
                        html.Label("Select Chart Type:"),
                        dcc.Dropdown(
                            id="chart-type",
                            options=[
                                {"label": "Bar Chart", "value": "bar"},
                                {"label": "Line Chart", "value": "line"},
                                {"label": "Scatter Plot", "value": "scatter"},
                                {"label": "Pie Chart", "value": "pie"},
                                {"label": "Box Plot", "value": "box"}
                            ],
                            value="bar"
                        )
                    ])
                ], width=3),
                
                dbc.Col([
                    html.H4("Metrics"),
                    html.Div([
                        html.Label("Select X-Axis:"),
                        dcc.Dropdown(
                            id="x-axis",
                            options=numeric_options + categorical_options + datetime_options,
                            value=(categorical_cols + datetime_cols + numeric_cols)[0] if 
                                  (categorical_cols + datetime_cols + numeric_cols) else None
                        )
                    ]),
                    html.Div([
                        html.Label("Select Y-Axis:"),
                        dcc.Dropdown(
                            id="y-axis",
                            options=numeric_options,
                            value=numeric_cols[0] if numeric_cols else None
                        )
                    ]) if numeric_cols else html.Div("No numeric columns available")
                ], width=6)
            ]),
            
            html.Hr(),
            
            # Main dashboard content
            dbc.Row([
                # Main visualization
                dbc.Col([
                    html.H4("Main Visualization", className="text-center"),
                    dcc.Graph(id="main-chart")
                ], width=8),
                
                # KPI cards and metrics
                dbc.Col([
                    html.H4("Key Metrics", className="text-center"),
                    html.Div(id="kpi-cards"),
                    html.Hr(),
                    html.H5("Data Summary"),
                    html.Div(id="data-summary")
                ], width=4)
            ]),
            
            html.Hr(),
            
            # Classification results section
            dbc.Row([
                dbc.Col([
                    html.H4("AI Classification Results", className="text-center"),
                    html.Div(id="classification-info"),
                    dcc.Graph(id="cluster-visualization")
                ])
            ]) if self.classification_results.get('status') == 'success' else dbc.Row([
                dbc.Col([
                    html.H4("AI Classification Results", className="text-center"),
                    html.Div(f"Classification Status: {self.classification_results.get('status', 'Not performed')}")
                ])
            ]),
            
            html.Hr(),
            
            # Data table section
            dbc.Row([
                dbc.Col([
                    html.H4("Data Preview", className="text-center"),
                    html.Div([
                        html.Label("Rows to display:"),
                        dcc.Slider(
                            id="row-slider",
                            min=5,
                            max=min(50, len(self.df)),
                            step=5,
                            value=10,
                            marks={i: str(i) for i in range(5, min(50, len(self.df))+1, 5)}
                        )
                    ]),
                    html.Div(id="data-table")
                ])
            ])
        ], fluid=True)
    
    def setup_callbacks(self):
        """Set up the interactive callbacks"""
        # Callback for main chart
        @self.app.callback(
            Output("main-chart", "figure"),
            [Input("chart-type", "value"),
             Input("x-axis", "value"),
             Input("y-axis", "value"),
             Input("category-filter", "value")]
        )
        def update_main_chart(chart_type, x_axis, y_axis, category):
            if not x_axis or (chart_type != "pie" and not y_axis):
                return go.Figure().update_layout(title="Please select valid axes")
            
            # Filter data if category is selected
            filtered_df = self.df
            if category and category in self.df.columns:
                # Get unique values for the category
                unique_values = filtered_df[category].unique()
                if len(unique_values) > 0:
                    # Use the first value as default
                    filtered_df = filtered_df[filtered_df[category] == unique_values[0]]
            
            # Create figure based on chart type
            fig = go.Figure()
            
            try:
                if chart_type == "bar":
                    fig = px.bar(filtered_df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                
                elif chart_type == "line":
                    fig = px.line(filtered_df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
                
                elif chart_type == "scatter":
                    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                
                elif chart_type == "pie":
                    # For pie charts, we need to aggregate the data
                    if x_axis in self.data_types.get('categorical', []):
                        counts = filtered_df[x_axis].value_counts()
                        fig = px.pie(names=counts.index, values=counts.values, title=f"Distribution of {x_axis}")
                    else:
                        fig = go.Figure().update_layout(title="Pie charts require categorical data")
                
                elif chart_type == "box":
                    fig = px.box(filtered_df, y=y_axis, title=f"Distribution of {y_axis}")
                
                # Enhance the layout
                fig.update_layout(
                    template="plotly_white",
                    xaxis_title=x_axis,
                    yaxis_title=y_axis if chart_type != "pie" else "",
                    legend_title="Legend",
                    height=500
                )
                
            except Exception as e:
                fig = go.Figure().update_layout(title=f"Error creating chart: {str(e)}")
            
            return fig
        
        # Callback for KPI cards
        @self.app.callback(
            Output("kpi-cards", "children"),
            [Input("y-axis", "value")]
        )
        def update_kpi_cards(y_axis):
            if not y_axis or y_axis not in self.df.columns:
                return html.Div("Select a numeric column to see KPIs")
            
            try:
                # Calculate KPIs for the selected column
                if pd.api.types.is_numeric_dtype(self.df[y_axis]):
                    mean_val = self.df[y_axis].mean()
                    median_val = self.df[y_axis].median()
                    min_val = self.df[y_axis].min()
                    max_val = self.df[y_axis].max()
                    
                    # Create KPI cards
                    kpi_cards = [
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Average", className="card-title"),
                                html.H3(f"{mean_val:.2f}", className="card-text text-primary")
                            ])
                        ], className="mb-2"),
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Median", className="card-title"),
                                html.H3(f"{median_val:.2f}", className="card-text text-success")
                            ])
                        ], className="mb-2"),
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Range", className="card-title"),
                                html.H3(f"{min_val:.2f} - {max_val:.2f}", className="card-text text-info")
                            ])
                        ], className="mb-2")
                    ]
                    return kpi_cards
                else:
                    return html.Div("Selected column is not numeric")
            except Exception as e:
                return html.Div(f"Error calculating KPIs: {str(e)}")
        
        # Callback for data summary
        @self.app.callback(
            Output("data-summary", "children"),
            [Input("category-filter", "value")]
        )
        def update_data_summary(category):
            try:
                # Basic dataset summary
                summary = [
                    html.P(f"Total Records: {len(self.df)}"),
                    html.P(f"Total Features: {len(self.df.columns)}"),
                    html.P(f"Numeric Features: {len(self.data_types.get('numeric', []))}"),
                    html.P(f"Categorical Features: {len(self.data_types.get('categorical', []))}"),
                    html.P(f"Datetime Features: {len(self.data_types.get('datetime', []))}"),
                ]
                
                # Add category-specific information if selected
                if category and category in self.df.columns:
                    unique_values = self.df[category].nunique()
                    summary.append(html.Hr())
                    summary.append(html.H6(f"{category} Summary:"))
                    summary.append(html.P(f"Unique Values: {unique_values}"))
                    
                    # Show top categories
                    top_cats = self.df[category].value_counts().nlargest(5)
                    summary.append(html.P("Top 5 Values:"))
                    for idx, (cat, count) in enumerate(top_cats.items()):
                        summary.append(html.P(f"{idx+1}. {cat}: {count} records"))
                
                return summary
            except Exception as e:
                return html.Div(f"Error generating summary: {str(e)}")
        
        # Callback for cluster visualization
        @self.app.callback(
            Output("cluster-visualization", "figure"),
            [Input("cluster-visualization", "id")]
        )
        def update_cluster_visualization(_):
            if self.classification_results.get('status') != 'success':
                return go.Figure().update_layout(title="Classification not available")
            
            try:
                # Check if we have PCA results for visualization
                if 'pca' in self.classification_results:
                    # Create scatter plot of PCA components colored by cluster
                    fig = px.scatter(
                        self.df, 
                        x="PCA1", 
                        y="PCA2", 
                        color="Cluster",
                        title="Data Clusters (PCA Visualization)",
                        labels={"PCA1": "Principal Component 1", "PCA2": "Principal Component 2"},
                        color_continuous_scale=px.colors.qualitative.G10
                    )
                    
                    # Add explained variance information
                    explained_var = self.classification_results['pca']['explained_variance']
                    fig.update_layout(
                        xaxis_title=f"PC1 ({explained_var[0]:.2%} variance)",
                        yaxis_title=f"PC2 ({explained_var[1]:.2%} variance)"
                    )
                else:
                    # If no PCA, use the first two numeric features
                    features = self.classification_results['features_used']
                    if len(features) >= 2:
                        fig = px.scatter(
                            self.df, 
                            x=features[0], 
                            y=features[1], 
                            color="Cluster",
                            title=f"Data Clusters ({features[0]} vs {features[1]})",
                            color_continuous_scale=px.colors.qualitative.G10
                        )
                    else:
                        fig = go.Figure().update_layout(title="Not enough features for cluster visualization")
                
                return fig
            except Exception as e:
                return go.Figure().update_layout(title=f"Error creating cluster visualization: {str(e)}")
        
        # Callback for classification info
        @self.app.callback(
            Output("classification-info", "children"),
            [Input("classification-info", "id")]
        )
        def update_classification_info(_):
            if self.classification_results.get('status') != 'success':
                return html.Div("Classification not available")
            
            try:
                # Get cluster statistics
                cluster_stats = self.classification_results.get('cluster_stats', {})
                
                # Create cluster summary
                cluster_summary = []
                for cluster, stats in cluster_stats.items():
                    cluster_summary.append(html.Div([
                        html.H6(f"{cluster} ({stats['size']} records, {stats['percentage']:.1f}%)"),
                        html.Ul([
                            html.Li(f"{feature}: Mean={feat_stats['mean']:.2f}, Median={feat_stats['median']:.2f}")
                            for feature, feat_stats in stats['features'].items()
                        ])
                    ]))
                
                return html.Div([
                    html.P(f"Classification performed using {len(self.classification_results['features_used'])} features"),
                    html.P(f"Number of clusters: {len(cluster_stats)}"),
                    html.Hr(),
                    html.H6("Cluster Summary:"),
                    html.Div(cluster_summary)
                ])
            except Exception as e:
                return html.Div(f"Error displaying classification info: {str(e)}")
        
        # Callback for data table
        @self.app.callback(
            Output("data-table", "children"),
            [Input("row-slider", "value")]
        )
        def update_data_table(rows):
            try:
                # Create a data table with the specified number of rows
                table_header = [
                    html.Thead(html.Tr([html.Th(col) for col in self.df.columns]))
                ]
                
                table_body = [
                    html.Tbody([
                        html.Tr([
                            html.Td(self.df.iloc[i][col]) for col in self.df.columns
                        ]) for i in range(min(rows, len(self.df)))
                    ])
                ]
                
                table = dbc.Table(table_header + table_body, striped=True, bordered=True, hover=True)
                return table
            except Exception as e:
                return html.Div(f"Error creating data table: {str(e)}")
    
    def run_dashboard(self, debug=False):
        """Run the dashboard directly"""
        if not self.app:
            self.create_dashboard()
        
        print(f"\n\n*** DASHBOARD STARTING ***")
        print(f"*** Access your dashboard at: http://127.0.0.1:{self.port} ***")
        print(f"*** Press Ctrl+C to stop the dashboard when finished ***\n\n")
        
        # Non-blocking version
        # Open browser in a separate thread
        def open_browser():
            webbrowser.open_new(f"http://127.0.0.1:{self.port}")
        
        # Start the server in a separate thread
        def run_server():
            try:
                from waitress import serve
                print(f"Starting Waitress server on http://0.0.0.0:{self.port}")
                serve(self.app.server, host='0.0.0.0', port=self.port)
            except ImportError:
                print("Waitress not found, falling back to Flask development server")
                self.app.run(debug=debug, port=self.port, host='0.0.0.0', use_reloader=False)
        
        # Start browser thread
        threading.Timer(1.5, open_browser).start()
        
        # Start server thread
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        return f"Dashboard running at http://127.0.0.1:{self.port}"
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=['https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap'])
server = app.server

# Define styles
COLORS = {
    'background': '#F8F9FA',
    'text': '#343A40',
    'primary': '#007BFF',
    'secondary': '#6C757D',
    'success': '#28A745',
    'danger': '#DC3545',
    'light': '#F8F9FA',
    'dark': '#343A40'
}

# Mahalanobis Filter class
class MahalanobisFilter:
    """Detects outliers using Mahalanobis distance."""

    def __init__(self, alpha=0.05):
        """Initialize with significance level alpha (default 0.05)"""
        self.alpha = alpha

    def fit(self, X):
        """Calculate mean and inverse covariance matrix from data"""
        # Exclude 'Pile No.' from calculations if it exists
        self.data_columns = X.columns.difference(["Pile No."])
        self.mean = np.mean(X[self.data_columns], axis=0)
        self.cov = np.cov(X[self.data_columns], rowvar=False)
        self.inv_cov = np.linalg.inv(self.cov)

    def mahalanobis_distance(self, x):
        """Calculate Mahalanobis distance for a single point"""
        # Ensure x is aligned with self.data_columns
        x_values = x[self.data_columns]
        return mahalanobis(x_values, self.mean, self.inv_cov)

    def filter(self, X, margin=1e-5):
        """
        Filter data into inliers and outliers based on Mahalanobis distance
        relative to an n-sigma threshold.
        Returns (inliers, outliers, distances, threshold) as DataFrames.
        """
        # Calculate distances
        distances = np.array(
            [self.mahalanobis_distance(X.iloc[i]) for i in range(len(X))]
        )
        threshold_distance = (
            np.sqrt(chi2.ppf(0.997, df=len(self.data_columns))) + margin
        )  # 3-sigma threshold with margin

        # Split data
        inlier_indices = distances <= threshold_distance
        outlier_indices = distances > threshold_distance

        return X[inlier_indices], X[outlier_indices], distances, threshold_distance

# Process Data
def process_data(data, sample_pairs):
    """Extract and pair non-null samples from raw data.

    Args:
        data: Input DataFrame containing sample data
        sample_pairs: Tuple of column names to pair

    Returns:
        DataFrame containing only complete pairs of samples
    """
    # Include 'Pile No.' if it exists
    columns_to_select = (
        ["Pile No."] + list(sample_pairs)
        if "Pile No." in data.columns
        else list(sample_pairs)
    )

    df = (
        data.loc[:, columns_to_select]  # Select only the necessary columns
        .dropna()  # Remove rows with any null values
        .copy()  # Return a copy to avoid SettingWithCopyWarning
    )
    return df

# Parse uploaded data
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',', 1)
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename.lower():
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, f'Unsupported file type: {filename}'
        return df, None
    except Exception as e:
        return None, f'Error processing file: {e}'

# Load data
def load_data():
    try:
        return pd.read_csv("data/sample_data_7d.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return a sample DataFrame if file not found
        return pd.DataFrame({
            'Pile No.': range(1, 11),
            'Sample 1': np.random.normal(4000, 200, 10),
            'Sample 2': np.random.normal(4000, 200, 10),
            'Sample 3': np.random.normal(4000, 200, 10),
        })

# Create the app layout
app.layout = html.Div([
    html.Div([
        html.H1("Mahalanobis Distance Filter for Quality Control", 
               style={'textAlign': 'center', 'color': COLORS['dark'], 'marginBottom': 20}),
        html.P("Interactive dashboard for outlier detection using Mahalanobis distance", 
               style={'textAlign': 'center', 'color': COLORS['secondary'], 'marginBottom': 30}),
    ], style={'padding': '20px', 'backgroundColor': COLORS['light']}),
    
    html.Div([
        html.Div([
            html.H4("Upload Data", style={'color': COLORS['dark']}),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a CSV File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0'
                },
                multiple=False
            ),
            html.Div(id='upload-status', style={'marginBottom': '10px', 'color': COLORS['primary']}),
            html.Br(),
            html.H4("Select Sample Pairs", style={'color': COLORS['dark']}),
            dcc.Dropdown(
                id='sample-pair-dropdown',
                options=[
                    {'label': 'Sample 1 vs Sample 2', 'value': 'pair1'},
                    {'label': 'Sample 2 vs Sample 3', 'value': 'pair2'},
                    {'label': 'Sample 1 vs Sample 3', 'value': 'pair3'}
                ],
                value='pair1',
                style={'width': '100%'}
            ),
            html.Br(),
            html.H4("Confidence Level", style={'color': COLORS['dark']}),
            dcc.Slider(
                id='confidence-slider',
                min=1,
                max=5,
                step=0.5,
                value=3,
                marks={i: f'{i}σ' for i in range(1, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),
            html.Button('Update Analysis', 
                       id='update-button', 
                       n_clicks=0, 
                       style={
                           'backgroundColor': COLORS['primary'],
                           'color': 'white',
                           'border': 'none',
                           'padding': '10px 20px',
                           'borderRadius': '5px',
                           'cursor': 'pointer',
                           'marginTop': '10px'
                       })
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        
        html.Div([
            dcc.Graph(id='analysis-graph')
        ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'}),
    ], style={'backgroundColor': COLORS['background'], 'padding': '20px'}),
    
    html.Div([
        html.Div([
            html.H4("Statistics", style={'color': COLORS['dark']}),
            html.Div(id='statistics-output')
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        
        html.Div([
            html.H4("Outliers Detected", style={'color': COLORS['dark']}),
            html.Div(id='outliers-table')
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px'}),
    ], style={'backgroundColor': COLORS['light'], 'padding': '20px'}),
    
    html.Footer([
        html.P("\u00A9 2025 Mahalanobis Filter Dashboard", 
               style={'textAlign': 'center', 'color': COLORS['secondary']}),
    ], style={'padding': '20px', 'marginTop': '20px'})
])

# Store uploaded data
app.clientside_callback(
    """
    function(n_clicks) {
        return window.dash_clientside.no_update;
    }
    """,
    Output('upload-data', 'contents'),
    [Input('update-button', 'n_clicks')],
    prevent_initial_call=True
)

# Define callback to process uploaded data
@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return 'Using default dataset. Upload a CSV file to use your own data.'
    
    df, error = parse_contents(contents, filename)
    if error:
        return html.Div([error], style={'color': COLORS['danger']})
    
    # Store the dataframe in a global variable or dcc.Store
    global uploaded_df
    uploaded_df = df
    
    # Get column names for dropdown options
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return html.Div([
        f'Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns.',
        html.Br(),
        f'Numeric columns available: {", ".join(numeric_columns)}'
    ], style={'color': COLORS['success']})

# Define callback to update the graph
@app.callback(
    [Output('analysis-graph', 'figure'),
     Output('statistics-output', 'children'),
     Output('outliers-table', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('sample-pair-dropdown', 'value'),
     State('confidence-slider', 'value'),
     State('upload-data', 'contents'),
     State('upload-data', 'filename')]
)
def update_graph(n_clicks, pair_value, n_std, contents, filename):
    # Load data - either uploaded or default
    if contents is not None:
        df, error = parse_contents(contents, filename)
        if error:
            df = load_data()  # Fall back to default data if there's an error
    else:
        df = load_data()
    
    # Define sample pairs based on dropdown selection
    pair_mapping = {
        'pair1': ('Sample 1', 'Sample 2'),
        'pair2': ('Sample 2', 'Sample 3'),
        'pair3': ('Sample 1', 'Sample 3')
    }
    sample_pair = pair_mapping[pair_value]
    
    # Check if the selected columns exist in the dataframe
    if not all(col in df.columns for col in sample_pair):
        # Find available numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            sample_pair = (numeric_cols[0], numeric_cols[1])
        else:
            # Not enough numeric columns, show error
            fig = go.Figure()
            fig.add_annotation(
                text="Error: Not enough numeric columns in uploaded data.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="red")
            )
            return fig, "Error: Invalid data format", "Error: Invalid data format"
    
    # Process data for the selected pair
    processed_data = process_data(df, sample_pair)
    
    # Apply Mahalanobis filter
    m_filter = MahalanobisFilter(alpha=0.05)
    m_filter.fit(processed_data)
    inliers, outliers, distances, threshold = m_filter.filter(processed_data, margin=1e-5)
    
    # Create subplot figure
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=('Original Data', 'Mahalanobis Distances', 'Filtered Data with Ellipse'),
                        specs=[[{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'scatter'}]])
    
    # 1. Original scatter plot
    fig.add_trace(
        go.Scatter(
            x=processed_data[sample_pair[0]],
            y=processed_data[sample_pair[1]],
            mode='markers',
            marker=dict(color='gray', size=10, opacity=0.6),
            name='Original Data'
        ),
        row=1, col=1
    )
    
    # 2. Mahalanobis distance histogram
    fig.add_trace(
        go.Histogram(
            x=distances,
            marker=dict(color='lightgray', line=dict(color='gray', width=1)),
            name='Distances'
        ),
        row=1, col=2
    )
    
    # Add threshold line to histogram
    fig.add_trace(
        go.Scatter(
            x=[threshold, threshold],
            y=[0, 10],  # Will be scaled automatically
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name=f'{n_std}σ-Sigma Threshold ({threshold:.2f})'
        ),
        row=1, col=2
    )
    
    # 3. Filtered results with ellipse
    # Add inliers
    fig.add_trace(
        go.Scatter(
            x=inliers[sample_pair[0]],
            y=inliers[sample_pair[1]],
            mode='markers',
            marker=dict(color='#2196F3', size=10, opacity=0.6),
            name='Inliers'
        ),
        row=1, col=3
    )
    
    # Add outliers
    fig.add_trace(
        go.Scatter(
            x=outliers[sample_pair[0]] if not outliers.empty else [],
            y=outliers[sample_pair[1]] if not outliers.empty else [],
            mode='markers+text',
            marker=dict(color='#D92906', size=10, opacity=0.6),
            text=outliers['Pile No.'].iloc[:] if 'Pile No.' in outliers.columns and not outliers.empty else [],
            textposition='top center',
            name='Outliers'
        ),
        row=1, col=3
    )
    
    # Add confidence ellipse
    # Calculate ellipse points
    eigvals, eigvecs = np.linalg.eigh(m_filter.cov)
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Calculate ellipse points
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = []
    ellipse_y = []
    
    for angle in theta:
        x = n_std * np.sqrt(eigvals[0]) * np.cos(angle)
        y = n_std * np.sqrt(eigvals[1]) * np.sin(angle)
        
        # Rotate point
        rotated_x = x * eigvecs[0, 0] + y * eigvecs[0, 1]
        rotated_y = x * eigvecs[1, 0] + y * eigvecs[1, 1]
        
        # Translate point
        ellipse_x.append(rotated_x + m_filter.mean.iloc[0])
        ellipse_y.append(rotated_y + m_filter.mean.iloc[1])
    
    # Add ellipse to plot
    fig.add_trace(
        go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode='lines',
            line=dict(color='green', dash='dash', width=2),
            name=f'{n_std}σ-Sigma Ellipse'
        ),
        row=1, col=3
    )
    
    # Add regression line for inliers
    if not inliers.empty:
        x = inliers[sample_pair[0]]
        y = inliers[sample_pair[1]]
        coeffs = np.polyfit(x, y, 1)
        line_x = np.array([min(x), max(x)])
        line_y = coeffs[0] * line_x + coeffs[1]
        
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                line=dict(color='red', width=2, dash='solid'),
                name='Regression Line'
            ),
            row=1, col=3
        )
    
    # Update layout
    fig.update_layout(
        title=f"Analysis of {sample_pair[0]} vs {sample_pair[1]}",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text=f"{sample_pair[0]}", row=1, col=1)
    fig.update_yaxes(title_text=f"{sample_pair[1]}", row=1, col=1)
    
    fig.update_xaxes(title_text="Mahalanobis Distance", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text=f"{sample_pair[0]}", row=1, col=3)
    fig.update_yaxes(title_text=f"{sample_pair[1]}", row=1, col=3)
    
    # Generate statistics output
    stats_output = html.Div([
        html.Table([
            html.Tr([html.Th("Metric"), html.Th("Value")]),
            html.Tr([html.Td("Total Samples"), html.Td(f"{len(processed_data)}")]),
            html.Tr([html.Td("Inliers"), html.Td(f"{len(inliers)}")]),
            html.Tr([html.Td("Outliers"), html.Td(f"{len(outliers)}")]),
            html.Tr([html.Td("Outlier %"), html.Td(f"{len(outliers)/len(processed_data)*100:.1f}%")]),
            html.Tr([html.Td("Threshold"), html.Td(f"{threshold:.3f}")]),
            html.Tr([html.Td("Confidence"), html.Td(f"{n_std}σ")]),
        ], style={'width': '100%', 'borderCollapse': 'collapse'})
    ], style={'overflowX': 'auto'})
    
    # Generate outliers table
    if outliers.empty:
        outliers_output = html.Div(["No outliers detected"])
    else:
        outliers_output = html.Div([
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in outliers.columns])] +
                # Body
                [html.Tr([html.Td(outliers.iloc[i][col]) for col in outliers.columns])
                 for i in range(len(outliers))],
                style={'width': '100%', 'borderCollapse': 'collapse'}
            )
        ], style={'overflowX': 'auto'})
    
    return fig, stats_output, outliers_output

# Add CSS for styling tables
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Mahalanobis Filter Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
                color: #333;
            }
            tr:nth-child(even) {background-color: #f9f9f9;}
            tr:hover {background-color: #f5f5f5;}
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

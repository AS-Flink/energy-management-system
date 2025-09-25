import pandas as pd
import plotly.graph_objects as go

def create_kpi_dataframe(kpis, kpi_map):
    """Formats a dictionary of KPIs into a structured DataFrame for display."""
    data = []
    for section, keys in kpi_map.items():
        data.append({'Metric': f'--- {section} ---', 'Value': ''})
        for key, unit in keys.items():
            if key in kpis:
                value = kpis[key]
                if unit == "€": formatted_value = f"€ {value:,.0f}"
                elif unit == "%": formatted_value = f"{value:.2%}"
                elif unit == "h": formatted_value = f"{value:.2f} h"
                elif unit == "kWp" or unit == "kWh": formatted_value = f"{value:,.0f} {unit}"
                else: formatted_value = f"{value:,.2f}"
                data.append({'Metric': key.replace('_', ' ').title(), 'Value': formatted_value})
    return pd.DataFrame(data).set_index('Metric')

def generate_summary_chart(df, y_bar, y_line, title):
    """Creates a combination bar and line chart for financial summaries."""
    fig = go.Figure()
    
    # Bar chart trace
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[y_bar],
        name=y_bar.replace('_', ' ').title(),
        marker_color='#1f77b4'
    ))
    
    # Line chart trace
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[y_line],
        name=y_line.replace('_', ' ').title(),
        mode='lines+markers',
        line=dict(color='#2ca02c', width=3)
    ))
    
    # Update layout for a clean look
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        yaxis_tickprefix="€",
        yaxis_tickformat="~s", # Formats large numbers (e.g., 1000000 -> 1M)
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        font=dict(size=11),
        margin=dict(l=20, r=20, t=50, b=20) # Reduce margins
    )
    return fig
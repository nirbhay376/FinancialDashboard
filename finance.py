import yfinance as yf
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# -------------------- Helper Functions --------------------
def safe_get(df, columns):
    """Try multiple column names, return first that exists."""
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if col in df.columns:
            return df[col]
    return pd.Series([np.nan] * len(df), index=df.index)

def calculate_yoy(series):
    """Calculate year-over-year % change."""
    return series.pct_change() * 100

def format_currency(value):
    """Format large numbers as billions/millions."""
    if pd.isna(value):
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"{sign}${abs_val/1e3:.2f}K"
    else:
        return f"{sign}${abs_val:,.0f}"

def calculate_cagr(series):
    """Calculate Compound Annual Growth Rate."""
    if len(series) < 2:
        return np.nan
    first_val = series.iloc[-1]
    last_val = series.iloc[0]
    years = len(series) - 1
    if first_val <= 0 or last_val <= 0:
        return np.nan
    return (((last_val / first_val) ** (1/years)) - 1) * 100

# -------------------- Dash App --------------------
app = dash.Dash(__name__)

app.layout = html.Div(style={
    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", 
    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "minHeight": "100vh",
    "padding": "20px"
}, children=[
    html.Div(style={
        "maxWidth": "1600px", 
        "margin": "0 auto", 
        "backgroundColor": "white", 
        "padding": "40px", 
        "borderRadius": "20px", 
        "boxShadow": "0 20px 60px rgba(0,0,0,0.3)"
    }, children=[
        # Header with gradient
        html.Div([
            html.H1("📊 Advanced Financial Analytics Dashboard", style={
                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "WebkitBackgroundClip": "text",
                "WebkitTextFillColor": "transparent",
                "textAlign": "center",
                "fontSize": "42px",
                "fontWeight": "bold",
                "marginBottom": "10px"
            }),
            html.P("Real-time financial analysis powered by Yahoo Finance", style={
                "textAlign": "center",
                "color": "#7f8c8d",
                "fontSize": "16px",
                "marginBottom": "30px"
            })
        ]),
        
        # Search Section with animation
        html.Div([
            html.Div([
                dcc.Input(
                    id="ticker-input", 
                    type="text", 
                    placeholder="Enter ticker (e.g., AAPL, MSFT, NVDA)",
                    style={
                        "width": "350px", 
                        "padding": "15px 20px", 
                        "marginRight": "15px", 
                        "border": "2px solid #667eea", 
                        "borderRadius": "10px",
                        "fontSize": "16px",
                        "outline": "none",
                        "transition": "all 0.3s"
                    }
                ),
                html.Button("🔍 Analyze", id="search-button", n_clicks=0, 
                    style={
                        "padding": "15px 40px", 
                        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                        "color": "white", 
                        "border": "none", 
                        "borderRadius": "10px", 
                        "cursor": "pointer", 
                        "fontWeight": "bold",
                        "fontSize": "16px",
                        "boxShadow": "0 4px 15px rgba(102, 126, 234, 0.4)",
                        "transition": "all 0.3s"
                    }
                ),
            ], style={"display": "flex", "justifyContent": "center", "alignItems": "center"})
        ], style={"marginBottom": "40px"}),
        
        # Messages
        html.Div(id="loading-message", style={"textAlign": "center", "color": "#667eea", "fontSize": "18px", "marginBottom": "10px", "fontWeight": "500"}),
        html.Div(id="error-message", style={"textAlign": "center", "color": "#e74c3c", "fontSize": "16px", "marginBottom": "20px"}),
        
        # Company Info Header
        html.Div(id="company-header"),
        
        # Tabs for different analyses
        html.Div(id="analysis-tabs")
    ])
])

@app.callback(
    [
        Output("loading-message", "children"),
        Output("error-message", "children"),
        Output("company-header", "children"),
        Output("analysis-tabs", "children"),
    ],
    [Input("search-button", "n_clicks")],
    [State("ticker-input", "value")]
)
def update_dashboard(n_clicks, ticker):
    if not n_clicks or not ticker:
        return "", "", "", ""
    
    ticker = ticker.upper().strip()
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get company info
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        current_price = info.get('currentPrice', 0)
        
        # Fetch financial statements
        income = stock.financials.T
        balance = stock.balance_sheet.T
        cashflow = stock.cashflow.T
        
        if income.empty and balance.empty and cashflow.empty:
            return "", f"❌ No financial data available for {ticker}. Please verify the ticker symbol.", "", ""
        
        # Convert to numeric
        income = income.apply(pd.to_numeric, errors='coerce')
        balance = balance.apply(pd.to_numeric, errors='coerce')
        cashflow = cashflow.apply(pd.to_numeric, errors='coerce')
        
        # Sort by date (most recent first)
        income = income.sort_index(ascending=False)
        balance = balance.sort_index(ascending=False)
        cashflow = cashflow.sort_index(ascending=False)
        
        # -------------------- Extract Key Metrics --------------------
        # Income Statement
        revenue = safe_get(income, ['Total Revenue', 'Revenue'])
        gross_profit = safe_get(income, ['Gross Profit'])
        operating_income = safe_get(income, ['Operating Income'])
        net_income = safe_get(income, ['Net Income'])
        ebitda = safe_get(income, ['EBITDA'])
        
        # Balance Sheet
        total_assets = safe_get(balance, ['Total Assets'])
        current_assets = safe_get(balance, ['Current Assets'])
        total_liabilities = safe_get(balance, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
        current_liabilities = safe_get(balance, ['Current Liabilities'])
        stockholder_equity = safe_get(balance, ['Total Equity Gross Minority Interest', 'Stockholders Equity', 'Total Stockholder Equity'])
        long_term_debt = safe_get(balance, ['Long Term Debt'])
        total_debt = safe_get(balance, ['Total Debt'])
        cash_equiv = safe_get(balance, ['Cash And Cash Equivalents', 'Cash'])
        
        # Cash Flow Statement
        operating_cf = safe_get(cashflow, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
        capex = safe_get(cashflow, ['Capital Expenditure', 'Capital Expenditures'])
        investing_cf = safe_get(cashflow, ['Investing Cash Flow', 'Total Cash From Investing Activities'])
        financing_cf = safe_get(cashflow, ['Financing Cash Flow', 'Total Cash From Financing Activities'])
        
        # Calculate Free Cash Flow (CapEx is usually negative, so we add it)
        fcf = operating_cf + capex
        
        # Calculate Financial Metrics
        revenue_yoy = calculate_yoy(revenue)
        net_income_yoy = calculate_yoy(net_income)
        
        # Profitability Ratios
        gross_margin = (gross_profit / revenue * 100).fillna(0)
        operating_margin = (operating_income / revenue * 100).fillna(0)
        net_margin = (net_income / revenue * 100).fillna(0)
        roa = (net_income / total_assets * 100).fillna(0)
        roe = (net_income / stockholder_equity * 100).fillna(0)
        
        # Liquidity Ratios
        current_ratio = (current_assets / current_liabilities).fillna(0)
        
        # Leverage Ratios
        debt_to_equity = (total_debt / stockholder_equity).fillna(0)
        debt_to_assets = (total_debt / total_assets * 100).fillna(0)
        
        # Efficiency Ratios
        asset_turnover = (revenue / total_assets).fillna(0)
        
        # Latest values
        latest_revenue = revenue.iloc[0] if len(revenue) > 0 else np.nan
        latest_net_income = net_income.iloc[0] if len(net_income) > 0 else np.nan
        latest_fcf = fcf.iloc[0] if len(fcf) > 0 else np.nan
        latest_net_margin = net_margin.iloc[0] if len(net_margin) > 0 else np.nan
        latest_roe = roe.iloc[0] if len(roe) > 0 else np.nan
        latest_debt_equity = debt_to_equity.iloc[0] if len(debt_to_equity) > 0 else np.nan
        
        # Calculate CAGRs
        revenue_cagr = calculate_cagr(revenue)
        net_income_cagr = calculate_cagr(net_income)
        
        # -------------------- Company Header --------------------
        company_header = html.Div([
            html.Div([
                html.Div([
                    html.H2(f"{company_name}", style={"color": "#2c3e50", "marginBottom": "5px", "fontSize": "32px"}),
                    html.H3(f"({ticker})", style={"color": "#7f8c8d", "marginBottom": "10px", "fontWeight": "normal"}),
                    html.Div([
                        html.Span(f"💼 {sector}", style={"marginRight": "20px", "color": "#555"}),
                        html.Span(f"🏭 {industry}", style={"color": "#555"})
                    ]),
                ], style={"flex": "1"}),
                
                html.Div([
                    html.Div([
                        html.H4("Market Cap", style={"color": "#7f8c8d", "fontSize": "14px", "marginBottom": "5px"}),
                        html.H3(format_currency(market_cap), style={"color": "#667eea", "margin": "0"})
                    ], style={"textAlign": "right", "marginRight": "30px"}),
                    html.Div([
                        html.H4("Stock Price", style={"color": "#7f8c8d", "fontSize": "14px", "marginBottom": "5px"}),
                        html.H3(f"${current_price:.2f}" if current_price else "N/A", style={"color": "#2ecc71", "margin": "0"})
                    ], style={"textAlign": "right"}),
                ], style={"display": "flex"})
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
        ], style={
            "padding": "30px", 
            "background": "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
            "borderRadius": "15px",
            "marginBottom": "30px",
            "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"
        })
        
        # -------------------- Key Metrics Cards --------------------
        key_metrics = html.Div([
            html.H3("📊 Key Performance Indicators", style={"color": "#2c3e50", "marginBottom": "20px", "fontSize": "24px"}),
            html.Div([
                create_metric_card("💰 Revenue", format_currency(latest_revenue), f"{revenue_cagr:.1f}% CAGR" if not pd.isna(revenue_cagr) else "N/A", "#667eea"),
                create_metric_card("📈 Net Income", format_currency(latest_net_income), f"{net_income_cagr:.1f}% CAGR" if not pd.isna(net_income_cagr) else "N/A", "#2ecc71"),
                create_metric_card("💵 Free Cash Flow", format_currency(latest_fcf), "Operating CF - CapEx", "#9b59b6"),
                create_metric_card("🎯 Net Margin", f"{latest_net_margin:.2f}%" if not pd.isna(latest_net_margin) else "N/A", "Profitability", "#e67e22"),
                create_metric_card("⚡ ROE", f"{latest_roe:.2f}%" if not pd.isna(latest_roe) else "N/A", "Return on Equity", "#3498db"),
                create_metric_card("📊 Debt/Equity", f"{latest_debt_equity:.2f}x" if not pd.isna(latest_debt_equity) else "N/A", "Leverage Ratio", "#e74c3c"),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))", "gap": "20px"})
        ], style={"marginBottom": "40px"})
        
        # -------------------- Create Interactive Tabs --------------------
        tabs_content = html.Div([
            dcc.Tabs(id="tabs", value='tab-1', children=[
                dcc.Tab(label='📊 Revenue & Profitability', value='tab-1', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='💰 Cash Flow Analysis', value='tab-2', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='📈 Financial Ratios', value='tab-3', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='⚖️ Balance Sheet Health', value='tab-4', style=tab_style, selected_style=tab_selected_style),
            ], style={"marginBottom": "20px"}),
            html.Div(id='tabs-content')
        ])
        
        # Create all charts
        charts_data = {
            'tab-1': create_revenue_charts(ticker, revenue, gross_profit, operating_income, net_income, revenue_yoy, net_income_yoy, gross_margin, operating_margin, net_margin),
            'tab-2': create_cashflow_charts(ticker, operating_cf, investing_cf, financing_cf, fcf, capex),
            'tab-3': create_ratio_charts(ticker, net_margin, roa, roe, current_ratio, asset_turnover),
            'tab-4': create_balance_charts(ticker, total_assets, total_liabilities, stockholder_equity, total_debt, cash_equiv, debt_to_equity, debt_to_assets)
        }
        
        # Create tabs callback content
        tabs_content = html.Div([
            key_metrics,
            dcc.Tabs(id="tabs-main", value='tab-1', children=[
                dcc.Tab(label='📊 Revenue & Profitability', value='tab-1', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='💰 Cash Flow Analysis', value='tab-2', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='📈 Financial Ratios', value='tab-3', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='⚖️ Balance Sheet Health', value='tab-4', style=tab_style, selected_style=tab_selected_style),
            ], style={"marginBottom": "20px"}),
            html.Div([
                charts_data['tab-1'],
                charts_data['tab-2'],
                charts_data['tab-3'],
                charts_data['tab-4']
            ], id='all-charts')
        ])
        
        # Add custom CSS for tab switching
        tabs_content.children.append(html.Script("""
            document.addEventListener('DOMContentLoaded', function() {
                const tabs = document.querySelectorAll('[role="tab"]');
                tabs.forEach((tab, index) => {
                    tab.addEventListener('click', function() {
                        const charts = document.querySelectorAll('#all-charts > div');
                        charts.forEach(chart => chart.style.display = 'none');
                        charts[index].style.display = 'block';
                    });
                });
                // Show first tab by default
                const charts = document.querySelectorAll('#all-charts > div');
                charts.forEach((chart, i) => {
                    chart.style.display = i === 0 ? 'block' : 'none';
                });
            });
        """))
        
        return "", "", company_header, tabs_content
        
    except Exception as e:
        return "", f"❌ Error fetching data: {str(e)}", "", ""

# Tab styles
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '12px',
    'fontWeight': 'bold',
    'fontSize': '14px'
}

tab_selected_style = {
    'borderTop': '3px solid #667eea',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#f0f0ff',
    'color': '#667eea',
    'padding': '12px',
    'fontWeight': 'bold',
    'fontSize': '14px'
}

def create_metric_card(title, value, subtitle, color):
    """Create an enhanced metric card."""
    return html.Div([
        html.Div(title, style={"fontSize": "16px", "color": "#7f8c8d", "marginBottom": "10px", "fontWeight": "600"}),
        html.Div(value, style={"fontSize": "28px", "color": color, "fontWeight": "bold", "marginBottom": "5px"}),
        html.Div(subtitle, style={"fontSize": "12px", "color": "#95a5a6"})
    ], style={
        "padding": "25px",
        "backgroundColor": "#f8f9fa",
        "borderRadius": "12px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
        "transition": "transform 0.3s, box-shadow 0.3s",
        "border": f"2px solid {color}20"
    })

def create_revenue_charts(ticker, revenue, gross_profit, operating_income, net_income, revenue_yoy, net_income_yoy, gross_margin, operating_margin, net_margin):
    """Create revenue and profitability charts."""
    
    # Revenue breakdown
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=revenue.index, y=revenue.values, name='Total Revenue', marker_color='#667eea'))
    fig1.add_trace(go.Bar(x=gross_profit.index, y=gross_profit.values, name='Gross Profit', marker_color='#2ecc71'))
    fig1.add_trace(go.Bar(x=operating_income.index, y=operating_income.values, name='Operating Income', marker_color='#f39c12'))
    fig1.add_trace(go.Bar(x=net_income.index, y=net_income.values, name='Net Income', marker_color='#3498db'))
    fig1.update_layout(
        title=f"{ticker} - Revenue & Income Breakdown",
        height=450,
        xaxis_title="Period",
        yaxis_title="Amount ($)",
        barmode='group',
        template='plotly_white',
        hovermode='x unified'
    )
    
    # YoY Growth
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=revenue_yoy.index, y=revenue_yoy.values, name='Revenue YoY %', marker_color='#667eea'))
    fig2.add_trace(go.Bar(x=net_income_yoy.index, y=net_income_yoy.values, name='Net Income YoY %', marker_color='#2ecc71'))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(
        title=f"{ticker} - Year-over-Year Growth Rate",
        height=450,
        xaxis_title="Period",
        yaxis_title="Growth Rate (%)",
        barmode='group',
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Profit Margins
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=gross_margin.index, y=gross_margin.values, mode='lines+markers', name='Gross Margin', line=dict(color='#2ecc71', width=3)))
    fig3.add_trace(go.Scatter(x=operating_margin.index, y=operating_margin.values, mode='lines+markers', name='Operating Margin', line=dict(color='#f39c12', width=3)))
    fig3.add_trace(go.Scatter(x=net_margin.index, y=net_margin.values, mode='lines+markers', name='Net Margin', line=dict(color='#3498db', width=3)))
    fig3.update_layout(
        title=f"{ticker} - Profit Margins Trend",
        height=450,
        xaxis_title="Period",
        yaxis_title="Margin (%)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3)
    ])

def create_cashflow_charts(ticker, operating_cf, investing_cf, financing_cf, fcf, capex):
    """Create comprehensive cash flow charts."""
    
    # Cash Flow Waterfall
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=operating_cf.index, y=operating_cf.values, name='Operating CF', marker_color='#2ecc71'))
    fig1.add_trace(go.Bar(x=investing_cf.index, y=investing_cf.values, name='Investing CF', marker_color='#e74c3c'))
    fig1.add_trace(go.Bar(x=financing_cf.index, y=financing_cf.values, name='Financing CF', marker_color='#f39c12'))
    fig1.add_hline(y=0, line_dash="dash", line_color="gray")
    fig1.update_layout(
        title=f"{ticker} - Cash Flow Statement Breakdown",
        height=450,
        xaxis_title="Period",
        yaxis_title="Cash Flow ($)",
        barmode='relative',
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Free Cash Flow Analysis
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=operating_cf.index, y=operating_cf.values, mode='lines+markers', name='Operating Cash Flow', 
                              line=dict(color='#2ecc71', width=3), fill='tonexty'))
    fig2.add_trace(go.Scatter(x=fcf.index, y=fcf.values, mode='lines+markers', name='Free Cash Flow',
                              line=dict(color='#9b59b6', width=3), fill='tozeroy'))
    fig2.add_trace(go.Bar(x=capex.index, y=capex.values, name='CapEx', marker_color='#e74c3c', opacity=0.6))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(
        title=f"{ticker} - Free Cash Flow Analysis (Operating CF - CapEx)",
        height=450,
        xaxis_title="Period",
        yaxis_title="Amount ($)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    # FCF Margin
    fcf_copy = fcf.copy()
    operating_cf_copy = operating_cf.copy()
    fcf_margin = (fcf_copy / operating_cf_copy * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=fcf_margin.index, y=fcf_margin.values, name='FCF as % of Operating CF',
                          marker_color='#667eea'))
    fig3.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="100%")
    fig3.update_layout(
        title=f"{ticker} - Free Cash Flow Efficiency",
        height=450,
        xaxis_title="Period",
        yaxis_title="FCF / Operating CF (%)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3)
    ])

def create_ratio_charts(ticker, net_margin, roa, roe, current_ratio, asset_turnover):
    """Create financial ratio charts."""
    
    # Profitability Ratios
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=net_margin.index, y=net_margin.values, mode='lines+markers', name='Net Margin %', 
                              line=dict(color='#3498db', width=3)))
    fig1.add_trace(go.Scatter(x=roa.index, y=roa.values, mode='lines+markers', name='ROA %',
                              line=dict(color='#2ecc71', width=3)))
    fig1.add_trace(go.Scatter(x=roe.index, y=roe.values, mode='lines+markers', name='ROE %',
                              line=dict(color='#9b59b6', width=3)))
    fig1.update_layout(
        title=f"{ticker} - Profitability Ratios",
        height=450,
        xaxis_title="Period",
        yaxis_title="Percentage (%)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Efficiency Ratios
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=current_ratio.index, y=current_ratio.values, mode='lines+markers', name='Current Ratio',
                              line=dict(color='#f39c12', width=3)))
    fig2.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Threshold = 1")
    fig2.add_hline(y=2, line_dash="dash", line_color="green", annotation_text="Healthy = 2")
    fig2.update_layout(
        title=f"{ticker} - Liquidity: Current Ratio",
        height=450,
        xaxis_title="Period",
        yaxis_title="Ratio",
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Asset Turnover
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=asset_turnover.index, y=asset_turnover.values, name='Asset Turnover',
                          marker_color='#667eea'))
    fig3.update_layout(
        title=f"{ticker} - Asset Turnover Ratio",
        height=450,
        xaxis_title="Period",
        yaxis_title="Times",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3)
    ])

def create_balance_charts(ticker, total_assets, total_liabilities, stockholder_equity, total_debt, cash_equiv, debt_to_equity, debt_to_assets):
    """Create balance sheet health charts."""
    
    # Balance Sheet Composition
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=total_assets.index, y=total_assets.values, name='Total Assets', marker_color='#2ecc71'))
    fig1.add_trace(go.Bar(x=total_liabilities.index, y=total_liabilities.values, name='Total Liabilities', marker_color='#e74c3c'))
    fig1.add_trace(go.Bar(x=stockholder_equity.index, y=stockholder_equity.values, name='Stockholder Equity', marker_color='#3498db'))
    fig1.update_layout(
        title=f"{ticker} - Balance Sheet Structure",
        height=450,
        xaxis_title="Period",
        yaxis_title="Amount ($)",
        barmode='group',
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Debt vs Cash
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=total_debt.index, y=total_debt.values, name='Total Debt', marker_color='#e74c3c'))
    fig2.add_trace(go.Bar(x=cash_equiv.index, y=cash_equiv.values, name='Cash & Equivalents', marker_color='#2ecc71'))
    net_debt = total_debt - cash_equiv
    fig2.add_trace(go.Scatter(x=net_debt.index, y=net_debt.values, mode='lines+markers', name='Net Debt',
                              line=dict(color='#9b59b6', width=3)))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(
        title=f"{ticker} - Debt Position Analysis",
        height=450,
        xaxis_title="Period",
        yaxis_title="Amount ($)",
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Leverage Ratios
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=debt_to_equity.index, y=debt_to_equity.values, mode='lines+markers', name='Debt-to-Equity',
                              line=dict(color='#e74c3c', width=3), fill='tozeroy'))
    fig3.add_trace(go.Scatter(x=debt_to_assets.index, y=debt_to_assets.values, mode='lines+markers', name='Debt-to-Assets %',
                              line=dict(color='#f39c12', width=3)))
    fig3.add_hline(y=1, line_dash="dash", line_color="orange", annotation_text="D/E = 1")
    fig3.update_layout(
        title=f"{ticker} - Leverage Ratios Over Time",
        height=450,
        xaxis_title="Period",
        yaxis_title="Ratio / Percentage",
        template='plotly_white',
        hovermode='x unified'
    )
    
    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3)
    ])

if __name__ == "__main__":
    app.run(debug=True)
import yfinance as yf
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
COLORS = {
    'light': {
        'background': '#f8f9fa',
        'card': '#ffffff',
        'text': '#2c3e50',
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#2ecc71',
        'danger': '#e74c3c',
        'warning': '#f39c12',
        'info': '#3498db',
        'border': '#e1e8ed',
        'hover': '#f0f0ff'
    },
    'dark': {
        'background': '#1a1a2e',
        'card': '#16213e',
        'text': '#eaeaea',
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#2ecc71',
        'danger': '#e74c3c',
        'warning': '#f39c12',
        'info': '#3498db',
        'border': '#2d3748',
        'hover': '#2d3748'
    }
}

# ==================== HELPER FUNCTIONS ====================
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
    if pd.isna(value) or value is None:
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

def format_number(value, decimals=2):
    """Format numbers with proper decimal places."""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"

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

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe Ratio."""
    if len(returns) < 2:
        return np.nan
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate Sortino Ratio."""
    if len(returns) < 2:
        return np.nan
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.nan
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown."""
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min() * 100

def create_feature_card(icon, title, description):
    """Create feature card for welcome screen."""
    return html.Div([
        html.Div(icon, style={'fontSize': '48px', 'marginBottom': '15px'}),
        html.H3(title, style={'marginBottom': '10px', 'fontSize': '20px'}),
        html.P(description, style={'opacity': '0.7', 'fontSize': '14px', 'lineHeight': '1.6'})
    ], style={
        'padding': '30px',
        'borderRadius': '15px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.1)',
        'transition': 'transform 0.3s, box-shadow 0.3s',
        'cursor': 'default',
        'background': 'white',
        'textAlign': 'center'
    })


# ==================== DATA FETCH HELPERS (cached) ====================
from functools import lru_cache


@lru_cache(maxsize=128)
def fetch_stock_info(ticker: str) -> dict:
    """Fetch basic stock info via yfinance and cache results by ticker."""
    try:
        t = yf.Ticker(ticker)
        return dict(t.info or {})
    except Exception:
        return {}


@lru_cache(maxsize=128)
def fetch_history(ticker: str, period: str = '1y') -> pd.DataFrame:
    """Fetch historical price data for ticker with caching."""
    try:
        return yf.Ticker(ticker).history(period=period) or pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=128)
def fetch_financials(ticker: str) -> dict:
    """Fetch income, balance sheet and cashflow tables and cache them.
    Returns dict with keys: income, balance, cashflow (each a DataFrame).
    """
    try:
        t = yf.Ticker(ticker)
        income = (t.financials.T if hasattr(t, 'financials') else pd.DataFrame())
        balance = (t.balance_sheet.T if hasattr(t, 'balance_sheet') else pd.DataFrame())
        cashflow = (t.cashflow.T if hasattr(t, 'cashflow') else pd.DataFrame())
        return {
            'income': income if isinstance(income, pd.DataFrame) else pd.DataFrame(),
            'balance': balance if isinstance(balance, pd.DataFrame) else pd.DataFrame(),
            'cashflow': cashflow if isinstance(cashflow, pd.DataFrame) else pd.DataFrame()
        }
    except Exception:
        return {'income': pd.DataFrame(), 'balance': pd.DataFrame(), 'cashflow': pd.DataFrame()}

def create_stat_box(label, value, colors):
    """Create a stat box for company header."""
    return html.Div([
        html.P(label, style={'fontSize': '12px', 'opacity': '0.6', 'marginBottom': '5px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.P(value, style={'fontSize': '20px', 'fontWeight': 'bold', 'margin': '0'})
    ], style={
        'padding': '15px',
        'background': colors['background'],
        'borderRadius': '10px',
        'border': f'1px solid {colors["border"]}'
    })

def create_metric_card(title, value, subtitle, color, colors):
    """Create an enhanced metric card with theme support."""
    return html.Div([
        html.Div(title, style={
            'fontSize': '14px',
            'color': colors['text'],
            'opacity': '0.7',
            'marginBottom': '10px',
            'fontWeight': '600',
            'textTransform': 'uppercase',
            'letterSpacing': '0.5px'
        }),
        html.Div(value, style={
            'fontSize': '32px',
            'color': color,
            'fontWeight': 'bold',
            'marginBottom': '8px',
            'lineHeight': '1'
        }),
        html.Div(subtitle, style={
            'fontSize': '12px',
            'color': colors['text'],
            'opacity': '0.5'
        })
    ], style={
        'padding': '25px',
        'backgroundColor': colors['card'],
        'borderRadius': '15px',
        'boxShadow': '0 4px 15px rgba(0,0,0,0.08)',
        'transition': 'all 0.3s',
        'border': f'2px solid {color}20',
        'position': 'relative',
        'overflow': 'hidden'
    })

# ==================== DASH APP ====================
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Improve mobile rendering and load a web font
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
        {%favicon%}
        {%css%}
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

app.layout = html.Div([
    # Theme Store
    dcc.Store(id='theme-store', data='light'),
    
    # Main Container
    html.Div(id='main-container', children=[
        # Top Navigation Bar
        html.Div(id='nav-bar', children=[
            html.Div([
                html.Div([
                    html.H2("📊 FinanceHub Pro", style={'margin': '0', 'color': 'white', 'fontSize': '24px'}),
                    html.P("Advanced Financial Intelligence Platform", style={'margin': '0', 'fontSize': '12px', 'opacity': '0.8'})
                ]),
                html.Div([
                    html.Button("🌙", id='theme-toggle', n_clicks=0, style={
                        'padding': '10px 15px',
                        'border': 'none',
                        'borderRadius': '8px',
                        'cursor': 'pointer',
                        'fontSize': '20px',
                        'marginRight': '10px',
                        'background': 'rgba(255,255,255,0.2)',
                        'transition': 'all 0.3s'
                    }),
                    dcc.Input(
                        id="ticker-input",
                        type="text",
                        placeholder="Enter ticker (e.g., AAPL)",
                        style={
                            'padding': '12px 20px',
                            'border': 'none',
                            'borderRadius': '8px',
                            'fontSize': '14px',
                            'width': '250px',
                            'marginRight': '10px'
                        }
                    ),
                    html.Button("🔍 Analyze", id="search-button", n_clicks=0, style={
                        'padding': '12px 30px',
                        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '8px',
                        'cursor': 'pointer',
                        'fontWeight': 'bold',
                        'fontSize': '14px',
                        'boxShadow': '0 4px 15px rgba(0,0,0,0.2)',
                        'transition': 'all 0.3s'
                    })
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'padding': '20px 40px',
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'
            })
        ]),
        
        # Messages
        html.Div([
            html.Div(id="loading-message", style={'textAlign': 'center', 'padding': '10px', 'fontSize': '16px'}),
            html.Div(id="error-message", style={'textAlign': 'center', 'padding': '10px', 'fontSize': '16px', 'color': '#e74c3c'}),
        ]),
        
        # Main Content Area
        html.Div(id='content-area', style={'padding': '20px 40px'}, children=[
            # Welcome Screen
            html.Div(id='welcome-screen', children=[
                html.Div([
                    html.H1("Welcome to FinanceHub Pro", style={'fontSize': '48px', 'marginBottom': '20px', 'textAlign': 'center'}),
                    html.P("Your comprehensive financial analysis platform with advanced metrics, real-time data, and AI-powered insights", 
                           style={'fontSize': '18px', 'textAlign': 'center', 'marginBottom': '40px', 'opacity': '0.8'}),
                    html.Div([
                        create_feature_card("📊", "Advanced Analytics", "Deep dive into financial statements with 40+ metrics"),
                        create_feature_card("📈", "Technical Analysis", "Charts, indicators, and price action analysis"),
                        create_feature_card("🎯", "Valuation Models", "DCF, multiples, and comparative analysis"),
                        create_feature_card("🔄", "Portfolio Tools", "Compare stocks and build watch lists"),
                        create_feature_card("📰", "News & Events", "Real-time news and earnings calendar"),
                        create_feature_card("🤖", "AI Insights", "Machine learning predictions and analysis"),
                    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '20px', 'marginTop': '40px'})
                ], style={'maxWidth': '1400px', 'margin': '100px auto', 'textAlign': 'center'})
            ])
        ]),
        
        # Stock Data Content (Hidden initially)
        html.Div(id='stock-content', style={'display': 'none'}, children=[
            # Company Header
            html.Div(id="company-header"),
            
            # Navigation Tabs
            html.Div([
                dcc.Tabs(id="main-tabs", value='overview', children=[
                    dcc.Tab(label='📊 Overview', value='overview'),
                    dcc.Tab(label='💰 Financials', value='financials'),
                    dcc.Tab(label='📈 Technical', value='technical'),
                    dcc.Tab(label='🎯 Valuation', value='valuation'),
                    dcc.Tab(label='📊 Comparison', value='comparison'),
                    dcc.Tab(label='📰 News', value='news'),
                ], style={'marginBottom': '20px'})
            ]),
            
            # Tab Content
            html.Div(id='tab-content')
        ])
    ])
])

def create_feature_card(icon, title, description):
    """Create feature card for welcome screen."""
    return html.Div([
        html.Div(icon, style={'fontSize': '48px', 'marginBottom': '15px'}),
        html.H3(title, style={'marginBottom': '10px'}),
        html.P(description, style={'opacity': '0.7', 'fontSize': '14px'})
    ], style={
        'padding': '30px',
        'borderRadius': '15px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.1)',
        'transition': 'transform 0.3s',
        'cursor': 'default',
        'background': 'white'
    })

# ==================== THEME TOGGLE ====================
@app.callback(
    [Output('theme-store', 'data'),
     Output('theme-toggle', 'children')],
    [Input('theme-toggle', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(n_clicks, current_theme):
    if n_clicks == 0:
        return 'light', '🌙'
    new_theme = 'dark' if current_theme == 'light' else 'light'
    icon = '☀️' if new_theme == 'dark' else '🌙'
    return new_theme, icon

@app.callback(
    Output('main-container', 'style'),
    [Input('theme-store', 'data')]
)
def update_theme(theme):
    colors = COLORS[theme]
    return {
        'fontFamily': "'Inter', 'Segoe UI', sans-serif",
        'backgroundColor': colors['background'],
        'color': colors['text'],
        'minHeight': '100vh',
        'transition': 'all 0.3s'
    }

# ==================== MAIN CALLBACK ====================
@app.callback(
    [
        Output("loading-message", "children"),
        Output("error-message", "children"),
        Output("welcome-screen", "style"),
        Output("stock-content", "style"),
        Output("company-header", "children"),
    ],
    [Input("search-button", "n_clicks")],
    [State("ticker-input", "value"),
     State("theme-store", "data")]
)
def update_dashboard(n_clicks, ticker, theme):
    colors = COLORS[theme]
    
    if not n_clicks or not ticker:
        return "", "", {'display': 'block'}, {'display': 'none'}, ""
    
    ticker = ticker.upper().strip()
    
    try:
        # Loading message
        loading_msg = html.Div([
            html.Div(className="spinner"),
            html.P(f"⏳ Fetching comprehensive data for {ticker}...", style={'marginTop': '10px'})
        ], style={'textAlign': 'center', 'color': colors['primary']})
        
        # Use cached fetch helpers to reduce repeated yfinance calls
        info = fetch_stock_info(ticker)
        financials = fetch_financials(ticker)
        income = financials.get('income', pd.DataFrame())
        balance = financials.get('balance', pd.DataFrame())
        cashflow = financials.get('cashflow', pd.DataFrame())

        # Get company info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        pe_ratio = info.get('trailingPE', 0)
        dividend_yield = info.get('dividendYield', 0)
        beta = info.get('beta', 0)
        week_52_high = info.get('fiftyTwoWeekHigh', 0)
        week_52_low = info.get('fiftyTwoWeekLow', 0)
        avg_volume = info.get('averageVolume', 0)
        
        # (financials already populated via fetch_financials above)
        
        if income.empty and balance.empty and cashflow.empty:
            return "", f"❌ No financial data available for {ticker}", {'display': 'block'}, {'display': 'none'}, ""
        
        # Convert to numeric
        income = income.apply(pd.to_numeric, errors='coerce')
        balance = balance.apply(pd.to_numeric, errors='coerce')
        cashflow = cashflow.apply(pd.to_numeric, errors='coerce')
        
        # Sort by date
        income = income.sort_index(ascending=False)
        balance = balance.sort_index(ascending=False)
        cashflow = cashflow.sort_index(ascending=False)
        
        # Extract key metrics
        revenue = safe_get(income, ['Total Revenue', 'Revenue'])
        net_income = safe_get(income, ['Net Income'])
        
        # Latest values
        latest_revenue = revenue.iloc[0] if len(revenue) > 0 else np.nan
        latest_net_income = net_income.iloc[0] if len(net_income) > 0 else np.nan
        
        # Calculate price change
        hist = fetch_history(ticker, period='1d')
        if not hist.empty:
            prev_close = hist['Close'].iloc[-1] if len(hist) > 0 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close * 100) if prev_close != 0 else 0
        else:
            price_change = 0
            price_change_pct = 0
        
        # Company Header
        company_header = html.Div([
            html.Div([
                # Left Section
                html.Div([
                    html.H1(company_name, style={'fontSize': '36px', 'marginBottom': '5px', 'color': colors['text']}),
                    html.H2(f"({ticker})", style={'fontSize': '20px', 'color': colors['text'], 'opacity': '0.6', 'fontWeight': 'normal', 'marginBottom': '15px'}),
                    html.Div([
                        html.Span([
                            html.Span("💼 ", style={'marginRight': '5px'}),
                            html.Span(sector, style={'fontWeight': '500'})
                        ], style={'marginRight': '20px', 'padding': '8px 15px', 'background': colors['card'], 'borderRadius': '20px', 'border': f'1px solid {colors["border"]}'}),
                        html.Span([
                            html.Span("🏭 ", style={'marginRight': '5px'}),
                            html.Span(industry, style={'fontWeight': '500'})
                        ], style={'padding': '8px 15px', 'background': colors['card'], 'borderRadius': '20px', 'border': f'1px solid {colors["border"]}'})
                    ])
                ], style={'flex': '1'}),
                
                # Right Section - Price
                html.Div([
                    html.Div([
                        html.H1(f"${current_price:.2f}" if current_price else "N/A", style={
                            'fontSize': '48px',
                            'margin': '0',
                            'color': colors['success'] if price_change >= 0 else colors['danger']
                        }),
                        html.Div([
                            html.Span(f"{'▲' if price_change >= 0 else '▼'} ${abs(price_change):.2f}", style={
                                'fontSize': '18px',
                                'marginRight': '10px',
                                'color': colors['success'] if price_change >= 0 else colors['danger']
                            }),
                            html.Span(f"({price_change_pct:+.2f}%)", style={
                                'fontSize': '18px',
                                'color': colors['success'] if price_change >= 0 else colors['danger']
                            })
                        ], style={'marginTop': '5px'})
                    ], style={'textAlign': 'right'})
                ], style={'marginLeft': 'auto'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '30px'}),
            
            # Key Metrics Grid
            html.Div([
                create_stat_box("Market Cap", format_currency(market_cap), colors),
                create_stat_box("P/E Ratio", format_number(pe_ratio) if pe_ratio else "N/A", colors),
                create_stat_box("Dividend Yield", f"{dividend_yield*100:.2f}%" if dividend_yield else "N/A", colors),
                create_stat_box("Beta", format_number(beta) if beta else "N/A", colors),
                create_stat_box("52W High", f"${week_52_high:.2f}" if week_52_high else "N/A", colors),
                create_stat_box("52W Low", f"${week_52_low:.2f}" if week_52_low else "N/A", colors),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(180px, 1fr))', 'gap': '15px'})
        ], style={
            'padding': '40px',
            'background': colors['card'],
            'borderRadius': '20px',
            'marginBottom': '30px',
            'boxShadow': '0 4px 20px rgba(0,0,0,0.1)',
            'border': f'1px solid {colors["border"]}'
        })
        
        return "", "", {'display': 'none'}, {'display': 'block', 'padding': '20px 40px'}, company_header
        
    except Exception as e:
        return "", f"❌ Error: {str(e)}", {'display': 'block'}, {'display': 'none'}, ""

def create_stat_box(label, value, colors):
    """Create a stat box for company header."""
    return html.Div([
        html.P(label, style={'fontSize': '12px', 'opacity': '0.6', 'marginBottom': '5px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.P(value, style={'fontSize': '20px', 'fontWeight': 'bold', 'margin': '0'})
    ], style={
        'padding': '15px',
        'background': colors['background'],
        'borderRadius': '10px',
        'border': f'1px solid {colors["border"]}'
    })

# ==================== TAB CONTENT CALLBACK ====================
@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('ticker-input', 'value')],
    [State('theme-store', 'data')]
)
def render_tab_content(active_tab, ticker, theme):
    if not ticker:
        return html.Div("Please enter a ticker symbol", style={'textAlign': 'center', 'padding': '50px'})
    
    colors = COLORS[theme]
    ticker = ticker.upper().strip()
    
    try:
        stock = yf.Ticker(ticker)
        
        if active_tab == 'overview':
            return create_overview_tab(stock, ticker, colors)
        elif active_tab == 'financials':
            return create_financials_tab(stock, ticker, colors)
        elif active_tab == 'technical':
            return create_technical_tab(stock, ticker, colors)
        elif active_tab == 'valuation':
            return create_valuation_tab(stock, ticker, colors)
        elif active_tab == 'comparison':
            return create_comparison_tab(stock, ticker, colors)
        elif active_tab == 'news':
            return create_news_tab(stock, ticker, colors)
            
    except Exception as e:
        return html.Div(f"Error loading tab: {str(e)}", style={'color': colors['danger'], 'padding': '20px'})

# ==================== TAB CREATORS ====================
def create_overview_tab(stock, ticker, colors):
    """Create overview tab with key metrics and charts."""
    try:
        # Get historical data
        hist = stock.history(period='1y')
        info = stock.info
        
        # Price chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=hist.index, y=hist['Close'],
            mode='lines',
            name='Price',
            line=dict(color=colors['primary'], width=2),
            fill='tozeroy',
            fillcolor=f'rgba(102, 126, 234, 0.1)'
        ))
        fig_price.update_layout(
            title=f"{ticker} - 1 Year Price Chart",
            height=400,
            template='plotly_white' if colors == COLORS['light'] else 'plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font=dict(color=colors['text']),
            hovermode='x unified'
        )
        
        # Volume chart
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=hist.index, y=hist['Volume'],
            name='Volume',
            marker_color=colors['info']
        ))
        fig_volume.update_layout(
            title=f"{ticker} - Trading Volume",
            height=300,
            template='plotly_white' if colors == COLORS['light'] else 'plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font=dict(color=colors['text']),
            hovermode='x unified'
        )
        
        # Calculate returns
        returns = hist['Close'].pct_change().dropna()
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(hist['Close'])
        
        return html.Div([
            # Performance Metrics
            html.Div([
                html.H3("📊 Performance Metrics", style={'marginBottom': '20px', 'color': colors['text']}),
                html.Div([
                    create_metric_card("Sharpe Ratio", format_number(sharpe), "Risk-adjusted return", colors['info'], colors),
                    create_metric_card("Max Drawdown", f"{max_dd:.2f}%" if not pd.isna(max_dd) else "N/A", "Peak to trough decline", colors['danger'], colors),
                    create_metric_card("YTD Return", f"{((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100):.2f}%", "Year to date performance", colors['success'], colors),
                    create_metric_card("Volatility", f"{returns.std() * np.sqrt(252) * 100:.2f}%", "Annualized std dev", colors['warning'], colors),
                ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '20px', 'marginBottom': '30px'})
            ]),
            
            # Charts
            html.Div([
                dcc.Graph(figure=fig_price, config={'displayModeBar': False}),
                dcc.Graph(figure=fig_volume, config={'displayModeBar': False})
            ])
        ])
        
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={'color': colors['danger']})

def create_financials_tab(stock, ticker, colors):
    """Create detailed financials tab."""
    income = stock.financials.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
    balance = stock.balance_sheet.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
    cashflow = stock.cashflow.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
    
    # Extract metrics
    revenue = safe_get(income, ['Total Revenue', 'Revenue'])
    gross_profit = safe_get(income, ['Gross Profit'])
    operating_income = safe_get(income, ['Operating Income'])
    net_income = safe_get(income, ['Net Income'])
    
    # Charts
    fig = go.Figure()
    fig.add_trace(go.Bar(x=revenue.index, y=revenue.values, name='Revenue', marker_color=colors['primary']))
    fig.add_trace(go.Bar(x=gross_profit.index, y=gross_profit.values, name='Gross Profit', marker_color=colors['success']))
    fig.add_trace(go.Bar(x=net_income.index, y=net_income.values, name='Net Income', marker_color=colors['info']))
    
    fig.update_layout(
        title=f"{ticker} - Income Statement",
        height=500,
        template='plotly_white' if colors == COLORS['light'] else 'plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        barmode='group',
        hovermode='x unified'
    )
    
    return html.Div([
        html.H3("💰 Financial Statements", style={'marginBottom': '20px', 'color': colors['text']}),
        dcc.Graph(figure=fig, config={'displayModeBar': False})
    ])

def create_technical_tab(stock, ticker, colors):
    """Create technical analysis tab."""
    hist = stock.history(period='6mo')
    
    # Calculate technical indicators
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
    hist['EMA_12'] = hist['Close'].ewm(span=12, adjust=False).mean()
    hist['EMA_26'] = hist['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{ticker} Price & Moving Averages', 'MACD', 'RSI')
    )
    
    # Price and MAs
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], 
                                  low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        height=900,
        template='plotly_white' if colors == COLORS['light'] else 'plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return html.Div([
        html.H3("📈 Technical Analysis", style={'marginBottom': '20px', 'color': colors['text']}),
        html.Div([
            create_metric_card("Current RSI", f"{hist['RSI'].iloc[-1]:.2f}" if not hist['RSI'].empty else "N/A", 
                             "Overbought > 70, Oversold < 30", colors['info'], colors),
            create_metric_card("MACD Signal", "Bullish" if hist['MACD'].iloc[-1] > hist['Signal'].iloc[-1] else "Bearish",
                             "Momentum indicator", colors['success'] if hist['MACD'].iloc[-1] > hist['Signal'].iloc[-1] else colors['danger'], colors),
            create_metric_card("20-Day SMA", f"${hist['SMA_20'].iloc[-1]:.2f}" if not hist['SMA_20'].empty else "N/A",
                             "Short-term trend", colors['warning'], colors),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '20px', 'marginBottom': '30px'}),
        dcc.Graph(figure=fig, config={'displayModeBar': True})
    ])

def create_valuation_tab(stock, ticker, colors):
    """Create valuation analysis tab."""
    info = stock.info
    income = stock.financials.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
    
    # Get valuation metrics
    pe_ratio = info.get('trailingPE', np.nan)
    forward_pe = info.get('forwardPE', np.nan)
    peg_ratio = info.get('pegRatio', np.nan)
    price_to_book = info.get('priceToBook', np.nan)
    price_to_sales = info.get('priceToSalesTrailing12Months', np.nan)
    ev_to_revenue = info.get('enterpriseToRevenue', np.nan)
    ev_to_ebitda = info.get('enterpriseToEbitda', np.nan)
    
    # Get earnings data
    net_income = safe_get(income, ['Net Income'])
    revenue = safe_get(income, ['Total Revenue', 'Revenue'])
    
    # Calculate growth rates
    revenue_growth = calculate_cagr(revenue)
    earnings_growth = calculate_cagr(net_income)
    
    # Create valuation multiples chart
    multiples_data = {
        'Metric': ['P/E Ratio', 'Forward P/E', 'PEG Ratio', 'P/B Ratio', 'P/S Ratio', 'EV/Revenue', 'EV/EBITDA'],
        'Value': [pe_ratio, forward_pe, peg_ratio, price_to_book, price_to_sales, ev_to_revenue, ev_to_ebitda]
    }
    
    fig_multiples = go.Figure()
    fig_multiples.add_trace(go.Bar(
        x=multiples_data['Metric'],
        y=multiples_data['Value'],
        marker_color=colors['primary'],
        text=[f"{v:.2f}" if not pd.isna(v) else "N/A" for v in multiples_data['Value']],
        textposition='auto',
    ))
    
    fig_multiples.update_layout(
        title=f"{ticker} - Valuation Multiples",
        height=400,
        template='plotly_white' if colors == COLORS['light'] else 'plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        yaxis_title="Multiple Value",
        showlegend=False
    )
    
    # Growth rates chart
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Bar(
        x=['Revenue CAGR', 'Earnings CAGR'],
        y=[revenue_growth, earnings_growth],
        marker_color=[colors['success'], colors['info']],
        text=[f"{revenue_growth:.2f}%" if not pd.isna(revenue_growth) else "N/A",
              f"{earnings_growth:.2f}%" if not pd.isna(earnings_growth) else "N/A"],
        textposition='auto',
    ))
    
    fig_growth.update_layout(
        title=f"{ticker} - Historical Growth Rates",
        height=400,
        template='plotly_white' if colors == COLORS['light'] else 'plotly_dark',
        paper_bgcolor=colors['card'],
        plot_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        yaxis_title="CAGR (%)",
        showlegend=False
    )
    
    return html.Div([
        html.H3("🎯 Valuation Analysis", style={'marginBottom': '20px', 'color': colors['text']}),
        
        # Key valuation metrics
        html.Div([
            create_metric_card("P/E Ratio", f"{pe_ratio:.2f}" if not pd.isna(pe_ratio) else "N/A",
                             "Price-to-Earnings", colors['primary'], colors),
            create_metric_card("PEG Ratio", f"{peg_ratio:.2f}" if not pd.isna(peg_ratio) else "N/A",
                             "P/E to Growth", colors['info'], colors),
            create_metric_card("P/B Ratio", f"{price_to_book:.2f}" if not pd.isna(price_to_book) else "N/A",
                             "Price-to-Book", colors['success'], colors),
            create_metric_card("EV/EBITDA", f"{ev_to_ebitda:.2f}" if not pd.isna(ev_to_ebitda) else "N/A",
                             "Enterprise Value Multiple", colors['warning'], colors),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '20px', 'marginBottom': '30px'}),
        
        # Charts
        html.Div([
            html.Div([dcc.Graph(figure=fig_multiples, config={'displayModeBar': False})], style={'flex': '1'}),
            html.Div([dcc.Graph(figure=fig_growth, config={'displayModeBar': False})], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'}),
        
        # Valuation interpretation
        html.Div([
            html.H4("💡 Valuation Insights", style={'color': colors['text'], 'marginBottom': '15px'}),
            html.Ul([
                html.Li(f"P/E Ratio of {pe_ratio:.2f} suggests the stock is trading at {'a premium' if pe_ratio > 25 else 'a reasonable' if pe_ratio > 15 else 'a discount'} compared to market averages." if not pd.isna(pe_ratio) else "P/E ratio not available."),
                html.Li(f"PEG Ratio of {peg_ratio:.2f} indicates {'potentially overvalued' if peg_ratio > 2 else 'fairly valued' if peg_ratio > 1 else 'potentially undervalued'} growth prospects." if not pd.isna(peg_ratio) else "PEG ratio not available."),
                html.Li(f"Revenue growing at {revenue_growth:.2f}% CAGR over the historical period." if not pd.isna(revenue_growth) else "Revenue growth data not available."),
            ], style={'color': colors['text'], 'opacity': '0.8', 'lineHeight': '1.8'})
        ], style={
            'padding': '25px',
            'background': colors['card'],
            'borderRadius': '15px',
            'border': f'1px solid {colors["border"]}',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'
        })
    ])

def create_comparison_tab(stock, ticker, colors):
    """Create comparison tab for analyzing multiple stocks."""
    return html.Div([
        html.H3("📊 Stock Comparison Tool", style={'marginBottom': '20px', 'color': colors['text']}),
        
        html.Div([
            html.P("Compare multiple stocks side-by-side:", style={'marginBottom': '15px', 'color': colors['text']}),
            html.Div([
                dcc.Input(
                    id='compare-ticker-1',
                    type='text',
                    placeholder='Ticker 1',
                    style={
                        'padding': '12px',
                        'marginRight': '10px',
                        'border': f'2px solid {colors["border"]}',
                        'borderRadius': '8px',
                        'background': colors['card'],
                        'color': colors['text']
                    }
                ),
                dcc.Input(
                    id='compare-ticker-2',
                    type='text',
                    placeholder='Ticker 2',
                    style={
                        'padding': '12px',
                        'marginRight': '10px',
                        'border': f'2px solid {colors["border"]}',
                        'borderRadius': '8px',
                        'background': colors['card'],
                        'color': colors['text']
                    }
                ),
                dcc.Input(
                    id='compare-ticker-3',
                    type='text',
                    placeholder='Ticker 3',
                    style={
                        'padding': '12px',
                        'marginRight': '10px',
                        'border': f'2px solid {colors["border"]}',
                        'borderRadius': '8px',
                        'background': colors['card'],
                        'color': colors['text']
                    }
                ),
                html.Button("Compare", id='compare-button', n_clicks=0, style={
                    'padding': '12px 30px',
                    'background': colors['primary'],
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '8px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold'
                })
            ], style={'marginBottom': '30px'}),
            
            html.Div(id='comparison-output')
        ], style={
            'padding': '30px',
            'background': colors['card'],
            'borderRadius': '15px',
            'border': f'1px solid {colors["border"]}',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'
        })
    ])

def create_news_tab(stock, ticker, colors):
    """Create news and events tab."""
    try:
        news = stock.news[:10] if hasattr(stock, 'news') else []
        
        news_items = []
        for item in news:
            news_items.append(
                html.Div([
                    html.A([
                        html.H4(item.get('title', 'No Title'), style={
                            'color': colors['primary'],
                            'marginBottom': '8px',
                            'fontSize': '18px'
                        }),
                    ], href=item.get('link', '#'), target='_blank', style={'textDecoration': 'none'}),
                    html.P(item.get('publisher', 'Unknown Source'), style={
                        'fontSize': '12px',
                        'color': colors['text'],
                        'opacity': '0.6',
                        'marginBottom': '10px'
                    }),
                    html.P(item.get('summary', '')[:200] + '...' if item.get('summary') else '', style={
                        'color': colors['text'],
                        'opacity': '0.8',
                        'lineHeight': '1.6'
                    })
                ], style={
                    'padding': '20px',
                    'background': colors['card'],
                    'borderRadius': '12px',
                    'marginBottom': '15px',
                    'border': f'1px solid {colors["border"]}',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.05)',
                    'transition': 'transform 0.2s',
                    'cursor': 'pointer'
                })
            )
        
        return html.Div([
            html.H3(f"📰 Latest News for {ticker}", style={'marginBottom': '25px', 'color': colors['text']}),
            html.Div(news_items if news_items else html.P("No news available", style={'color': colors['text'], 'textAlign': 'center', 'padding': '50px'}))
        ])
    except:
        return html.Div([
            html.H3(f"📰 Latest News for {ticker}", style={'marginBottom': '25px', 'color': colors['text']}),
            html.P("News data temporarily unavailable", style={'color': colors['text'], 'textAlign': 'center', 'padding': '50px'})
        ])

# ==================== COMPARISON CALLBACK ====================
@app.callback(
    Output('comparison-output', 'children'),
    [Input('compare-button', 'n_clicks')],
    [State('compare-ticker-1', 'value'),
     State('compare-ticker-2', 'value'),
     State('compare-ticker-3', 'value'),
     State('theme-store', 'data')]
)
def compare_stocks(n_clicks, ticker1, ticker2, ticker3, theme):
    if not n_clicks or not ticker1:
        return html.Div()
    
    colors = COLORS[theme]
    tickers = [t.upper().strip() for t in [ticker1, ticker2, ticker3] if t]
    
    try:
        comparison_data = []
        
        for ticker in tickers:
            info = fetch_stock_info(ticker)
            hist = fetch_history(ticker, period='1y')
            
            comparison_data.append({
                'Ticker': ticker,
                'Price': f"${info.get('currentPrice', 0):.2f}",
                'Market Cap': format_currency(info.get('marketCap', 0)),
                'P/E Ratio': f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A",
                'Dividend Yield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A",
                '1Y Return': f"{((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100):.2f}%" if len(hist) > 0 else "N/A"
            })
        
        # Create comparison table
        df = pd.DataFrame(comparison_data)
        
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_cell={
                'textAlign': 'left',
                'padding': '15px',
                'backgroundColor': colors['card'],
                'color': colors['text'],
                'border': f'1px solid {colors["border"]}'
            },
            style_header={
                'backgroundColor': colors['primary'],
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'left',
                'padding': '15px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': colors['background']
                }
            ]
        )
        
        # Create comparison chart
        fig = go.Figure()
        for ticker in tickers:
            hist = fetch_history(ticker, period='1y')
            normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=normalized,
                mode='lines',
                name=ticker,
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Normalized Price Comparison (Base = 100)",
            height=400,
            template='plotly_white' if colors == COLORS['light'] else 'plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font=dict(color=colors['text']),
            hovermode='x unified',
            yaxis_title="Normalized Price",
            xaxis_title="Date"
        )
        
        return html.Div([
            html.H4("Comparison Table", style={'color': colors['text'], 'marginBottom': '20px'}),
            table,
            html.Div(style={'height': '30px'}),
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ])
        
    except Exception as e:
        return html.Div(f"Error comparing stocks: {str(e)}", style={'color': colors['danger'], 'padding': '20px'})

# ==================== SERVER EXPORT ====================
# Export server for Gunicorn
server = app.server

# ==================== RUN APP ====================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
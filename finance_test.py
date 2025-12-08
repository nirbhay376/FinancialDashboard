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
    """Fetch income, balance sheet and cashflow tables and cache them."""
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

def create_financial_table(df, colors, title):
    """Create a formatted financial statement table."""
    if df.empty:
        return html.Div("No data available", style={'padding': '20px', 'textAlign': 'center', 'color': colors['text']})
    
    # Format the dataframe for display
    df_display = df.copy()
    
    # Convert index to string dates
    df_display.index = df_display.index.strftime('%Y-%m-%d')
    
    # Reset index to make dates a column
    df_display = df_display.reset_index()
    df_display.columns = ['Date'] + list(df_display.columns[1:])
    
    # Format numbers in billions
    for col in df_display.columns[1:]:
        df_display[col] = df_display[col].apply(lambda x: format_currency(x) if pd.notna(x) else 'N/A')
    
    # Create table data
    table_data = df_display.to_dict('records')
    
    return html.Div([
        html.H3(title, style={'marginBottom': '20px', 'color': colors['text']}),
        dash_table.DataTable(
            data=table_data,
            columns=[{'name': col, 'id': col} for col in df_display.columns],
            style_table={
                'overflowX': 'auto',
                'border': f'1px solid {colors["border"]}'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '12px',
                'backgroundColor': colors['card'],
                'color': colors['text'],
                'border': f'1px solid {colors["border"]}',
                'fontSize': '14px',
                'fontFamily': 'Inter, sans-serif'
            },
            style_header={
                'backgroundColor': colors['primary'],
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'padding': '15px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': colors['background']
                }
            ],
            page_size=20
        )
    ], style={'marginBottom': '40px'})

# ==================== DASH APP ====================
app = dash.Dash(__name__, suppress_callback_exceptions=True)

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
    dcc.Store(id='theme-store', data='light'),
    
    html.Div(id='main-container', children=[
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
        
        html.Div([
            html.Div(id="loading-message", style={'textAlign': 'center', 'padding': '10px', 'fontSize': '16px'}),
            html.Div(id="error-message", style={'textAlign': 'center', 'padding': '10px', 'fontSize': '16px', 'color': '#e74c3c'}),
        ]),
        
        html.Div(id='content-area', style={'padding': '20px 40px'}, children=[
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
        
        html.Div(id='stock-content', style={'display': 'none'}, children=[
            html.Div(id="company-header"),
            
            html.Div([
                dcc.Tabs(id="main-tabs", value='overview', children=[
                    dcc.Tab(label='📊 Overview', value='overview'),
                    dcc.Tab(label='📋 Full Statements', value='statements'),
                    dcc.Tab(label='💰 Financial Analysis', value='financials'),
                    dcc.Tab(label='📈 Technical', value='technical'),
                    dcc.Tab(label='🎯 Valuation', value='valuation'),
                ], style={'marginBottom': '20px'})
            ]),
            
            html.Div(id='tab-content')
        ])
    ])
])

# ==================== CALLBACKS ====================
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
        info = fetch_stock_info(ticker)
        financials = fetch_financials(ticker)
        income = financials.get('income', pd.DataFrame())
        balance = financials.get('balance', pd.DataFrame())
        cashflow = financials.get('cashflow', pd.DataFrame())

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
        
        if income.empty and balance.empty and cashflow.empty:
            return "", f"❌ No financial data available for {ticker}", {'display': 'block'}, {'display': 'none'}, ""
        
        hist = fetch_history(ticker, period='1d')
        if not hist.empty:
            prev_close = hist['Close'].iloc[-1] if len(hist) > 0 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close * 100) if prev_close != 0 else 0
        else:
            price_change = 0
            price_change_pct = 0
        
        company_header = html.Div([
            html.Div([
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
        elif active_tab == 'statements':
            return create_statements_tab(stock, ticker, colors)
        elif active_tab == 'financials':
            return create_financials_tab(stock, ticker, colors)
        elif active_tab == 'technical':
            return create_technical_tab(stock, ticker, colors)
        elif active_tab == 'valuation':
            return create_valuation_tab(stock, ticker, colors)
            
    except Exception as e:
        return html.Div(f"Error loading tab: {str(e)}", style={'color': colors['danger'], 'padding': '20px'})

# ==================== TAB CREATORS ====================
def create_overview_tab(stock, ticker, colors):
    """Create overview tab with key metrics and charts."""
    try:
        hist = stock.history(period='1y')
        
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
        
        returns = hist['Close'].pct_change().dropna()
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(hist['Close'])
        
        return html.Div([
            html.Div([
                html.H3("📊 Performance Metrics", style={'marginBottom': '20px', 'color': colors['text']}),
                html.Div([
                    create_metric_card("Sharpe Ratio", format_number(sharpe), "Risk-adjusted return", colors['info'], colors),
                    create_metric_card("Max Drawdown", f"{max_dd:.2f}%" if not pd.isna(max_dd) else "N/A", "Peak to trough decline", colors['danger'], colors),
                    create_metric_card("YTD Return", f"{((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100):.2f}%", "Year to date performance", colors['success'], colors),
                    create_metric_card("Volatility", f"{returns.std() * np.sqrt(252) * 100:.2f}%", "Annualized std dev", colors['warning'], colors),
                ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '20px', 'marginBottom': '30px'})
            ]),
            
            html.Div([
                dcc.Graph(figure=fig_price, config={'displayModeBar': False}),
                dcc.Graph(figure=fig_volume, config={'displayModeBar': False})
            ])
        ])
        
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={'color': colors['danger']})

def create_statements_tab(stock, ticker, colors):
    """Create full financial statements tab."""
    try:
        income = stock.financials.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
        balance = stock.balance_sheet.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
        cashflow = stock.cashflow.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
        
        return html.Div([
            html.Div([
                html.H2("📋 Complete Financial Statements", style={
                    'fontSize': '32px',
                    'marginBottom': '10px',
                    'color': colors['text']
                }),
                html.P("View complete income statement, balance sheet, and cash flow statement", style={
                    'fontSize': '16px',
                    'opacity': '0.7',
                    'marginBottom': '40px'
                })
            ]),
            
            create_financial_table(income, colors, "📊 Income Statement"),
            create_financial_table(balance, colors, "💼 Balance Sheet"),
            create_financial_table(cashflow, colors, "💵 Cash Flow Statement"),
        ])
        
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={'color': colors['danger']})

def create_financials_tab(stock, ticker, colors):
    """Create financial analysis tab."""
    try:
        income = stock.financials.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
        balance = stock.balance_sheet.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
        cashflow = stock.cashflow.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
        
        revenue = safe_get(income, ['Total Revenue', 'Revenue'])
        gross_profit = safe_get(income, ['Gross Profit'])
        net_income = safe_get(income, ['Net Income'])
        total_assets = safe_get(balance, ['Total Assets'])
        total_equity = safe_get(balance, ['Total Equity Gross Minority Interest', 'Total Stockholders Equity'])
        operating_cf = safe_get(cashflow, ['Operating Cash Flow'])
        
        # Calculate margins
        gross_margin = (gross_profit / revenue * 100) if not revenue.empty else pd.Series()
        net_margin = (net_income / revenue * 100) if not revenue.empty else pd.Series()
        
        # Revenue chart
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Bar(
            x=revenue.index.strftime('%Y'),
            y=revenue.values,
            name='Revenue',
            marker_color=colors['primary']
        ))
        fig_revenue.update_layout(
            title="Revenue Trend",
            height=350,
            template='plotly_white' if colors == COLORS['light'] else 'plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font=dict(color=colors['text']),
            yaxis_title="Revenue ($)"
        )
        
        # Profitability chart
        fig_profit = go.Figure()
        fig_profit.add_trace(go.Scatter(
            x=gross_margin.index.strftime('%Y'),
            y=gross_margin.values,
            name='Gross Margin',
            line=dict(color=colors['success'], width=3)
        ))
        fig_profit.add_trace(go.Scatter(
            x=net_margin.index.strftime('%Y'),
            y=net_margin.values,
            name='Net Margin',
            line=dict(color=colors['info'], width=3)
        ))
        fig_profit.update_layout(
            title="Profitability Margins",
            height=350,
            template='plotly_white' if colors == COLORS['light'] else 'plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font=dict(color=colors['text']),
            yaxis_title="Margin (%)"
        )
        
        # Key metrics
        latest_revenue = revenue.iloc[0] if len(revenue) > 0 else 0
        latest_net_income = net_income.iloc[0] if len(net_income) > 0 else 0
        latest_gross_margin = gross_margin.iloc[0] if len(gross_margin) > 0 else 0
        latest_net_margin = net_margin.iloc[0] if len(net_margin) > 0 else 0
        
        revenue_cagr = calculate_cagr(revenue)
        
        return html.Div([
            html.Div([
                html.H3("💰 Key Financial Metrics", style={'marginBottom': '20px', 'color': colors['text']}),
                html.Div([
                    create_metric_card("Revenue", format_currency(latest_revenue), "Latest period", colors['primary'], colors),
                    create_metric_card("Net Income", format_currency(latest_net_income), "Latest period", colors['success'], colors),
                    create_metric_card("Gross Margin", f"{latest_gross_margin:.2f}%" if not pd.isna(latest_gross_margin) else "N/A", "Profitability", colors['info'], colors),
                    create_metric_card("Revenue CAGR", f"{revenue_cagr:.2f}%" if not pd.isna(revenue_cagr) else "N/A", "Growth rate", colors['warning'], colors),
                ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '20px', 'marginBottom': '30px'})
            ]),
            
            html.Div([
                html.Div([dcc.Graph(figure=fig_revenue, config={'displayModeBar': False})], 
                         style={'marginBottom': '20px'}),
                html.Div([dcc.Graph(figure=fig_profit, config={'displayModeBar': False})]),
            ])
        ])
        
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={'color': colors['danger']})

def create_technical_tab(stock, ticker, colors):
    """Create technical analysis tab."""
    try:
        hist = stock.history(period='2y')
        
        # Calculate moving averages
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        hist['MA200'] = hist['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Price chart with MAs
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['Close'],
            name='Price',
            line=dict(color=colors['primary'], width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['MA50'],
            name='MA50',
            line=dict(color=colors['warning'], width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['MA200'],
            name='MA200',
            line=dict(color=colors['danger'], width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['RSI'],
            name='RSI',
            line=dict(color=colors['info'], width=2)
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color=colors['danger'], row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=colors['success'], row=2, col=1)
        
        fig.update_layout(
            title=f"{ticker} - Technical Analysis",
            height=600,
            template='plotly_white' if colors == COLORS['light'] else 'plotly_dark',
            paper_bgcolor=colors['card'],
            plot_bgcolor=colors['card'],
            font=dict(color=colors['text']),
            hovermode='x unified',
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        
        # Calculate signals
        current_price = hist['Close'].iloc[-1]
        ma50 = hist['MA50'].iloc[-1]
        ma200 = hist['MA200'].iloc[-1]
        rsi = hist['RSI'].iloc[-1]
        
        trend = "Bullish" if current_price > ma50 > ma200 else "Bearish" if current_price < ma50 < ma200 else "Neutral"
        rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        
        return html.Div([
            html.Div([
                html.H3("📈 Technical Indicators", style={'marginBottom': '20px', 'color': colors['text']}),
                html.Div([
                    create_metric_card("Trend", trend, f"MA50: ${ma50:.2f}", colors['primary'], colors),
                    create_metric_card("RSI", f"{rsi:.2f}", rsi_signal, colors['info'], colors),
                    create_metric_card("MA50", f"${ma50:.2f}", "50-day average", colors['warning'], colors),
                    create_metric_card("MA200", f"${ma200:.2f}", "200-day average", colors['danger'], colors),
                ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '20px', 'marginBottom': '30px'})
            ]),
            
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ])
        
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={'color': colors['danger']})

def create_valuation_tab(stock, ticker, colors):
    """Create valuation analysis tab."""
    try:
        info = stock.info
        income = stock.financials.T.apply(pd.to_numeric, errors='coerce').sort_index(ascending=False)
        
        # Get valuation metrics
        pe_ratio = info.get('trailingPE', 0)
        forward_pe = info.get('forwardPE', 0)
        pb_ratio = info.get('priceToBook', 0)
        ps_ratio = info.get('priceToSalesTrailing12Months', 0)
        peg_ratio = info.get('pegRatio', 0)
        ev_ebitda = info.get('enterpriseToEbitda', 0)
        
        return html.Div([
            html.Div([
                html.H3("🎯 Valuation Metrics", style={'marginBottom': '20px', 'color': colors['text']}),
                html.Div([
                    create_metric_card("P/E Ratio", format_number(pe_ratio) if pe_ratio else "N/A", "Trailing twelve months", colors['primary'], colors),
                    create_metric_card("Forward P/E", format_number(forward_pe) if forward_pe else "N/A", "Next year estimate", colors['info'], colors),
                    create_metric_card("P/B Ratio", format_number(pb_ratio) if pb_ratio else "N/A", "Price to book", colors['success'], colors),
                    create_metric_card("P/S Ratio", format_number(ps_ratio) if ps_ratio else "N/A", "Price to sales", colors['warning'], colors),
                    create_metric_card("PEG Ratio", format_number(peg_ratio) if peg_ratio else "N/A", "P/E to growth", colors['secondary'], colors),
                    create_metric_card("EV/EBITDA", format_number(ev_ebitda) if ev_ebitda else "N/A", "Enterprise value", colors['danger'], colors),
                ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '20px'})
            ]),
            
            html.Div([
                html.H3("📊 Valuation Commentary", style={'marginTop': '40px', 'marginBottom': '20px', 'color': colors['text']}),
                html.Div([
                    html.P([
                        html.Strong("P/E Analysis: "),
                        f"A P/E ratio of {pe_ratio:.2f} " if pe_ratio else "P/E ratio not available. ",
                        "A lower P/E may indicate undervaluation relative to earnings." if pe_ratio and pe_ratio < 15 else 
                        "A higher P/E may indicate growth expectations or overvaluation." if pe_ratio and pe_ratio > 25 else ""
                    ], style={'marginBottom': '15px', 'lineHeight': '1.8'}),
                    html.P([
                        html.Strong("PEG Analysis: "),
                        f"PEG ratio of {peg_ratio:.2f} " if peg_ratio else "PEG ratio not available. ",
                        "suggests good value relative to growth." if peg_ratio and peg_ratio < 1 else
                        "indicates premium valuation relative to growth." if peg_ratio and peg_ratio > 2 else ""
                    ], style={'marginBottom': '15px', 'lineHeight': '1.8'}),
                ], style={
                    'padding': '30px',
                    'background': colors['card'],
                    'borderRadius': '15px',
                    'border': f'1px solid {colors["border"]}'
                })
            ])
        ])
        
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={'color': colors['danger']})

if __name__ == '__main__':
    app.run(debug=True, port=8050)
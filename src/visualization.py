"""
Interactive Plotly Visualization Library for Quantitative Portfolio and Risk Analysis
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Unified Palette & Styling Constants
THEME = "plotly_dark"
PALETTE = {
    "nvda": "#76B900",
    "market": "#4EA5D9",
    "positive": "#2EC4B6",
    "negative": "#FF5A5F",
    "accent": "#B79CED",
    "muted": "#9AA4B2",
    "background": "#121418",
    "card": "#1E222B",
    "grid": "#2A2E39"
}

PEER_COLORS = {
    "NVDA": "#76B900",
    "META": "#0668E1",
    "AMZN": "#FF9900",
    "MSFT": "#00A4EF",
    "TSLA": "#CC0000",
    "GOOGL": "#4285F4",
    "AMD": "#ED1C24",
    "AVGO": "#CC292B",
    "SPY": "#4EA5D9",
    "QQQ": "#A370F7",
    "SOXX": "#F39C12"
}


def _apply_standard_layout(fig: go.Figure, title: str, height: int = 500) -> go.Figure:
    """Apply uniform publication-grade dark styling."""
    fig.update_layout(
        template=THEME,
        title=dict(text=title, font=dict(size=18, color="#FFFFFF")),
        height=height,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
        paper_bgcolor=PALETTE["background"],
        plot_bgcolor=PALETTE["card"],
        margin=dict(l=60, r=40, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0.3)"
        ),
        xaxis=dict(gridcolor=PALETTE["grid"]),
        yaxis=dict(gridcolor=PALETTE["grid"])
    )
    return fig


def save_figure_html(fig: go.Figure, filepath: Union[str, Path]) -> None:
    """Save Plotly figure to standalone HTML file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(p), include_plotlyjs="cdn", full_html=True)


# 1. Historical Stock Performance
def plot_historical_prices(prices: pd.DataFrame, focus_tickers: Optional[List[str]] = None) -> go.Figure:
    if focus_tickers is None:
        focus_tickers = [c for c in ["NVDA", "AMD", "AVGO", "SOXX", "QQQ", "SPY"] if c in prices.columns]
    fig = go.Figure()
    for t in focus_tickers:
        color = PEER_COLORS.get(t, None)
        width = 2.8 if t == "NVDA" else 1.5
        fig.add_trace(go.Scatter(
            x=prices.index, y=prices[t], name=t,
            line=dict(color=color, width=width),
            hovertemplate=f"<b>{t}</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:,.2f}}<extra></extra>"
        ))
    _apply_standard_layout(fig, "Historical Stock & Benchmark Closing Prices", height=500)
    fig.update_xaxes(title="Trading Date")
    fig.update_yaxes(title="Adjusted Close ($)")
    return fig


# 2. Normalized Stock Performance
def plot_normalized_performance(prices: pd.DataFrame, focus_tickers: Optional[List[str]] = None) -> go.Figure:
    if focus_tickers is None:
        focus_tickers = [c for c in ["NVDA", "META", "AMZN", "TSLA", "MSFT", "SPY"] if c in prices.columns]
    fig = go.Figure()
    for t in focus_tickers:
        series = prices[t].dropna()
        if series.empty:
            continue
        norm = (series / series.iloc[0]) * 100.0
        color = PEER_COLORS.get(t, None)
        width = 2.8 if t == "NVDA" else 1.5
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm, name=t,
            line=dict(color=color, width=width),
            hovertemplate=f"<b>{t}</b><br>Date: %{{x|%Y-%m-%d}}<br>Indexed: %{{y:,.1f}}<extra></extra>"
        ))
    fig.add_hline(y=100.0, line_dash="dash", line_color=PALETTE["muted"])
    _apply_standard_layout(fig, "Normalized Asset Performance (Base 100 = First Observation)", height=500)
    fig.update_xaxes(title="Trading Date")
    fig.update_yaxes(title="Indexed Wealth ($100 Base)")
    return fig


# 3. Returns Distribution
def plot_returns_distribution(returns: pd.DataFrame, focus_tickers: Optional[List[str]] = None) -> go.Figure:
    if focus_tickers is None:
        focus_tickers = [c for c in ["NVDA", "META", "AMZN", "TSLA", "MSFT"] if c in returns.columns]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Daily Return Distributions (KDE / Histogram)", "5th-Percentile Downside Daily Return (VaR Proxy)")
    )
    for t in focus_tickers:
        r = returns[t].dropna() * 100.0
        q05 = float(returns[t].quantile(0.05) * 100.0)
        color = PEER_COLORS.get(t, None)
        fig.add_trace(go.Histogram(
            x=r, name=t, opacity=0.45, histnorm="probability density",
            marker_color=color, nbinsx=40
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=[t], y=[q05], name=t, marker_color=color, showlegend=False,
            text=[f"{q05:.2f}%"], textposition="outside"
        ), row=1, col=2)

    _apply_standard_layout(fig, "Return Distributions and Downside Tail Diagnostics", height=480)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Daily Return (%)", row=1, col=1)
    fig.update_yaxes(title="Probability Density", row=1, col=1)
    fig.update_xaxes(title="Asset", row=1, col=2)
    fig.update_yaxes(title="5th Percentile Return (%)", row=1, col=2)
    return fig


# 4. Rolling Volatility and Beta
def plot_rolling_volatility_beta(
    vol20: pd.Series, vol60: pd.Series, downside20: pd.Series, beta60: pd.Series, target: str = "NVDA"
) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=(
            f"{target} Annualized Realized Volatility",
            f"{target} 60-Session Rolling Market Beta (SPY Benchmark)",
            f"{target} Downside Semivolatility (Negative Returns Only)"
        )
    )
    fig.add_trace(go.Scatter(x=vol20.index, y=vol20 * 100.0, name="20-day Vol", line=dict(color=PALETTE["nvda"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=vol60.index, y=vol60 * 100.0, name="60-day Vol", line=dict(color=PALETTE["market"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=beta60.index, y=beta60, name="Rolling Beta", line=dict(color=PALETTE["accent"], width=2)), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color=PALETTE["muted"], row=2, col=1)
    fig.add_trace(go.Scatter(x=downside20.index, y=downside20 * 100.0, name="Downside Vol", line=dict(color=PALETTE["negative"], width=2)), row=3, col=1)

    _apply_standard_layout(fig, f"{target} Dynamic Risk Regimes & Market Sensitivity", height=780)
    fig.update_yaxes(title="Vol (%)", row=1, col=1)
    fig.update_yaxes(title="Beta", row=2, col=1)
    fig.update_yaxes(title="Downside Vol (%)", row=3, col=1)
    fig.update_xaxes(title="Trading Date", row=3, col=1)
    return fig


# 5. GARCH Volatility
def plot_garch_volatility(gvol: pd.Series, persistence: float, target: str = "NVDA") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gvol.index, y=gvol * 100.0, name="GARCH(1,1) Volatility",
        line=dict(color=PALETTE["accent"], width=2.2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>GARCH Vol: %{y:.2f}%<extra></extra>"
    ))
    _apply_standard_layout(
        fig,
        f"{target} GARCH(1,1) Conditional Volatility (Persistence α+β = {persistence:.3f})",
        height=480
    )
    fig.update_xaxes(title="Trading Date")
    fig.update_yaxes(title="Annualized Conditional Volatility (%)")
    return fig


# 6. Correlation Heatmap
def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        zmin=-1.0,
        zmax=1.0,
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    _apply_standard_layout(fig, "Cross-Asset Daily Return Correlation Matrix", height=600)
    fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
    return fig


# 7. Portfolio Risk-Reward Bubble Chart
def plot_risk_reward_bubble(summary_df: pd.DataFrame, target: str = "NVDA") -> go.Figure:
    fig = go.Figure()
    for t in summary_df.index:
        x_val = summary_df.loc[t, "annualized_volatility"] * 100.0
        y_val = summary_df.loc[t, "annualized_return"] * 100.0
        ratio = summary_df.loc[t, "return_vol_ratio"]
        size = 18 if t == target else 14
        color = PALETTE["nvda"] if t == target else (PALETTE["positive"] if y_val >= 0 else PALETTE["negative"])

        fig.add_trace(go.Scatter(
            x=[x_val], y=[y_val], mode="markers+text",
            name=t, text=[t], textposition="top center",
            marker=dict(size=size, color=color, line=dict(width=1.5, color="#FFFFFF")),
            hovertemplate=f"<b>{t}</b><br>Annualized Vol: {x_val:.1f}%<br>Annualized Return: {y_val:.1f}%<br>Sharpe Proxy: {ratio:.2f}<extra></extra>"
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["muted"])
    _apply_standard_layout(fig, "Annualized Risk-Return Profile across Research Universe", height=520)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Annualized Volatility (%)")
    fig.update_yaxes(title="Annualized Return (%)")
    return fig


# 8. Markowitz Efficient Frontier
def plot_efficient_frontier(
    sim_df: pd.DataFrame,
    min_vol_port: Dict[str, float],
    max_sharpe_port: Dict[str, float],
    current_port: Optional[Dict[str, float]] = None
) -> go.Figure:
    fig = go.Figure()
    # Simulated clouds
    fig.add_trace(go.Scatter(
        x=sim_df["volatility"] * 100.0,
        y=sim_df["return"] * 100.0,
        mode="markers",
        marker=dict(
            size=4,
            color=sim_df["sharpe"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        name="Random Portfolios",
        hoverinfo="skip"
    ))
    # Min Vol Portfolio
    fig.add_trace(go.Scatter(
        x=[min_vol_port["volatility"] * 100.0],
        y=[min_vol_port["return"] * 100.0],
        mode="markers+text",
        name="Minimum Volatility",
        text=["Min Vol"],
        textposition="bottom right",
        marker=dict(size=14, color="#00E5FF", symbol="diamond")
    ))
    # Max Sharpe Portfolio
    fig.add_trace(go.Scatter(
        x=[max_sharpe_port["volatility"] * 100.0],
        y=[max_sharpe_port["return"] * 100.0],
        mode="markers+text",
        name="Max Sharpe",
        text=["Max Sharpe (Tangency)"],
        textposition="top left",
        marker=dict(size=14, color="#FFD700", symbol="star")
    ))
    # Current Portfolio
    if current_port is not None:
        fig.add_trace(go.Scatter(
            x=[current_port["volatility"] * 100.0],
            y=[current_port["return"] * 100.0],
            mode="markers+text",
            name="Target Portfolio",
            text=["Target Portfolio"],
            textposition="top right",
            marker=dict(size=16, color=PALETTE["negative"], symbol="cross")
        ))

    _apply_standard_layout(fig, "Markowitz Efficient Frontier & Portfolio Risk-Return Positioning", height=540)
    fig.update_xaxes(title="Annualized Volatility (%)")
    fig.update_yaxes(title="Annualized Expected Return (%)")
    return fig


# 9. Portfolio Component Risk Contribution
def plot_risk_contribution_comparison(risk_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=risk_df.index,
        y=risk_df["weight"] * 100.0,
        name="Portfolio Weight (%)",
        marker_color=PALETTE["market"],
        text=[f"{w:.1%}" for w in risk_df["weight"]],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        x=risk_df.index,
        y=risk_df["percentage_risk_contribution"] * 100.0,
        name="Risk Contribution (%)",
        marker_color=PALETTE["negative"],
        text=[f"{rc:.1%}" for rc in risk_df["percentage_risk_contribution"]],
        textposition="outside"
    ))
    _apply_standard_layout(fig, "Capital Allocation vs. Component Risk Contribution (Euler Decomposition)", height=500)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Portfolio Asset")
    fig.update_yaxes(title="Percentage (%)")
    return fig


# 10. Portfolio Equity Curve
def plot_portfolio_equity_curve(
    port_wealth: pd.Series,
    benchmark_wealth: Optional[pd.Series] = None
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_wealth.index, y=port_wealth, name="Target Portfolio",
        line=dict(color=PALETTE["accent"], width=2.8),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Wealth: $%{y:.2f}<extra></extra>"
    ))
    if benchmark_wealth is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_wealth.index, y=benchmark_wealth, name="S&P 500 (SPY)",
            line=dict(color=PALETTE["market"], width=1.8, dash="dot"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>SPY Wealth: $%{y:.2f}<extra></extra>"
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color=PALETTE["muted"])
    _apply_standard_layout(fig, "Portfolio Cumulative Wealth Index (Initial Wealth = $1.00)", height=480)
    fig.update_xaxes(title="Trading Date")
    fig.update_yaxes(title="Cumulative Wealth ($)")
    return fig


# 11. Maximum Drawdown and Underwater Analysis
def plot_underwater_drawdowns(
    drawdowns: pd.Series,
    top_episodes: Optional[pd.DataFrame] = None
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdowns.index, y=drawdowns * 100.0,
        name="Portfolio Drawdown",
        fill="tozeroy",
        fillcolor="rgba(255, 90, 95, 0.25)",
        line=dict(color=PALETTE["negative"], width=2.0),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>"
    ))
    # Mark maximum drawdown point
    mdd_date = drawdowns.idxmin()
    mdd_val = drawdowns.min() * 100.0
    fig.add_trace(go.Scatter(
        x=[mdd_date], y=[mdd_val],
        mode="markers+text",
        name=f"Max DD ({mdd_val:.2f}%)",
        text=[f"Max DD: {mdd_val:.2f}%<br>{mdd_date.strftime('%Y-%m-%d')}"],
        textposition="bottom center",
        marker=dict(size=12, color="#FF0055", symbol="circle")
    ))

    _apply_standard_layout(fig, "Portfolio Underwater Chart & Drawdown Severity", height=500)
    fig.update_xaxes(title="Trading Date")
    fig.update_yaxes(title="Drawdown from Prior Peak (%)")
    return fig


# 12. Rolling VaR and Expected Shortfall
def plot_rolling_var_es(rolling_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["historical_var"] * 100.0,
        name="Historical 95% VaR", line=dict(color=PALETTE["market"], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["historical_es"] * 100.0,
        name="Historical 95% ES (CVaR)", line=dict(color=PALETTE["negative"], width=2.2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["parametric_var"] * 100.0,
        name="Gaussian 95% VaR", line=dict(color=PALETTE["accent"], width=1.5, dash="dot")
    ))
    _apply_standard_layout(fig, "60-Session Rolling Value-at-Risk (VaR) and Expected Shortfall (ES)", height=480)
    fig.update_xaxes(title="Trading Date")
    fig.update_yaxes(title="1-Day Downside Risk (%)")
    return fig


# 13. Monthly Performance Heatmap
def plot_monthly_heatmap(monthly_table: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        monthly_table,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    _apply_standard_layout(fig, "Portfolio Monthly Returns Matrix (%)", height=380)
    fig.update_layout(coloraxis_colorbar=dict(title="Return (%)"))
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Year")
    return fig


# 14. Monte Carlo Portfolio Loss Distribution
def plot_monte_carlo_loss_distribution(
    loss_distribution: np.ndarray,
    var95: float,
    es95: float
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=loss_distribution * 100.0,
        nbinsx=65,
        name="Scenario Loss",
        marker_color=PALETTE["market"],
        opacity=0.75
    ))
    fig.add_vline(
        x=var95 * 100.0, line_dash="dash", line_color=PALETTE["negative"],
        annotation_text=f"95% Scenario VaR: {var95:.2%}", annotation_position="top left"
    )
    fig.add_vline(
        x=es95 * 100.0, line_dash="dot", line_color=PALETTE["accent"],
        annotation_text=f"95% Scenario ES: {es95:.2%}", annotation_position="top right"
    )
    _apply_standard_layout(fig, "Correlated Monte Carlo Reputational Shock Loss Distribution", height=480)
    fig.update_xaxes(title="Simulated Portfolio Loss (%)")
    fig.update_yaxes(title="Frequency")
    return fig


# 15. Scenario-Based Portfolio Stress Testing
def plot_stress_test_scenarios(
    parallel_df: pd.DataFrame,
    target_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    if target_df is not None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Parallel Market Shock Across Holdings", "Idiosyncratic Shock Concentrated in NVDA")
        )
        fig.add_trace(go.Bar(
            x=parallel_df["shock_label"], y=parallel_df["dollar_loss"],
            marker_color=PALETTE["negative"], name="Parallel Shock Loss ($)",
            text=[f"${l:,.0f}" for l in parallel_df["dollar_loss"]], textposition="outside"
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=target_df["target_shock_label"], y=target_df["dollar_loss"],
            marker_color=PALETTE["accent"], name="NVDA Event Shock Loss ($)",
            text=[f"${l:,.0f}" for l in target_df["dollar_loss"]], textposition="outside"
        ), row=1, col=2)
        _apply_standard_layout(fig, "Portfolio Downside Stress Testing Scenarios ($100k Base)", height=480)
        fig.update_xaxes(title="Shock Magnitude", row=1, col=1)
        fig.update_yaxes(title="Dollar Loss ($)", row=1, col=1)
        fig.update_xaxes(title="Target Shock Magnitude", row=1, col=2)
        fig.update_yaxes(title="Dollar Loss ($)", row=1, col=2)
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=parallel_df["shock_label"], y=parallel_df["dollar_loss"],
            marker_color=PALETTE["negative"],
            text=[f"${l:,.0f}" for l in parallel_df["dollar_loss"]], textposition="outside"
        ))
        _apply_standard_layout(fig, "Parallel Portfolio Downside Stress ($100k Base)", height=480)
        fig.update_xaxes(title="Assumed Parallel Shock")
        fig.update_yaxes(title="Portfolio Dollar Loss ($)")
    return fig


# 16. Event Study AR/CAR Visualization
def plot_event_study_template(
    event_id: str = "E_PLACEHOLDER",
    spec: str = "Market Model"
) -> go.Figure:
    """Generate event-time panel template when event study is unverified or active."""
    days = np.arange(-10, 11)
    # Visual placeholder illustrating the dual-panel structure
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=("Daily Abnormal Return (AR)", "Cumulative Abnormal Return (CAR) with 95% CI")
    )
    fig.add_hline(y=0, line_color="white", row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["negative"])
    _apply_standard_layout(
        fig,
        f"Event Study Specification Template: {event_id} ({spec}) [Awaiting Verified Event Records]",
        height=620
    )
    fig.update_xaxes(title="Trading Sessions from Announcement (t=0)", row=2, col=1)
    fig.update_yaxes(title="Abnormal Return (%)", row=1, col=1)
    fig.update_yaxes(title="Cumulative AR (%)", row=2, col=1)
    return fig


# 17. News Sentiment Analysis Overview
def plot_sentiment_distribution(
    sentiment_counts: Optional[Dict[str, int]] = None
) -> go.Figure:
    if sentiment_counts is None:
        sentiment_counts = {"Positive": 15, "Neutral": 42, "Negative": 23}
    fig = go.Figure(go.Pie(
        labels=list(sentiment_counts.keys()),
        values=list(sentiment_counts.values()),
        hole=0.45,
        marker=dict(colors=[PALETTE["positive"], PALETTE["muted"], PALETTE["negative"]])
    ))
    _apply_standard_layout(fig, "FinBERT News Tone Sentiment Distribution (Benchmark Schema)", height=450)
    return fig

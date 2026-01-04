from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Static, Label, Button, Sparkline, Header, Footer, ProgressBar
from textual.reactive import reactive

# --- Data Models & Constants ---

THEME_COLORS = {
    "background": "#0B0E11",
    "surface": "#151921",
    "card": "#1E232B",
    "primary": "#3b82f6",
    "bullish": "#10b981",
    "bearish": "#ef4444",
    "mixed": "#f59e0b",
    "neutral": "#94a3b8",
    "border": "#2A303C",
}

# --- Custom Widgets ---

class SymbolHeader(Static):
    """The top bar containing BTC price and controls."""
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="header-left"):
            # Icon placeholder using unicode
            yield Label("₿", classes="btc-icon")
            with Vertical(classes="header-title-box"):
                with Horizontal():
                    yield Label("BTC-USD", classes="symbol-name")
                with Horizontal(classes="price-line"):
                    yield Label("$64,242.50", classes="price")
                    yield Label("▲ 2.4%", classes="percent-change bullish")
            
            yield Static("", classes="separator")
            
            with Vertical(classes="volume-box"):
                yield Label("24H VOLUME", classes="label-tiny")
                yield Label("$32.4B", classes="volume-val")

        with Horizontal(classes="header-right"):
            yield Button("BTC", classes="ctrl-btn", variant="primary")
            yield Button("ETH", classes="ctrl-btn")
            yield Button("SOL", classes="ctrl-btn")
            yield Button("XRP", classes="ctrl-btn")
            yield Button("⚙", classes="settings-btn")

class AggregatorPanel(Static):
    """The middle panels for structure analysis."""
    
    def __init__(self, title, tags, signal_text, signal_color, icon, **kwargs):
        super().__init__(**kwargs)
        self.panel_title = title
        self.tags = tags
        self.signal_text = signal_text
        self.signal_color = signal_color
        self.icon = icon

    def compose(self) -> ComposeResult:
        with Horizontal(classes="agg-inner"):
            with Horizontal(classes="agg-left"):
                yield Label(self.icon, classes="agg-icon")
                with Vertical():
                    yield Label(self.panel_title, classes="agg-title")
                    yield Label(self.tags, classes="agg-tags")
            
            with Horizontal(classes="agg-right"):
                yield Label("AGGREGATE SIGNAL", classes="label-tiny")
                # Blinking dot simulation
                yield Label("●", classes=f"status-dot {self.signal_color}")
                yield Label(self.signal_text, classes=f"status-text {self.signal_color}")

class MarketCard(Static):
    """A generic card for the bottom grid."""

    def __init__(
        self, 
        tag_main, tag_sub, 
        score, score_color,
        title, subtitle,
        row1_label, row1_val, row1_color,
        row2_label, row2_val, row2_color,
        footer_label, footer_val, footer_extra,
        progress_val, progress_color,
        spark_data,
        ranges,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tag_main = tag_main
        self.tag_sub = tag_sub
        self.score = score
        self.score_color = score_color
        self.card_title = title
        self.subtitle = subtitle
        self.row1_label = row1_label
        self.row1_val = row1_val
        self.row1_color = row1_color
        self.row2_label = row2_label
        self.row2_val = row2_val
        self.row2_color = row2_color
        self.footer_label = footer_label
        self.footer_val = footer_val
        self.footer_extra = footer_extra
        self.progress_val = progress_val
        self.progress_color = progress_color
        self.spark_data = spark_data
        self.ranges = ranges

    def compose(self) -> ComposeResult:
        # Header Row
        with Horizontal(classes="card-header"):
            with Horizontal():
                yield Label("●", classes=f"dot-small {self.score_color}")
                yield Label(f" {self.tag_main} • {self.tag_sub}", classes="card-tag")
            
            with Horizontal(classes="score-box"):
                yield Label("SCORE", classes="label-micro")
                yield Label(str(self.score), classes=f"score-val {self.score_color}")

        # Title Row
        with Vertical(classes="card-title-box"):
            yield Label(self.card_title, classes="card-main-title")
            yield Label(self.subtitle, classes="card-subtitle")

        # Data Rows
        with Vertical(classes="card-data-grid"):
            with Horizontal(classes="data-row"):
                yield Label(self.row1_label, classes="data-label")
                yield Label(self.row1_val, classes=f"data-value {self.row1_color}")
            
            with Horizontal(classes="data-row"):
                yield Label(self.row2_label, classes="data-label")
                yield Label(self.row2_val, classes=f"data-value {self.row2_color}")

        # Footer / Progress / Sparkline
        with Vertical(classes="card-footer"):
            with Horizontal(classes="footer-text-row"):
                yield Label(self.footer_label, classes="data-label")
                with Horizontal(classes="footer-right"):
                    yield Label(self.footer_val, classes="footer-val")
                    if self.footer_extra:
                        yield Label(self.footer_extra, classes="footer-extra")
            
            # Use Sparkline to simulate the bottom graph, or ProgressBar for bars
            if self.progress_val is not None:
                # Custom styled container to look like a bar
                with Container(classes="bar-container"):
                    yield Static("", classes=f"bar-fill {self.progress_color}", id=f"bar-{self.score}")
            
            if self.spark_data:
                 yield Sparkline(self.spark_data, summary_function=max, classes=f"spark-{self.score_color}")

            with Horizontal(classes="range-labels"):
                yield Label(f"L: {self.ranges[0]}", classes="range-text")
                yield Label(f"H: {self.ranges[1]}", classes="range-text")

# --- Main App ---

class CryptoDashboardApp(App):
    CSS = """
    /* --- Global & Layout --- */
    Screen {
        background: #0B0E11;
        color: #94a3b8;
    }

    /* --- Typography Colors --- */
    .bullish { color: #10b981; }
    .bearish { color: #ef4444; }
    .mixed { color: #f59e0b; }
    .neutral { color: #94a3b8; }
    .primary { color: #3b82f6; }
    .white { color: white; }

    /* --- Header Component --- */
    SymbolHeader {
        height: 5;
        background: #1e232b;
        border: solid #2A303C;
        margin: 1;
        padding: 0 2;
        layout: horizontal;
    }

    .header-left { align-vertical: middle; width: 60%; }
    .header-right { align-vertical: middle; align: right middle; width: 40%; }
    
    .btc-icon {
        color: #3b82f6;
        background: rgba(59, 130, 246, 0.1);
        border: solid rgba(59, 130, 246, 0.2);
        width: 5;
        height: 3;
        content-align: center middle;
        text-style: bold;
        margin-right: 2;
    }

    .header-title-box { width: auto; height: auto; }
    .symbol-name { color: white; text-style: bold; }
    .price { color: white; }
    .label-tiny { color: #555; }
    .volume-val { color: #ccc; }
    
    .separator {
        width: 1;
        height: 2;
        background: #2A303C;
        margin: 0 2;
    }

    .ctrl-btn {
        min-width: 8;
        height: 1;
        border: none;
        background: #151921;
        color: #777;
        margin-right: 1;
    }
    .ctrl-btn.-primary { background: #2A303C; color: white; }
    .settings-btn { min-width: 4; background: #151921; border: solid #2A303C; }

    /* --- Aggregator Panels --- */
    .agg-container { height: 5; margin: 0 1; }
    
    AggregatorPanel {
        width: 1fr;
        height: 5;
        border: solid #2A303C;
        background: #151921;
        padding: 0 1;
    }

    .agg-inner { align-vertical: middle; }
    .agg-left { align-vertical: middle; }
    .agg-right { align-vertical: middle; align: right middle; }
    
    .agg-icon {
        width: 5;
        height: 3;
        content-align: center middle;
        border: solid #333;
        background: #0B0E11;
        margin-right: 2;
    }
    
    .agg-title { color: white; text-style: bold; }
    .agg-tags { color: #555; }
    
    .status-dot { margin-left: 2; margin-right: 1; }
    .status-text { text-style: bold; }

    /* --- Market Grid --- */
    .grid-container {
        layout: grid;
        grid-size: 3 2; /* 3 columns, 2 rows */
        grid-gutter: 1;
        margin: 1;
        height: auto;
    }

    MarketCard {
        background: #151921; /* glass panel approximation */
        border: solid #2A303C;
        height: 18;
        padding: 1;
    }
    
    /* Card Internals */
    .card-header { height: 1; margin-bottom: 1; }
    .dot-small { margin-right: 1; }
    
    .score-box { 
        border: solid #333; 
        background: rgba(0,0,0,0.3); 
        padding: 0 1; 
        height: 1;
        align-vertical: middle;
    }
    .label-micro { margin-right: 1; }
    .score-val { text-style: bold; }

    .card-title-box { height: 3; margin-bottom: 1; }
    .card-main-title { color: white; text-style: bold; }
    .card-subtitle { color: #666; }

    .card-data-grid { margin-bottom: 1; }
    .data-row { border-bottom: solid #222; height: 1; padding-bottom: 0; margin-bottom: 1; }
    .data-label { color: #777; }
    .data-value { text-style: bold; }

    .card-footer { height: auto; }
    .footer-text-row { margin-bottom: 1; }
    .footer-val { color: white; }
    .footer-extra { margin-left: 1; }

    /* Custom Progress Bar CSS Hack */
    .bar-container {
        height: 1;
        background: #0B0E11;
        width: 100%;
        margin-bottom: 1;
    }
    .bar-fill { height: 1; }
    #bar-12 { width: 85%; }
    #bar-42 { width: 35%; }
    #bar-85 { width: 72%; }
    #bar-76 { width: 60%; }
    #bar-50 { width: 0%; display: none; } /* Using sparkline or text instead */
    #bar-92 { width: 85%; background: #4b5563; }

    /* Sparklines */
    .spark-bearish { color: #ef4444; }
    .spark-neutral { color: #94a3b8; }
    .spark-bullish { color: #10b981; }
    .spark-primary { color: #3b82f6; }
    
    .range-labels { margin-top: 1; }
    .range-text { color: #444; }
    """

    def compose(self) -> ComposeResult:
        # Top Bar
        yield SymbolHeader()

        # Aggregator Row
        with Horizontal(classes="agg-container"):
            yield AggregatorPanel(
                title="Short-Term Structure",
                tags="10S • 1M • 15M",
                signal_text="MIXED",
                signal_color="mixed",
                icon="⏱"
            )
            yield AggregatorPanel(
                title="Long-Term Structure",
                tags="1H • 4H • DAILY",
                signal_text="BULLISH",
                signal_color="bullish",
                icon="📊"
            )

        # Card Grid
        with Grid(classes="grid-container"):
            # 1. Dump (10S)
            yield MarketCard(
                tag_main="10S", tag_sub="HFT", score=12, score_color="bearish",
                title="Dump", subtitle="High sell pressure detected",
                row1_label="Tick Delta", row1_val="High ▼", row1_color="bearish",
                row2_label="Spread", row2_val="Tight", row2_color="white",
                footer_label="Order Flow", footer_val="-45.2 BTC", footer_extra="",
                progress_val=85, progress_color="bearish",
                spark_data=[50, 48, 45, 40, 42, 35, 30, 25, 20, 15, 12, 10], # Down trend
                ranges=("64,100", "64,250")
            )
            
            # 2. Chop (1M)
            yield MarketCard(
                tag_main="1M", tag_sub="SCALP", score=42, score_color="neutral",
                title="Chop", subtitle="Lack of directional conviction",
                row1_label="Momentum", row1_val="Low", row1_color="neutral",
                row2_label="Volatility", row2_val="Muted", row2_color="white",
                footer_label="Vol Delta", footer_val="-24.5 BTC", footer_extra="",
                progress_val=35, progress_color="neutral",
                spark_data=[20, 25, 22, 18, 20, 28, 24, 20, 22, 25, 20, 22], # Choppy
                ranges=("64,180", "64,220")
            )

            # 3. Breakout (15M)
            yield MarketCard(
                tag_main="15M", tag_sub="INTRADAY", score=85, score_color="bullish",
                title="Breakout", subtitle="Price escaping consolidation",
                row1_label="Momentum", row1_val="Strong ▲", row1_color="bullish",
                row2_label="VWAP Dev", row2_val="+2.1σ", row2_color="bullish",
                footer_label="RSI (14)", footer_val="72.4", footer_extra="",
                progress_val=72, progress_color="bullish",
                spark_data=[10, 12, 11, 13, 15, 18, 25, 35, 45, 50, 55, 60], # Breakout
                ranges=("63,500", "64,400")
            )

            # 4. Trending (1H)
            yield MarketCard(
                tag_main="1H", tag_sub="HOURLY", score=76, score_color="primary",
                title="Trending", subtitle="Sustained directional movement",
                row1_label="ADX Strength", row1_val="35.2 ✓", row1_color="primary",
                row2_label="Volume", row2_val="Average", row2_color="white",
                footer_label="Open Interest", footer_val="+1.2M", footer_extra="(+2%)",
                progress_val=60, progress_color="primary",
                spark_data=[10, 15, 20, 25, 30, 35, 38, 42, 45, 48, 50, 52], # Linear Trend
                ranges=("62,800", "64,500")
            )

            # 5. Ranging (4H)
            yield MarketCard(
                tag_main="4H", tag_sub="SWING", score=50, score_color="neutral",
                title="Ranging", subtitle="Oscillating between key levels",
                row1_label="BB Width", row1_val="Tight ⇥", row1_color="white",
                row2_label="Stoch RSI", row2_val="82", row2_color="bearish",
                footer_label="Key Levels", footer_val="R: 65.2k | S: 62.1k", footer_extra="",
                progress_val=None, progress_color="neutral", # No bar, using spark
                spark_data=[30, 40, 50, 40, 30, 20, 30, 40, 50, 60, 50, 40], # Sine wave
                ranges=("60,000", "66,000")
            )

            # 6. Bullish (Daily)
            yield MarketCard(
                tag_main="DAILY", tag_sub="MACRO", score=92, score_color="bullish",
                title="Bullish", subtitle="Strong structural uptrend",
                row1_label="Trend", row1_val="> MA200 ▲", row1_color="bullish",
                row2_label="Vol Regime", row2_val="Expansion", row2_color="white",
                footer_label="Major Resistance", footer_val="69,000", footer_extra="",
                progress_val=85, progress_color="neutral", # Grey bar
                spark_data=[10, 12, 15, 20, 28, 35, 45, 55, 68, 80, 90, 95], # Exponential
                ranges=("54,000", "64,500")
            )

        yield Footer()

if __name__ == "__main__":
    app = CryptoDashboardApp()
    app.run()
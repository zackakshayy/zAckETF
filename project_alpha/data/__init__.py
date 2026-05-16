"""project_alpha.data — data access layer (point-in-time correct)."""
from project_alpha.data.universe import (
    load_russell_panel,
    universe_for_date,
    sector_targets_for_date,
)
from project_alpha.data.prices import (
    load_eod,
    load_dividends,
    load_splits,
    build_returns_panel,
    adv20_panel,
    last_price_series,
)
from project_alpha.data.fundamentals import (
    load_fundamental_features,
    point_in_time_features,
    point_in_time_earnings,
)
from project_alpha.data.news import (
    list_monthly_files,
    load_news_window,
    aggregate_sentiment_window,
)
from project_alpha.data.macro import (
    load_macro_panel,
)

__all__ = [
    "load_russell_panel", "universe_for_date", "sector_targets_for_date",
    "load_eod", "load_dividends", "load_splits", "build_returns_panel",
    "adv20_panel", "last_price_series",
    "load_fundamental_features", "point_in_time_features", "point_in_time_earnings",
    "list_monthly_files", "load_news_window", "aggregate_sentiment_window",
    "load_macro_panel",
]

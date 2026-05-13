#!/bin/bash
# Run the full PM analytics suite. Requires a completed backtest in reports_dir.
#
# Usage:
#   export PROJECT_ALPHA_REPORTS_DIR=/path/to/_px_reports
#   bash scripts/pm_run_all.sh
#
# Or pass the reports dir explicitly:
#   bash scripts/pm_run_all.sh /path/to/_px_reports

set -e

REPORTS_DIR="${1:-${PROJECT_ALPHA_REPORTS_DIR:-./reports}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================"
echo "  PM Analytics Suite — Alpha Engine ETF"
echo "  Reports dir: $REPORTS_DIR"
echo "================================================================"

echo ""
echo "▶ [1/3] Factor & Sector Exposure Analysis"
echo "----------------------------------------------------------------"
python "$SCRIPT_DIR/pm_exposure_analysis.py" --reports-dir "$REPORTS_DIR"

echo ""
echo "▶ [2/3] Tracking Error Attribution"
echo "----------------------------------------------------------------"
python "$SCRIPT_DIR/pm_tracking_error.py" --reports-dir "$REPORTS_DIR"

echo ""
echo "▶ [3/3] Investment Thesis & Consistency"
echo "----------------------------------------------------------------"
python "$SCRIPT_DIR/pm_investment_thesis.py" --reports-dir "$REPORTS_DIR"

echo ""
echo "================================================================"
echo "  ✓ COMPLETE"
echo "  Outputs in: $REPORTS_DIR/plots_pm/"
echo "================================================================"
ls -la "$REPORTS_DIR/plots_pm/" 2>/dev/null | grep -E "^-" | awk '{print "  ", $NF}'

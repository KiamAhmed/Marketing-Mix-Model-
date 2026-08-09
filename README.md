# Marketing Mix Model (MMM) for Division A

## Project Summary
This project builds a regression model to understand how different marketing 
channels (Email, Google, Facebook, Paid Views, Organic Views, Affiliate) 
drive sales for Division A.

## Business Question
Which marketing channels contribute most to sales, and which channels give 
the best return on investment?

## Data
- **Source:** Sample Media Spend Data.xlsx (Division A only)
- **Time period:** January 2018 to November 2020 (113 weeks)
- **Channels:** Paid Views, Organic Views, Google Impressions, Email Impressions, 
  Facebook Impressions, Affiliate Impressions
- **Target:** Sales

## Method
I built an Ordinary Least Squares (OLS) regression model predicting sales from 
marketing channel activity. The model uses raw (non-transformed) spend data.

**Why regression?** It shows statistical relationships between channels and sales, 
allowing us to estimate each channel's contribution.

**Model performance:** R² = 0.665 (explains 66.5% of sales variation)

## Key Findings

### By Total Contribution:
1. **Email Impressions:** $4.2M contribution (largest)
2. **Google Impressions:** $2.8M contribution
3. **Facebook Impressions:** $2.6M contribution

### By Efficiency (ROI per unit of activity):
1. **Organic Views:** 0.21 (most efficient)
2. **Facebook Impressions:** 0.14
3. **Email Impressions:** 0.10

### Concerning Findings:
- **Paid Views** (ROI: -0.54) and **Affiliate Impressions** (ROI: -6.32) 
  show negative correlations with sales
- This suggests these channels either aren't working effectively or there's 
  a data quality issue worth investigating

## Recommendation
**Focus budget on Organic Views and Facebook** for efficiency. Maintain Email 
and Google for volume. **Investigate Paid Views and Affiliate channels** — their 
negative ROI suggests poor performance or targeting issues.

## Limitations
1. **Correlation ≠ Causation** — The model shows associations, not proof that 
   these channels cause sales. Other factors could be at play.
2. **No time-lag effects** — The model assumes marketing effects are immediate. 
   In reality, ads may influence purchases weeks later (addressed via adstock 
   testing in Step 6, which didn't improve fit).
3. **No external factors** — The model doesn't account for seasonality, 
   competitor activity, pricing changes, or promotions.
4. **Multicollinearity** — Email (VIF=9.31) and Affiliate (VIF=7.11) channels 
   are correlated with others, making their individual coefficients less reliable.
5. **Small dataset** — 113 observations is modest; more data would improve 
   model stability.

## Future Improvements
- Test non-linear diminishing returns curves (beyond simple log transformation)
- Include external variables (holidays, competitor spend, price changes)
- Build separate models by product category or geography
- Validate model on a held-out test period to check prediction accuracy
- Investigate why Paid and Affiliate show negative effects

## Files
- `mmm_analysis.py` — Main analysis code
- `Sample_Media_Spend_Data.xlsx` — Input data
- This README

---

**Built by:** [Kiam Ahmed]  
**Date:** August 2026

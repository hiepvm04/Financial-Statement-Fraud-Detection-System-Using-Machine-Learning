## EDA Summary

The exploratory data analysis provides several important insights into the characteristics of firms associated with potential financial statement fraud.

### 1. Fraud Distribution

The dataset contains both fraudulent and non-fraudulent observations, with fraud representing a minority but still substantial portion of the sample. This moderate class imbalance is common in fraud detection problems and should be considered during model development.

### 2. Temporal Pattern

The fraud rate varies slightly across years but remains relatively stable overall. While certain years exhibit higher fraud rates, no clear long-term upward or downward trend is observed. This suggests that fraudulent reporting behavior may be influenced by firm-level factors rather than purely macroeconomic conditions.

### 3. Financial Ratio Distributions

Most financial ratios exhibit skewed distributions and heavy tails, which is typical for accounting data. This indicates substantial heterogeneity across firms in terms of profitability, leverage, asset structure, and operational performance.

### 4. Differences Between Fraud and Non-Fraud Firms

Comparisons between fraud and non-fraud observations reveal several systematic differences:

- Fraudulent firms tend to have **lower profitability**, particularly in ROA and ROE.
- **Cash flow quality indicators** such as CFO-based ratios appear weaker for fraud firms.
- **Receivables-related ratios** (e.g., Receivables to Revenue, Receivables to Assets) are generally higher among fraud observations, suggesting potential revenue recognition issues.
- Fraud firms also exhibit **higher leverage**, reflected in higher Debt-to-Assets and Debt-to-Equity ratios.
- The proportion of **soft assets** is typically higher among fraud firms, indicating a greater reliance on assets that may be more difficult to verify.

These patterns align with the intuition that firms experiencing financial pressure or declining performance may have stronger incentives to manipulate financial statements.

### 5. Correlation Structure

Correlation analysis indicates that several financial variables are moderately related, particularly among profitability measures and leverage indicators. While these relationships are expected given the structure of financial statements, potential multicollinearity should be considered in the feature selection stage.

### 6. Feature Relevance

Both correlation analysis and mutual information measures suggest that profitability indicators (such as ROA and ROE), receivables-related ratios, leverage metrics, and cash-flow-based indicators contain useful signals for distinguishing fraudulent firms.

### Conclusion

Overall, the exploratory analysis highlights several financial characteristics that may be associated with fraudulent financial reporting. These insights provide a foundation for the next stage of the analysis, where feature selection techniques will be applied to identify the most informative variables for building predictive fraud detection models.

# Dow Jones Industrial Average Trend Analysis

## Overview
This research delves into the stock market trends of the Dow Jones Industrial Average (DJIA), focusing on data starting from 1992. The project combines time series analysis, moving averages, and volatility measures to understand market behaviors and predict future movements.

## Features
- **Time Series Analysis**: Employed to identify patterns or trends in the dataset over time.
- **Moving Averages**: Utilized to smooth out short-term fluctuations and highlight longer-term trends.
- **Average True Range (ATR)**: A measure developed to assess market volatility.
- **LSTM Model**: Used for predicting the next day's closing price based on historical data.

## Data Retrieval
- Data for the Dow Jones Industrial Average was retrieved from Yahoo Finance using the `yfinance` Python module.

## Key Insights
- Explored the correlation between major global events and market reactions.
- Identified significant growth of the DJIA since 1992 despite numerous challenges.
- Analyzed market volume data to spot trends in buying and selling.
- Used LSTM for short-term stock price predictions, highlighting the complexities of longer-term predictions.

## Challenges
- **Data Management**: Dealing with vast datasets and ensuring data integrity.
- **Technique Selection**: Deciding on suitable analytical techniques considering various options.
- **Data Accuracy**: Ensuring the reliability of data sourced from 'yfinance'.
- **Predictive Challenges with LSTM**: Faced difficulties in extending predictions beyond a single day.

## Conclusion
This research offers a holistic perspective on stock market dynamics, providing a balanced view between quantitative data and understanding human behaviors in the market. While the predictive model works best for short-term predictions, the overarching insights are invaluable for anyone in the finance sector.

## How to Run
1. Ensure Python is installed and set up on your system.
2. Install the required libraries: `pip install pandas matplotlib yfinance tensorflow`.
3. Clone the repository: `git clone <repository-link>`.
4. Navigate to the directory and run the main script.

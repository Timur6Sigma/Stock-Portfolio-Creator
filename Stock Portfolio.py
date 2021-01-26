import numpy as np
import pandas as pd
import requests

# load the data
df = pd.read_csv("Stocks.csv")

# Set the Date as the index
df = df.set_index(pd.DatetimeIndex(df["Date"].values))

# Remove the Date column
df.drop(columns=["Date"], axis=1, inplace=True)

# Get the assets /tickers
assets = df.columns

# Optimize the portfolio
from pypfopt.efficient_frontier import  EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Calculate the expected annualized returns and the annualized sample covariance matrix of the daily asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimzie for the maximal Sharpe ratio
ef = EfficientFrontier(mu, S) # Create the Efficient Frontier Object
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(cleaned_weights)
print(ef.portfolio_performance(verbose=True))

# Get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

portfolio_val = 5000
latest_prices = get_latest_prices(df)
print("\n", latest_prices)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_val)
allocation, leftover = da.lp_portfolio()
print("\n", "Discrete allocation [Ticker:Share size]: ", allocation)
print("Fund Remaining /Not invested: $", leftover)


# Create a function to get the companies name
def get_company_name(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query="+symbol+"&region=1&lang=en"
    result = requests.get(url).json()
    for r in result["ResultSet"]["Result"]:
        if r["symbol"] == symbol:
            return r["name"]


# Store the company name into a list
company_name = []
for symbol in allocation:
    company_name.append(get_company_name(symbol))

# Get the discrete allocation values
discrete_allocation_list = []
for symbol in allocation:
    discrete_allocation_list.append(allocation.get(symbol))

# Create a dataframe for the portfolio
portfolio_df = pd.DataFrame(columns= ["Company_name", "Company_Ticker", "Discrete_val"+str(portfolio_val)])

portfolio_df["Company_name"] = company_name
portfolio_df["Company_Ticker"] = allocation
portfolio_df["Discrete_val"+str(portfolio_val)] = discrete_allocation_list

# Show the portfolio
print("\n", portfolio_df)
# Load libraries and models
import streamlit as st
import io
from utils.models import VietnamStocks, NasdaqStocks

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

def main():
    with st.sidebar:
        st.header("Configuration")
        
        # Choose market
        market_options = ("HNX", "UPCOM", "NASDAQ")
        market_selected = st.selectbox(
            label = "Choose stock market",
            options = market_options,
        )

        # Choose sector
        sectors = {
            "NASDAQ": ["Consumer Non-Durables", "Consumer Services", "Health Technology", "Retail Trade", "Technology Services"],
            "UPCOM": ["Food & Beverage", "Health Care", "Technology"],
            "HNX": ["Basic Resources", "Chemicals", "Media"]}
        sector_options = sectors[market_selected]
        sector_selected = st.selectbox(
            label = "Choose sector",
            options = sector_options,
        )
        
        # Choose company
        companies = {
            "Basic Resources": ["NBC", "TKU"],
            "Chemicals": ["PLC"],
            "Consumer Non-Durables": ["MDLZ", "MNST"],
            "Consumer Services": ["CMCSA", "SBUX", "MAR", "CTAS", "SIRI"],
            "Food & Beverage": ["AGF", "HNM", "ICF", "IFS", "MPC", "TS4"],
            "Health Technology": ["ISRG", "AMGN", "GILD", "REGN", "ILMN", "VRTX", "IDXX", "ALGN", "DXCM", "BIIB", "SGEN"],
            "Media": ["DAE", "EBS", "HTP", "SGD", "STC", "TPH"],
            "Real Estate": ["KHA"],
            "Retail Trade": ["AMZN", "COST", "ORLY", "WBA", "EBAY", "ROST", "DLTR"],
            "Technology": ["LTC"],
            "Technology Services": ["MSFT", "CSCO", "NFLX", "INTU", "ADP", "FISV", "ADSK", "NTES", "SNPS", "BIDU", "CDNS", "PAYX", "CTSH", "VRSN"]}
        company_options = companies[sector_selected]
        company_selected = st.selectbox(
            label = "Choose ticker",
            options = company_options,
        )

    # Build model
    if market_selected == "NASDAQ":
        currency = "$"
        stk_model = NasdaqStocks(sectors = [sector_selected], pre_trained = True)
    else:
        currency = "VND "
        stk_model = VietnamStocks(market = market_selected, sectors = [sector_selected], pre_trained = True)
        
    _ = stk_model.init_model()
    
    # Forecast
    _, mse, img_test = stk_model.get_test(company_selected, currency)
    _, img_forecast = stk_model.get_forecast(company_selected, currency)
    
    # Get recommendations
    sec_port = stk_model.get_portfolio(currency)
    com_stat = stk_model.get_statistics(company_selected, currency)
    com_rec = stk_model.get_recommendation(company_selected, currency)
        
    # Display
    st.title(f"Stock Prediction: {company_selected} ({market_selected})")
    
    ## Section 1
    st.header("Overview")
    st.markdown(f"  - Ticker: {company_selected}")
    st.markdown(f"  - Market: {market_selected}")
    st.markdown(f"  - Sector: {sector_selected}")
    
    ## Section 2
    st.header(f"Sector forecast")
    st.markdown(sec_port)
    
    ## Section 3
    st.header(f"Stock forecast")
    st.subheader("Stock Prices")
    st.markdown(f"Stock prices of *{company_selected}* on the next 7 days:")
    st.image(img_forecast)
    col3, col4 = st.columns([2, 2])
    col3.subheader("Stock statistics")
    col3.markdown(com_stat)
    col4.subheader("Trading recommendation")
    col4.markdown(com_rec)
    
    ## Section 4
    st.header("Model performance")
    col1, col2 = st.columns([3, 1])
    col1.subheader("Loss")
    col1.image(img_test)
    col2.subheader("MSE")
    col2.markdown(f"Testing's mean squared error: {mse}")
        

if __name__ == "__main__":
    st.set_page_config(
        page_title = "Stock Prediction", page_icon = ":chart_with_upwards_trend:"
    )
    main()
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made by <a href="https://github.com/ducquando">@ducquando</a></h6>',
            unsafe_allow_html = True,
        )

# Load libraries and models
import streamlit as st
import io
from utils.models import VietnamStocks, NasdaqStocks

def main():
    st.title("Stock Prediction")

    with st.sidebar:
        st.header("Configuration")
        
        # Choose market
        market_options = ("HNX", "UPCOM", "NASDAQ")
        market_selected = st.selectbox(
            label = "Choose an stock market",
            options = market_options,
        )

        # Choose sector
        sectors = {
            "NASDAQ": ["Consumer Non-Durables", "Consumer Services", "Health Technology", "Retail Trade", "Technology Services"],
            "UPCOM": ["Food & Beverage", "Health Care", "Technology"],
            "HNX": ["Basic Resources", "Chemicals", "Media"]}
        sector_options = sectors[market_selected]
        sector_selected = st.selectbox(
            label = "Choose a sector",
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
            label = "Choose a sector",
            options = company_options,
        )

    # Build model
    if market_selected == "NASDAQ":
        stk_model = NasdaqStocks(market = market_selected, sectors = [sector_selected], pre_trained = True)
    else:
        stk_model = VietnamStocks(market = market_selected, sectors = [sector_selected], pre_trained = True)
        
    _ = stk_model.init_model()
    
    # Forecast
    _, mse, img_test = stk_model.get_test(company_selected)
    _, img_forecast = stk_model.get_forecast(company_selected)
    
    # Get recommendations
    sec_port = stk_model.get_portfilio()
    com_stat = stk_model.get_statistics(company_selected)
    com_rec = stk_model.get_recommendation(company_selected)
        
    # Display
    st.header("Sector overview")
    st.markdown(sec_port)
    st.header("Model performance")
    st.image(img_test, channels = color)
    st.markdown(f"Testing's mean squared error: {mse}")
    st.header("Future forecast")
    st.image(img_forecast, channels = color)
    st.markdown(com_stat)
    st.markdown(com_rec)
        

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

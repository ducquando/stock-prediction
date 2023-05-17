# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import glob
import inspect
import textwrap
import streamlit as st

# Import models
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
            "UPCOM": ["Banks", "Financial Services", "Food & Beverage"],
            "HNX": ["Basic Resources", "Chemicals", "Technology"]}
        sector_options = sectors[market_selected]
        sector_selected = st.selectbox(
            label = "Choose a sector",
            options = sector_options,
        )
        
        # Choose company
        companies = {
            "Banks": ,
            "Basic Resources": ,
            "Chemicals": ,
            "Consumer Non-Durables": ["MDLZ", "MNST"],
            "Consumer Services": ["CMCSA", "SBUX", "MAR", "CTAS", "SIRI"],
            "Financial Services": ,
            "Food & Beverage": ,
            "Health Technology": ["ISRG", "AMGN", "GILD", "REGN", "ILMN", "VRTX", "IDXX", "ALGN", "DXCM", "BIIB", "SGEN"],
            "Retail Trade": ["AMZN", "COST", "ORLY", "WBA", "EBAY", "ROST", "DLTR"],
            "Technology",
            "Technology Services": ["MSFT", "CSCO", "NFLX", "INTU", "ADP", "FISV", "ADSK", "NTES", "SNPS", "BIDU", "CDNS", "PAYX", "CTSH", "VRSN"]}
        company_options = companies[sector_selected]
        company_selected = st.selectbox(
            label = "Choose a sector",
            options = company_options,
        )
        
        
        demo, url = (
            ST_DEMOS[selected_page]
            if selected_api == "echarts"
            else ST_PY_DEMOS[selected_page]
        )

        if selected_api == "echarts":
            st.caption(
                """ECharts demos are extracted from https://echarts.apache.org/examples/en/index.html,
            by copying/formattting the 'option' json object into st_echarts.
            Definitely check the echarts example page, convert the JSON specs to Python Dicts and you should get a nice viz."""
            )
        if selected_api == "pyecharts":
            st.caption(
                """Pyecharts demos are extracted from https://github.com/pyecharts/pyecharts-gallery,
            by copying the pyecharts object into st_pyecharts.
            Pyecharts is still using ECharts 4 underneath, which is why the theming between st_echarts and st_pyecharts is different."""
            )

    demo()

    sourcelines, _ = inspect.getsourcelines(demo)
    with st.expander("Source Code"):
        st.code(textwrap.dedent("".join(sourcelines[1:])))
    st.markdown(f"Credit: {url}")


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

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

st.set_page_config(page_title="Portfólió Optimalizáló", layout="wide")

st.title("💰 Pénzügyi Portfólió Optimalizáló")
st.markdown("""
Ez az interaktív alkalmazás a **Modern Portfólió Elmélet (MPT)** alapján segít megtalálni az optimális befektetési arányokat.
Válaszd ki a részvényeket, és a Monte Carlo szimuláció megkeresi a leghatékonyabb portfóliót!
""")

STATIC_SP500_LIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'V', 
    'PG', 'MA', 'HD', 'DIS', 'NFLX', 'ADBE', 'PFE', 'KO', 'TMO', 
    'CSCO', 'CRM', 'ORCL', 'NKE', 'INTC', 'CMCSA', 'PEP', 'ABT', 
    'WMT', 'UNH', 'VZ', 'MCD', 'COST', 'CVX', 'XOM', 'MRK', 'BAC',
    'T', 'SBUX', 'LOW', 'LMT', 'GE', 'GM', 'F', 'AMD', 'SPY', 
    'BABA', 'BTC-USD', 'ETH-USD', 'VOO', 'QQQ', 'MS', 'GS', 'CAT',
    'HON', 'MMM', 'BA', 'LRCX', 'MU', 'ZM', 'SHOP'
]

# Töröld a get_sp500_tickers() függvényt!

# --- 3. FÜGGVÉNYEK (CACHING-EL) ---

@st.cache_data # Ez a dekorátor elmenti az eredményt, hogy ne kelljen mindig letölteni
def get_stock_data(tickers, start):
    if not tickers:
        return None
    # Adatok letöltése
    data = yf.download(tickers, start=start)
    
    # Adattisztítás: Kezeljük a 'Close' vagy 'Adj Close' oszlopokat
    if 'Adj Close' in data:
        stock_data = data['Adj Close']
    elif 'Close' in data:
        stock_data = data['Close']
    else:
        return None # Hiba esetén
        
    return stock_data

def calculate_portfolio(stock_data, num_simulations, risk_free_rate):
    # Log hozamok
    log_returns = np.log(stock_data / stock_data.shift(1)).dropna()
    
    # Statisztikák
    TRADING_DAYS = 252
    annual_returns = log_returns.mean() * TRADING_DAYS
    cov_matrix = log_returns.cov() * TRADING_DAYS
    
    num_assets = len(annual_returns)
    real_tickers = annual_returns.index.tolist()
    
    # Tömbök előkészítése
    all_weights = np.zeros((num_simulations, num_assets))
    port_returns = np.zeros(num_simulations)
    port_volatility = np.zeros(num_simulations)
    sharpe_ratio = np.zeros(num_simulations)
    
    # Monte Carlo Ciklus
    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        
        port_returns[i] = np.sum(annual_returns * weights)
        port_volatility[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio[i] = (port_returns[i] - risk_free_rate) / port_volatility[i]
        all_weights[i,:] = weights
        
    return all_weights, port_returns, port_volatility, sharpe_ratio, real_tickers


# --- 2. OLDALSÁV (SIDEBAR) - BEÁLLÍTÁSOK ---
st.sidebar.header("⚙️ Beállítások")

# 1. Ticker lista használata (a statikus listával)
all_available_tickers = STATIC_SP500_LIST

# Alapértelmezett beállítás
default_selection = all_available_tickers[:10] 

# 2. Multiselect a fő listához
selected_from_list = st.sidebar.multiselect(
    "1. Válassz a listából (kb. 50+ db):", 
    all_available_tickers, 
    default=default_selection
)

# 3. Kézi beviteli mező
custom_ticker_input = st.sidebar.text_input(
    "2. Kézi hozzáadás (Vesszővel elválasztva, Pl: 'OTP.BU, RICHTER'):", 
    value=""
)

# 4. Kombinálás és Tisztítás
custom_tickers = []
if custom_ticker_input:
    # A beírt szöveget vessző mentén szétválasztjuk, kiszedjük a szóközöket, és nagybetűsre alakítjuk (yfinance-hoz)
    # Csak azok az elemek kerülnek be, amelyek nem üresek
    custom_tickers = [t.strip().upper() for t in custom_ticker_input.split(',') if t.strip()]

# Végleges lista: Multiselect + Kézi beviteli lista, duplikációk kiszűrése (set)
final_ticker_set = set(selected_from_list) | set(custom_tickers)
selected_tickers = list(final_ticker_set)

# Biztosítani kell, hogy a selected_tickers változó legyen átadva a get_stock_data-nak
if not selected_tickers:
    st.sidebar.warning("Kérlek, válassz ki vagy gépelj be legalább egy tickert a futtatáshoz!")

# A többi beállítás változatlan
start_date = st.sidebar.date_input("Kezdő dátum:", value=pd.to_datetime("2020-12-01"))
num_simulations = st.sidebar.slider("Szimulációk száma (Monte Carlo):", 1000, 20000, 10000)
risk_free_rate = st.sidebar.number_input("Kockázatmentes hozam (pl. 0.04 = 4%):", value=0.04, step=0.01)

run_button = st.sidebar.button("🚀 Szimuláció Futtatása")


# --- 4. FŐ LOGIKA ---

if run_button:
    with st.spinner('Adatok letöltése és szimuláció futtatása... ⏳'):
        
        # 1. Adatok beszerzése
        stock_data = get_stock_data(selected_tickers, start_date)
        
        if stock_data is None or stock_data.empty:
            st.error("Hiba az adatok letöltésekor. Ellenőrizd a Ticker kódokat!")
        else:
            # 2. Számítások
            weights, returns, volatility, sharpe, tickers = calculate_portfolio(stock_data, num_simulations, risk_free_rate)
            
            # 3. Optimumok keresése
            max_sharpe_idx = sharpe.argmax()
            min_vol_idx = volatility.argmin()
            
            # --- EREDMÉNYEK MEGJELENÍTÉSE ---
            
            st.success("✅ Szimuláció sikeresen lefutott!")
            
            # Két oszlop létrehozása az eredményeknek
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Maximális Sharpe-ráta (Ajánlott)")
                st.metric("Várható Hozam", f"{(returns[max_sharpe_idx]*100):.2f}%")
                st.metric("Volatilitás (Kockázat)", f"{(volatility[max_sharpe_idx]*100):.2f}%")
                st.metric("Sharpe-ráta", f"{sharpe[max_sharpe_idx]:.3f}")
                
                # Súlyok kördiagram
                fig1, ax1 = plt.subplots()
                ax1.pie(weights[max_sharpe_idx], labels=tickers, autopct='%1.1f%%', startangle=90)
                ax1.set_title("Optimális Portfólió Összetétele")
                st.pyplot(fig1)

            with col2:
                st.subheader("🛡️ Minimális Volatilitás (Biztonságos)")
                st.metric("Várható Hozam", f"{(returns[min_vol_idx]*100):.2f}%")
                st.metric("Volatilitás (Kockázat)", f"{(volatility[min_vol_idx]*100):.2f}%")
                st.metric("Sharpe-ráta", f"{sharpe[min_vol_idx]:.3f}")
                
                # Súlyok kördiagram
                fig2, ax2 = plt.subplots()
                ax2.pie(weights[min_vol_idx], labels=tickers, autopct='%1.1f%%', startangle=90)
                ax2.set_title("Legbiztonságosabb Portfólió Összetétele")
                st.pyplot(fig2)
            
            # --- HATÉKONY HATÁR GRAFIKON ---
            st.markdown("---")
            st.subheader("📈 A Hatékony Határ (Efficient Frontier)")
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sc = ax3.scatter(volatility, returns, c=sharpe, cmap='viridis', s=10, alpha=0.5)
            plt.colorbar(sc, label='Sharpe Ratio')
            
            # Kiemelt pontok
            ax3.scatter(volatility[max_sharpe_idx], returns[max_sharpe_idx], marker='*', color='blue', s=300, label='Max Sharpe')
            ax3.scatter(volatility[min_vol_idx], returns[min_vol_idx], marker='*', color='red', s=300, label='Min Volatility')
            
            ax3.set_xlabel('Volatilitás (Kockázat)')
            ax3.set_ylabel('Várható Hozam')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            st.pyplot(fig3)
                      
            st.markdown("---")
            st.subheader("🔥 Korrelációs Hőtérkép")
            st.markdown("Ez a diagram megmutatja, mennyire mozognak együtt az eszközök. Az alacsonyabb (vagy negatív) értékek jobb diverzifikációs lehetőséget jelentenek.")
            
            # Számítsuk ki a korrelációs mátrixot
            corr_matrix = stock_data.pct_change().corr()
            
            # Ábrázolás Seaborn segítségével
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax4)
            ax4.set_title("Eszközök Napi Hozamainak Korrelációja")

            st.pyplot(fig4)
            
            st.markdown("---")
            st.subheader("💰 Kumulált Hozam (Backtesting)")
            st.markdown("Hogyan teljesített volna a Maximális Sharpe-rátájú portfólió az időszak alatt, összehasonlítva az egyes eszközökkel?")
            
            # 1. Hozamok előkészítése
            daily_returns = stock_data.pct_change().dropna()
            
            # 2. Az Optimális Portfólió súlyozott napi hozama
            # (A weights[max_sharpe_idx] a korábbi számításból jön)
            # Figyelem: A weights sorrendjének egyeznie kell az oszlopok sorrendjével!
            opt_portfolio_returns = (daily_returns * weights[max_sharpe_idx]).sum(axis=1)
            
            # 3. Kumulált hozam számítása (1-ből indulunk)
            cumulative_returns = (1 + daily_returns).cumprod()
            cumulative_portfolio = (1 + opt_portfolio_returns).cumprod()
            
            # 4. Ábrázolás
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            
            # Az egyes részvények halványan
            for col in cumulative_returns.columns:
                ax5.plot(cumulative_returns.index, cumulative_returns[col], label=col, alpha=0.3, linestyle='--')
            
            # Az OPTIMÁLIS PORTFÓLIÓ vastagon kiemelve
            ax5.plot(cumulative_portfolio.index, cumulative_portfolio, label='OPTIMÁLIS PORTFÓLIÓ', color='black', linewidth=3)
            
            ax5.set_title("Befektetés Növekedése (1 USD kezdőtőkével)")
            ax5.set_ylabel("Portfólió Értéke")
            ax5.set_xlabel("Dátum")
            ax5.legend()
            ax5.grid(True)
            
            st.pyplot(fig5)
            
            total_return = (cumulative_portfolio.iloc[-1] - 1) * 100
            st.metric(label="Az Optimális Portfólió Teljes Hozama (Időszak alatt)", value=f"{total_return:.2f}%")
            
            # 2. Maximális Visszaesés (Max Drawdown - MDD)
            # Az MDD megmutatja a legnagyobb visszaesést a csúcsponttól (peak-to-trough)
            rolling_max = cumulative_portfolio.cummax()
            drawdown = cumulative_portfolio / rolling_max - 1
            max_drawdown = drawdown.min()
            
            st.metric(label="Maximális Visszaesés (Max Drawdown)", value=f"{max_drawdown*100:.2f}%")


            
else:
    st.info("👈 Állítsd be a paramétereket a bal oldalon, és kattints a 'Szimuláció Futtatása' gombra!")


    
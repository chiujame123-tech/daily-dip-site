# ... (Previous parts of the script remain the same) ...

for sector, tickers in SECTORS.items():
    cards_in_sector = ""
    for t in tickers:
        try:
            df_d = data_daily[t].dropna()
            df_h = data_hourly[t].dropna()
            if len(df_d) < 200: continue
            
            current_price = df_d['Close'].iloc[-1]
            
            # 筛選條件
            sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
            dollar_vol = (df_d['Close'] * df_d['Volume']).rolling(21).mean().iloc[-1] * 21
            combo = pd.DataFrame({'S': df_d['Close'].pct_change(), 'M': spy_ret}).dropna()
            beta = combo['S'].cov(combo['M']) / combo['M'].var() if len(combo)>30 else 0
            
            pass_filter = (current_price > sma200 and dollar_vol > FILTER_MIN_MONTHLY_VOL and beta >= FILTER_MIN_BETA)

            # --- OPTIMIZATION START ---
            # Determine preliminary signal based on price action (without full chart generation yet)
            # We need basic levels to determine signal. 
            # A lightweight calculation or using previous day's levels could be used here if speed is paramount.
            # However, since generate_chart_image does the analysis, we can run it conditionally.
            
            # For this specific code structure where generate_chart_image returns critical levels (tp, sl),
            # we might need to separate calculation from plotting for true optimization.
            # But as a simple step, we can generate the daily chart to get levels, determine signal,
            # and then decide whether to generate the hourly chart and keep the daily chart.
            
            img_d, tp, sl = generate_chart_image(df_d, t, "Daily")
            
            if not img_d: continue # Analysis failed

            s_low, s_high = sl, tp
            range_len = s_high - s_low
            pos_pct = (current_price - s_low) / range_len if range_len > 0 else 0.5
            signal = "LONG" if pos_pct < 0.4 else "WAIT"
            cls = "b-long" if signal == "LONG" else "b-wait"

            # MEMORY OPTIMIZATION:
            # Only keep chart data and generate hourly chart if:
            # 1. Stock passes strict screener filters OR
            # 2. There is an actionable LONG signal
            should_keep_full_data = pass_filter or (signal == "LONG")

            if should_keep_full_data:
                 img_h, _, _ = generate_chart_image(df_h if not df_h.empty else df_d, t, "Hourly")
            else:
                 # Discard image data to save memory for non-interesting stocks
                 img_d = "" 
                 img_h = ""
            # --- OPTIMIZATION END ---
            
            # ... (Rest of logic for AI text, HTML generation, etc.)
            
            # Important: Update the openModal call to handle empty images gracefully
            # You might want to pass a flag or check in JS if img src is empty
            
            # ...

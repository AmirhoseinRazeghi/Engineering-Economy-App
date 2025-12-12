import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt

# --- تنظیمات صفحه و استایل ---
st.set_page_config(page_title="Engineering Economy Analysis", layout="wide", page_icon="💰")

# استایل‌دهی برای راست‌چین کردن متن‌های فارسی و تنظیم فونت
st.markdown("""
<style>
    .main {direction: rtl; text-align: right; font-family: 'Tahoma', sans-serif;}
    h1, h2, h3 {text-align: right;}
    .stDataFrame {direction: ltr;} 
    .stMetric {text-align: right;}
    /* چپ‌چین کردن متن‌های داخل نمودارها و فوتر انگلیسی */
    .footer-text {direction: ltr; text-align: center; font-weight: bold; color: #555;}
</style>
""", unsafe_allow_html=True)

# --- توابع محاسباتی ---

def calculate_metrics(p, n, marr, cf_in, cf_out, s):
    cash_flows = [-p] 
    annual_net = cf_in - cf_out
    for _ in range(n - 1):
        cash_flows.append(annual_net)
    cash_flows.append(annual_net + s)
    
    npw = npf.npv(marr, cash_flows)
    try:
        irr = npf.irr(cash_flows)
    except:
        irr = np.nan

    if marr == 0:
        euaw = npw / n
    else:
        capital_recovery_factor = (marr * (1 + marr)**n) / ((1 + marr)**n - 1)
        euaw = npw * capital_recovery_factor

    discounted_cf = [cf / ((1 + marr)**t) for t, cf in enumerate(cash_flows)]
    cumulative_discounted_cf = np.cumsum(discounted_cf)
    
    dpbp = None
    for t, cum_val in enumerate(cumulative_discounted_cf):
        if cum_val >= 0:
            dpbp = t
            break
            
    return cash_flows, npw, irr, euaw, dpbp

# --- رابط کاربری (UI) ---

st.title("📊 سامانه هوشمند تحلیل اقتصاد مهندسی")
st.markdown("---")

# بخش ورودی‌ها در سایدبار (همچنان فارسی)
with st.sidebar:
    st.header("ورودی‌های پروژه")
    # نام پیش‌فرض را انگلیسی گذاشتم تا در نمودار درست دیده شود
    project_name = st.text_input("نام پروژه (ترجیحاً انگلیسی)", "Factory Project")
    
    st.subheader("پارامترهای مالی")
    p = st.number_input("هزینه اولیه سرمایه‌گذاری (P)", min_value=0.0, value=10000.0, step=1000.0)
    n = st.number_input("عمر پروژه (سال - N)", min_value=1, value=5, step=1)
    marr_percent = st.number_input("نرخ بهره مورد انتظار (MARR %)", min_value=0.0, value=15.0, step=0.5)
    marr = marr_percent / 100.0
    
    st.subheader("جریان‌های نقدی سالانه")
    cf_in = st.number_input("درآمد سالانه (CF in)", min_value=0.0, value=4000.0, step=500.0)
    cf_out = st.number_input("هزینه عملیاتی سالانه (CF out)", min_value=0.0, value=500.0, step=100.0)
    s = st.number_input("ارزش اسقاط در پایان عمر (S)", min_value=0.0, value=2000.0, step=500.0)

    st.markdown("---")
    run_calc = st.button("محاسبه و تحلیل")

if run_calc or True:
    
    cash_flows, npw, irr, euaw, dpbp = calculate_metrics(p, n, marr, cf_in, cf_out, s)
    
    # --- نمایش نتایج خلاصه (KPIs) - فارسی ---
    st.subheader(f"نتیجه ارزیابی: {project_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="ارزش فعلی خالص (NPW)", value=f"{npw:,.0f}", delta="سودده" if npw > 0 else "زیان‌ده")
    with col2:
        irr_display = f"{irr*100:.2f}%" if not np.isnan(irr) else "تعریف نشده"
        st.metric(label="نرخ بازده داخلی (IRR)", value=irr_display, 
                  delta=f"{irr*100 - marr_percent:.2f}% نسبت به MARR" if not np.isnan(irr) else None)
    with col3:
        st.metric(label="ارزش سالانه (EUAW)", value=f"{euaw:,.0f}")
    with col4:
        dpbp_display = f"{dpbp} سال" if dpbp is not None else "بازگشت ندارد"
        st.metric(label="بازگشت سرمایه (DPBP)", value=dpbp_display)

    if npw > 0 and (np.isnan(irr) or irr > marr):
        st.success("✅ **نتیجه:** این پروژه از نظر اقتصادی **توجیه‌پذیر** است.")
    else:
        st.error("❌ **نتیجه:** این پروژه از نظر اقتصادی **رد** می‌شود.")

    st.markdown("---")

    # --- تب‌بندی نمودارها (نمودارها انگلیسی شده‌اند) ---
    tab1, tab2, tab3 = st.tabs(["📉 نمودار جریان نقدی", "🔍 تحلیل حساسیت (Tornado)", "📈 نمودار NPW vs i"])

    with tab1:
        st.subheader("Cash Flow Diagram")
        years = np.arange(0, n + 1)
        colors = ['red' if cf < 0 else 'green' for cf in cash_flows]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(years, cash_flows, color=colors, edgecolor='black')
        ax.axhline(0, color='black', linewidth=1)
        
        # --- ENGLISH LABELS ---
        ax.set_xlabel("Year")
        ax.set_ylabel("Cash Flow ($)")
        ax.set_title(f"Cash Flow Diagram: {project_name}")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', va='bottom' if yval > 0 else 'top', ha='center')
            
        st.pyplot(fig)

    with tab2:
        st.subheader("Sensitivity Analysis (Tornado Chart)")
        
        changes = [-0.20, 0.20]
        base_npw = npw
        sensitivity_data = []
        
        # English Parameter Names
        # 1. Investment
        p_low_npw = calculate_metrics(p * 0.8, n, marr, cf_in, cf_out, s)[1]
        p_high_npw = calculate_metrics(p * 1.2, n, marr, cf_in, cf_out, s)[1]
        sensitivity_data.append({'Parameter': 'Initial Investment (P)', 'Low': p_high_npw, 'High': p_low_npw, 'Range': abs(p_high_npw - p_low_npw)})
        
        # 2. Revenue
        r_low_npw = calculate_metrics(p, n, marr, cf_in * 0.8, cf_out, s)[1]
        r_high_npw = calculate_metrics(p, n, marr, cf_in * 1.2, cf_out, s)[1]
        sensitivity_data.append({'Parameter': 'Annual Revenue', 'Low': r_low_npw, 'High': r_high_npw, 'Range': abs(r_high_npw - r_low_npw)})
        
        # 3. MARR
        m_low_npw = calculate_metrics(p, n, marr * 0.8, cf_in, cf_out, s)[1]
        m_high_npw = calculate_metrics(p, n, marr * 1.2, cf_in, cf_out, s)[1]
        sensitivity_data.append({'Parameter': 'Interest Rate (MARR)', 'Low': m_high_npw, 'High': m_low_npw, 'Range': abs(m_high_npw - m_low_npw)})

        df_sens = pd.DataFrame(sensitivity_data).sort_values(by='Range', ascending=True)
        
        fig_tor, ax_tor = plt.subplots(figsize=(10, 5))
        y_pos = np.arange(len(df_sens))
        
        # --- ENGLISH LABELS ---
        ax_tor.barh(y_pos, df_sens['High'] - base_npw, left=base_npw, color='green', label='+20% Change (Positive Impact)', align='center')
        ax_tor.barh(y_pos, df_sens['Low'] - base_npw, left=base_npw, color='red', label='-20% Change (Negative Impact)', align='center')
        
        ax_tor.set_yticks(y_pos)
        ax_tor.set_yticklabels(df_sens['Parameter'])
        ax_tor.axvline(base_npw, color='black', linestyle='--', label=f'Base NPW: {base_npw:,.0f}')
        ax_tor.set_xlabel('Net Present Worth (NPW)')
        ax_tor.set_title('Tornado Chart: Sensitivity Analysis')
        ax_tor.legend()
        
        st.pyplot(fig_tor)

    with tab3:
        st.subheader("NPW vs Interest Rate Profile")
        
        rates = np.linspace(0, marr * 2.5, 50)
        npw_values = [npf.npv(r, cash_flows) for r in rates]
        
        fig_line, ax_line = plt.subplots(figsize=(10, 5))
        ax_line.plot(rates * 100, npw_values, linewidth=2, color='blue')
        ax_line.axhline(0, color='black', linewidth=1)
        ax_line.axvline(marr_percent, color='red', linestyle='--', label=f'MARR ({marr_percent}%)')
        
        if not np.isnan(irr) and 0 <= irr <= marr * 2.5:
             ax_line.plot(irr * 100, 0, 'ro', label=f'IRR ({irr*100:.1f}%)')
             
        # --- ENGLISH LABELS ---
        ax_line.set_xlabel('Interest Rate (%)')
        ax_line.set_ylabel('Net Present Worth (NPW)')
        ax_line.set_title('NPW Sensitivity to Interest Rate')
        ax_line.legend()
        ax_line.grid(True, alpha=0.3)
        
        st.pyplot(fig_line)

# --- فوتر با نام شما ---
st.markdown("---")
st.markdown("""
<div class="footer-text">
    Designed & Developed by Amirhosein Razeghi
</div>
""", unsafe_allow_html=True)

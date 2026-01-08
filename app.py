import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
import streamlit as st

# Configuration for embedding
st.set_page_config(
    page_title="Solar Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit branding and make it embeddable
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    iframe {border: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Advanced Solar Economics Calculator", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 32px; font-weight: bold; color: #FFD700; text-align: center;}
    .metric-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #FFD700;}
    .stHeader {color: #2c3e50;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: GLOBAL SETTINGS ---
st.sidebar.title("Configuration")
analysis_duration = st.sidebar.slider("Analysis Period (Years)", 5, 25, 25)
currency_symbol = "₹"

# --- HELPER FUNCTIONS ---
def calculate_emi(principal, rate_pa, tenure_months):
    """Calculates EMI for Reducing Balance Method"""
    if rate_pa == 0:
        return principal / tenure_months
    r = rate_pa / (12 * 100)
    return principal * r * ((1 + r)**tenure_months) / (((1 + r)**tenure_months) - 1)

def calculate_flat_emi(principal, rate_pa, tenure_months):
    """Calculates EMI for Flat Rate Method"""
    total_interest = principal * (rate_pa / 100) * (tenure_months / 12)
    total_amount = principal + total_interest
    return total_amount / tenure_months

# --- MAIN APP ---
col1, col2 = st.columns([8 , 1])

with col1:
    st.markdown("## ☀️ Advanced Solar Economics Calculator")
with col2:
    st.image("DWlogo.png",use_container_width=True)



st.divider()

# ==========================================
# SECTION 1: INVESTMENT CALCULATOR
# ==========================================
st.header("1. Investment Calculator")
col1, col2, col3 = st.columns(3)

with col1:
    capacity_kw = st.number_input("Project Capacity (kW)", min_value=1.0, value=10.0, step=0.5)
    price_per_watt = st.number_input(f"Price per Watt ({currency_symbol})", min_value=10.0, value=35.0, step=0.5)

with col2:
    gst_rate = st.number_input("GST/Tax Rate (%)", min_value=0.0, value=13.8, step=0.1)
    is_inclusive = st.checkbox("Price is GST Inclusive", value=False)
    itc_available = st.checkbox("Input Tax Credit (ITC) Available?", value=True, help="Check if the company can claim GST back.")

# Cost Calculation Logic
base_cost = capacity_kw * 1000 * price_per_watt
if is_inclusive:
    project_cost_excl_tax = base_cost / (1 + (gst_rate/100))
    tax_amount = base_cost - project_cost_excl_tax
    total_project_cost = base_cost
else:
    project_cost_excl_tax = base_cost
    tax_amount = base_cost * (gst_rate/100)
    total_project_cost = base_cost + tax_amount

net_invested_capital = total_project_cost - (tax_amount if itc_available else 0)

with col3:
    st.markdown(f"**Total Project Cost:** {currency_symbol}{total_project_cost:,.2f}")
    if itc_available:
        st.success(f"Net CapEx (after ITC): {currency_symbol}{net_invested_capital:,.2f}")

st.subheader("Financing")
use_financing = st.checkbox("Financing Needed?", value=True)

loan_amount = 0
emi_monthly = 0
downpayment = total_project_cost

if use_financing:
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        percent_financed = st.slider("% of Capital Financed", 0, 100, 70)
        loan_amount = total_project_cost * (percent_financed / 100)
        downpayment = total_project_cost - loan_amount
        
    with f_col2:
        tenure_months = st.number_input("Loan Tenure (Months)", 12, 120, 60)
        interest_rate = st.number_input("Interest Rate (P.A %)", 0.0, 20.0, 9.5)
        moratorium_months = st.number_input("Moratorium Period (Months)", 0, 12, 0)

    with f_col3:
        method = st.selectbox("Financing Method", ["EMI (Reducing Balance)", "Flat Rate", "Principal + Interest"])
        
    # Loan Calculations
    if method == "EMI (Reducing Balance)":
        emi_monthly = calculate_emi(loan_amount, interest_rate, tenure_months)
    elif method == "Flat Rate":
        emi_monthly = calculate_flat_emi(loan_amount, interest_rate, tenure_months)
    else:
        # Principal + Interest calc happens in the cashflow loop
        emi_monthly = (loan_amount / tenure_months) # Base principal only for display
        st.caption("*Showing Principal component only. Interest varies monthly.*")

    st.info(f"💰 **Upfront Downpayment:** {currency_symbol}{downpayment:,.2f} | **Est. Monthly EMI:** {currency_symbol}{emi_monthly:,.2f}")

# ==========================================
# SECTION 2: POWER GENERATION
# ==========================================
st.markdown("---")
st.header("2. Power Generation")
gen_col1, gen_col2 = st.columns(2)

with gen_col1:
    gen_mode = st.radio("Calculation Mode", ["Manual CUF", "Location Based (Lat/Long)"])
    
    if gen_mode == "Manual CUF":
        cuf = st.slider("Solar CUF (%)", 12.0, 30.0, 19.0, step=0.1) / 100
        daily_hours = 24 * cuf
        annual_generation = capacity_kw * 24 * 365 * cuf
        st.metric("Avg. Daily Generation (Hrs)", f"{daily_hours:.2f} hrs")
        
    else:
        # Simple Expert Approximation based on Lat (Simulated)
        lat = st.number_input("Latitude", 8.0, 37.0, 28.6) # Default Delhi
        long = st.number_input("Longitude", 68.0, 97.0, 77.2)
        # Physics approximation: Higher Lat = Lower Gen (simplified for demo)
        base_cuf = 0.18
        lat_factor = max(0, (20 - abs(lat - 20)) * 0.002) 
        cuf = base_cuf + lat_factor
        st.success(f"Estimated CUF for Location: {cuf*100:.2f}%")
        annual_generation = capacity_kw * 24 * 365 * cuf

with gen_col2:
    degradation = st.slider("Annual Degradation (%)", 0.0, 2.0, 0.7, step=0.1) / 100
    st.metric("First Year Generation", f"{annual_generation:,.0f} kWh")

# Plot Monthly Gen (Simulated seasonality)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Seasonality curve for India (High in March-May, Low in July-Aug/Dec-Jan)
seasonality = np.array([0.85, 0.95, 1.1, 1.15, 1.2, 1.05, 0.9, 0.85, 0.95, 1.0, 0.95, 0.85])
monthly_gen = (annual_generation / 12) * seasonality
fig_gen = px.bar(x=months, y=monthly_gen, labels={'x':'Month', 'y':'Generation (kWh)'}, title="Estimated Monthly Generation Pattern")
fig_gen.update_layout(height=300)
st.plotly_chart(fig_gen, use_container_width=True)

# ==========================================
# SECTION 3: EARNINGS & SAVINGS
# ==========================================
st.markdown("---")
st.header("3. Earnings & Savings")
sav_col1, sav_col2 = st.columns(2)

with sav_col1:
    tariff = st.number_input("Electricity Tariff (INR/Unit)", 3.0, 20.0, 8.5, step=0.1)
    escalation = st.slider("Tariff Escalation (% p.a.)", 0.0, 10.0, 2.0) / 100

with sav_col2:
    first_year_savings = annual_generation * tariff
    st.metric("1st Year Savings", f"{currency_symbol}{first_year_savings:,.0f}")
    st.caption("Savings calculated based on 100% self-consumption of generated power.")

# ==========================================
# SECTION 4: FINANCIAL PARAMETERS
# ==========================================
st.markdown("---")
st.header("4. Operational Expenses & Depreciation")
fin_col1, fin_col2, fin_col3 = st.columns(3)

with fin_col1:
    st.subheader("O&M Cost")
    om_cost_per_kw = st.number_input("Annual O&M (INR/kW)", 0, 5000, 500)
    annual_om = om_cost_per_kw * capacity_kw
    om_escalation = st.number_input("O&M Escalation (%)", 0.0, 10.0, 5.0) / 100

with fin_col2:
    st.subheader("Insurance")
    insurance_rate = st.number_input("Insurance (% of CapEx)", 0.0, 5.0, 0.3) / 100
    insurance_cost = total_project_cost * insurance_rate

with fin_col3:
    st.subheader("Tax Benefits")
    acc_depreciation = st.checkbox("Claim Accelerated Depreciation?", value=True)
    dep_rate = st.number_input("Depreciation Rate (%)", 0.0, 100.0, 40.0) / 100
    corp_tax_rate = st.number_input("Corporate Tax Rate (%)", 0.0, 50.0, 25.0) / 100

# ==========================================
# SECTION 5: FINAL ANALYSIS & CASH FLOW ENGINE
# ==========================================
st.markdown("---")
st.header("5. Financial Analysis")

# --- CASH FLOW ENGINE ---
years = list(range(1, analysis_duration + 1))
cashflow_data = []

# Loan amortization tracking
remaining_loan = loan_amount
cumulative_depreciation = 0
book_value = net_invested_capital

for year in years:
    # 1. Generation & Revenue
    year_gen = annual_generation * ((1 - degradation) ** (year - 1))
    year_tariff = tariff * ((1 + escalation) ** (year - 1))
    revenue = year_gen * year_tariff
    
    # 2. Expenses
    year_om = annual_om * ((1 + om_escalation) ** (year - 1))
    year_insurance = insurance_cost # Assuming flat insurance on asset value for simplicity or manual escalation
    
    # 3. Debt Service (Annualized for simplicity in yearly view)
    interest_payment = 0
    principal_payment = 0
    
    if use_financing:
        # Check if loan is active in this year
        months_in_year = 12
        start_month = (year - 1) * 12 + 1
        end_month = year * 12
        
        if start_month <= (tenure_months + moratorium_months):
            # Simple annualized handling
            if method == "Principal + Interest":
                # Assuming simple P+I logic annually
                if remaining_loan > 0:
                    principal_payment = loan_amount / (tenure_months/12)
                    interest_payment = remaining_loan * (interest_rate/100)
                    remaining_loan -= principal_payment
            elif method == "Flat Rate":
                 # Flat interest is constant
                 total_annual_pay = emi_monthly * 12
                 interest_payment = loan_amount * (interest_rate/100) # constant
                 principal_payment = total_annual_pay - interest_payment
                 remaining_loan -= principal_payment
            else: # Reducing Balance EMI
                # Complex annual aggregation of monthly reducing balance
                annual_p = 0
                annual_i = 0
                for m in range(12):
                    if remaining_loan <= 0: break
                    interest = remaining_loan * (interest_rate/100/12)
                    principal = emi_monthly - interest
                    annual_i += interest
                    annual_p += principal
                    remaining_loan -= principal
                interest_payment = annual_i
                principal_payment = annual_p
    
    # 4. Depreciation Benefit (Tax Shield)
    tax_savings = 0
    if acc_depreciation:
        # WDV Method usually
        dep_amount = (book_value) * dep_rate
        if book_value > 0:
            tax_savings = dep_amount * corp_tax_rate
            book_value -= dep_amount
    
    # 5. Net Cash Flow
    # Inflow = Revenue + Tax Savings
    # Outflow = O&M + Insurance + Interest + Principal + Tax on Profit(ignored for simplicity, focusing on savings)
    
    # Simplified Solar Cashflow: Savings - Expenses - EMI + Tax Benefits
    total_outflow = year_om + year_insurance + interest_payment + principal_payment
    net_savings = revenue - year_om - year_insurance
    
    # Net Cash Flow to Equity (Post-Finance)
    ncf = net_savings - interest_payment - principal_payment + tax_savings
    
    cashflow_data.append({
        "Year": year,
        "Generation (kWh)": year_gen,
        "Tariff": year_tariff,
        "Gross Savings": revenue,
        "O&M + Ins": year_om + year_insurance,
        "Interest": interest_payment,
        "Principal Repay": principal_payment,
        "Tax Benefit": tax_savings,
        "Net Cash Flow": ncf,
        "Cumulative Cash Flow": 0 # To be calculated
    })

# Convert to DataFrame
df = pd.DataFrame(cashflow_data)
df['Cumulative Cash Flow'] = df['Net Cash Flow'].cumsum() - downpayment

# --- METRICS CALCULATION ---

# NPV
cash_flows = [-downpayment] + df['Net Cash Flow'].tolist()
discount_rate = 0.08 # 8% Discount rate default
npv = npf.npv(discount_rate, cash_flows)

# IRR
try:
    irr = npf.irr(cash_flows) * 100
except:
    irr = 0

# Payback Period
try:
    payback_year = df[df['Cumulative Cash Flow'] >= 0]['Year'].iloc[0]
except:
    payback_year = "25+"

# LCOE Calculation
# LCOE = Total Lifecycle Cost / Total Lifetime Energy
total_lifecycle_cost = downpayment + df['O&M + Ins'].sum() + df['Interest'].sum() + df['Principal Repay'].sum() - df['Tax Benefit'].sum()
total_lifetime_energy = df['Generation (kWh)'].sum()
lcoe = total_lifecycle_cost / total_lifetime_energy

# --- DISPLAY METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Internal Rate of Return (IRR)", f"{irr:.2f}%", help="The expected annual return on investment.")
m2.metric("Net Present Value (NPV)", f"{currency_symbol}{npv:,.0f}", help="Total value added by project in today's money.")
m3.metric("Payback Period", f"{payback_year} Years", help="Time to recover the initial investment.")
m4.metric("LCOE (Cost of Power)", f"{currency_symbol}{lcoe:.2f} / kWh", help="Levelized Cost of Energy over the project life.")

# --- VISUALIZATION ---
tab1, tab2 = st.tabs(["Cash Flow Chart", "Detailed Table"])

with tab1:
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(x=df['Year'], y=df['Gross Savings'], name='Savings', marker_color='green'))
    fig_cf.add_trace(go.Bar(x=df['Year'], y=-df['Principal Repay'], name='Loan Repayment', marker_color='red'))
    fig_cf.add_trace(go.Scatter(x=df['Year'], y=df['Cumulative Cash Flow'], name='Cumulative Cash Flow', mode='lines+markers', line=dict(color='blue', width=3)))
    fig_cf.update_layout(title="Annual Financial Overview", xaxis_title="Year", yaxis_title="Amount (INR)")
    st.plotly_chart(fig_cf, use_container_width=True)

with tab2:
    st.dataframe(df.style.format("{:,.0f}"))

# ==========================================
# SECTION 6: CONCLUSION & EXPERT INSIGHTS
# ==========================================
st.markdown("---")
st.header("6. Expert Conclusion")

roi = (df['Net Cash Flow'].sum() / downpayment) * 100

col_con1, col_con2 = st.columns([2, 1])

with col_con1:
    st.markdown(f"""
    ### 📝 Verdict
    * **Financial Health:** The project generates an IRR of **{irr:.2f}%**, which is {'excellent' if irr > 15 else 'moderate' if irr > 10 else 'low'} compared to standard market returns.
    * **Savings Multiplier:** Over {analysis_duration} years, you will save **{currency_symbol}{df['Gross Savings'].sum():,.0f}** on electricity bills.
    * **Return on Investment:** For every {currency_symbol}1 invested upfront, the project returns **{currency_symbol}{roi/100:.2f}** over its lifetime.
    * **LCOE vs Grid:** Your cost of solar generation is **{currency_symbol}{lcoe:.2f}/unit** vs the Grid Tariff of **{currency_symbol}{tariff:.2f}/unit**. This spread is your profit margin.
    """)

with col_con2:
    st.success("### 🌱 Environmental Impact")
    # 0.7 kg CO2 per kWh (Indian Grid Avg)
    co2_saved = total_lifetime_energy * 0.7 / 1000 # tonnes
    trees_planted = co2_saved * 45 # approx 45 trees absorb 1 ton CO2
    st.write(f"**CO₂ Offset:** {co2_saved:,.1f} Tonnes")
    st.write(f"**Equivalent Trees:** {trees_planted:,.0f} Trees")

st.info("💡 **Expert Tip:** If your Payback Period is under 5 years and IRR is above 15%, this is considered a 'Grade A' solar investment. Ensure you maintain the panels (cleaning) to keep CUF high.")

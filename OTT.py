import streamlit as st
from supabase import create_client
import pandas as pd
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Options Trading Tracker",
    page_icon="🥕🥕🥕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_supabase()


def authenticate_session():
    """Ensure the supabase client is authenticated with the current session"""
    if "session" in st.session_state and st.session_state["session"]:
        try:
            supabase.auth.set_session(
                st.session_state["session"]["access_token"], 
                st.session_state["session"]["refresh_token"]
            )
            return True
        except Exception as e:
            st.error(f"Session authentication failed: {str(e)}")
            clear_session()
            return False
    return False


def clear_session():
    """Clear all session data"""
    keys_to_clear = ["user", "session", "show_new_trade", "edit_trade_id", "show_new_transaction", "edit_transaction_id"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    if "page_mode" not in st.session_state:
        st.session_state.page_mode = "login"


def login_page():
    if "page_mode" not in st.session_state:
        st.session_state.page_mode = "login"

    if st.session_state.page_mode == "login":
        st.title("🥕🥕🥕 Options Trading Tracker - Login")
        
        st.markdown("""
        Welcome to your personal options trading tracker! 
        Keep track of all your options trades and account activity in one secure place.
        """)

        with st.form("login_form"):
            email = st.text_input("✉️ Email", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            login_submit = st.form_submit_button("🚀 Login", use_container_width=True)

            if login_submit and email and password:
                with st.spinner("Logging in..."):
                    try:
                        response = supabase.auth.sign_in_with_password({
                            "email": email, 
                            "password": password
                        })
                        
                        if response.user and response.session:
                            # Store user and session info
                            st.session_state["user"] = response.user
                            st.session_state["session"] = {
                                "access_token": response.session.access_token,
                                "refresh_token": response.session.refresh_token
                            }
                            
                            # Authenticate the client
                            supabase.auth.set_session(
                                response.session.access_token, 
                                response.session.refresh_token
                            )
                            
                            st.success("✅ Login successful!")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Login failed: {str(e)}")

        st.divider()
        st.write("Don't have an account?")
        if st.button("🆕 Create an account", use_container_width=True):
            st.session_state.page_mode = "signup"
            st.rerun()

    elif st.session_state.page_mode == "signup":
        st.title("🆕 Create Your Account")
        
        st.markdown("""
        Join thousands of traders tracking their options performance!
        """)

        with st.form("signup_form"):
            email = st.text_input("✉️ Email", key="signup_email", placeholder="Enter your email")
            password = st.text_input("🔒 Password", type="password", key="signup_pw", 
                                   placeholder="Create a strong password", 
                                   help="Password should be at least 8 characters long")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", 
                                           key="confirm_pw", placeholder="Confirm your password")
            
            signup_submit = st.form_submit_button("🎯 Create Account", use_container_width=True)

            if signup_submit and email and password:
                if password != confirm_password:
                    st.error("❌ Passwords don't match!")
                elif len(password) < 8:
                    st.error("❌ Password must be at least 8 characters long!")
                else:
                    with st.spinner("Creating your account..."):
                        try:
                            response = supabase.auth.sign_up({
                                "email": email, 
                                "password": password
                            })
                            st.success("✅ Account created! Please check your email for verification, then log in.")
                            st.session_state.page_mode = "login"
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to create account: {str(e)}")

        st.divider()
        if st.button("⬅️ Back to Login", use_container_width=True):
            st.session_state.page_mode = "login"
            st.rerun()


def trade_form(trade_data=None, is_edit=False):
    """Reusable form for adding/editing trades"""
    form_key = "edit_trade_form" if is_edit else "new_trade_form"
    submit_text = "💾 Update Trade" if is_edit else "💾 Save Trade"
    
    with st.form(form_key):
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("📈 Ticker Symbol", 
                                 value=trade_data.get("ticker", "") if trade_data else "",
                                 placeholder="e.g., AAPL, TSLA, SPY")
            action = st.selectbox("📊 Action", ["BUY", "SELL"], 
                                index=0 if not trade_data else (0 if trade_data.get("action") == "BUY" else 1))
            option_type = st.selectbox("🎯 Option Type", ["CALL", "PUT"],
                                     index=0 if not trade_data else (0 if trade_data.get("option_type") == "CALL" else 1))
            strike = st.number_input("💰 Strike Price", step=0.5, min_value=0.0,
                                   value=float(trade_data.get("strike_price", 0)) if trade_data else 0.0)
        
        with col2:
            premium = st.number_input("💵 Premium per Contract", step=0.01, min_value=0.0,
                                    value=float(trade_data.get("premium", 0)) if trade_data else 0.0)
            contracts = st.number_input("📋 Number of Contracts", step=1, min_value=1,
                                      value=int(trade_data.get("contracts", 1)) if trade_data else 1)
            commission = st.number_input("💸 Commission", step=0.01, min_value=0.0,
                           value=float(trade_data.get("commission", 0.0)) if trade_data else 0.0)

            # Handle expiry date
            expiry_value = date.today()
            if trade_data and trade_data.get("expiry"):
                try:
                    if isinstance(trade_data["expiry"], str):
                        expiry_value = datetime.strptime(trade_data["expiry"], "%Y-%m-%d").date()
                    else:
                        expiry_value = trade_data["expiry"]
                except:
                    expiry_value = date.today()
            
            expiry = st.date_input("📅 Expiry Date", value=expiry_value)
        
        # Calculate total cost/credit
        total = premium * contracts * 100  # Options are typically multiplied by 100
        action_text = "Cost" if action == "BUY" else "Credit"
        st.info(f"💡 Total {action_text}: ${total:,.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(submit_text, use_container_width=True)
        with col2:
            cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)
        
        if cancelled:
            if is_edit and "edit_trade_id" in st.session_state:
                del st.session_state["edit_trade_id"]
            if "show_new_trade" in st.session_state:
                st.session_state.show_new_trade = False
            st.rerun()
        
        if submitted and ticker:
            return {
                "ticker": ticker.upper(),
                "action": action,
                "option_type": option_type,
                "strike_price": strike,
                "premium": premium,
                "contracts": contracts,
                "commission": commission,
                "expiry": expiry.isoformat()
            }
    
    return None


def transaction_form(transaction_data=None, is_edit=False):
    """Reusable form for adding/editing account transactions"""
    form_key = "edit_transaction_form" if is_edit else "new_transaction_form"
    submit_text = "💾 Update Transaction" if is_edit else "💾 Save Transaction"
    
    with st.form(form_key):
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_type = st.selectbox("📋 Transaction Type", 
                                          ["DEPOSIT", "WITHDRAWAL", "PROFIT"], 
                                          index=0 if not transaction_data else 
                                          ["DEPOSIT", "WITHDRAWAL", "PROFIT"].index(transaction_data.get("transaction_type", "DEPOSIT")))
            
            amount = st.number_input("💵 Amount", step=0.01, min_value=0.01,
                                   value=float(transaction_data.get("amount", 0.01)) if transaction_data else 0.01)

            # ✅ Add date picker
            created_value = date.today()
            if transaction_data and transaction_data.get("created_at"):
                try:
                    created_value = pd.to_datetime(transaction_data["created_at"]).date()
                except:
                    created_value = date.today()
            created_at = st.date_input("📅 Transaction Date", value=created_value)
        
        with col2:
            description = st.text_area("📝 Description (Optional)", 
                                     value=transaction_data.get("description", "") if transaction_data else "",
                                     placeholder="e.g., Initial deposit, Profit from SPY calls, etc.")
        
        # Show impact on account balance
        if transaction_type == "DEPOSIT":
            st.success(f"✅ This will add ${amount:,.2f} to your account")
        elif transaction_type == "WITHDRAWAL":
            st.warning(f"⚠️ This will subtract ${amount:,.2f} from your account")
        elif transaction_type == "PROFIT":
            st.info(f"📈 This will record ${amount:,.2f} as trading profit")
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(submit_text, use_container_width=True)
        with col2:
            cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)
        
        if cancelled:
            if is_edit and "edit_transaction_id" in st.session_state:
                del st.session_state["edit_transaction_id"]
            if "show_new_transaction" in st.session_state:
                st.session_state.show_new_transaction = False
            st.rerun()
        
        if submitted and amount > 0:
            return {
                "transaction_type": transaction_type,
                "amount": amount,
                "description": description.strip() if description and description.strip() else None,
                "created_at": created_at.isoformat()  # ✅ Save picked date
            }
    
    return None



def display_trades_table(trades_data):
    """Display trades in a filterable dataframe with inline action buttons"""
    if not trades_data:
        st.info("📭 No trades recorded yet. Add your first trade to get started!")
        return
    
    # Convert to dataframe and format
    df = pd.DataFrame(trades_data)
    
    # Format data for display
    df['strike_price'] = df['strike_price'].astype(float)
    df['premium'] = df['premium'].astype(float)
    df['contracts'] = df['contracts'].astype(int)
    df['total_value'] = (df['premium'] * df['contracts'] * 100) - df['commission']
    df['expiry'] = pd.to_datetime(df['expiry']).dt.date
    
    # Add filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ticker_filter = st.multiselect("Filter by Ticker", options=sorted(df['ticker'].unique()), default=[], key="ticker_filter")
    with col2:
        action_filter = st.multiselect("Filter by Action", options=['BUY', 'SELL'], default=[], key="action_filter")
    with col3:
        type_filter = st.multiselect("Filter by Type", options=['CALL', 'PUT'], default=[], key="type_filter")
    with col4:
        min_date = df['expiry'].min()
        max_date = df['expiry'].max()
        date_range = st.date_input("Expiry Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="date_filter")
    
    # Apply filters
    filtered_df = df.copy()
    if ticker_filter:
        filtered_df = filtered_df[filtered_df['ticker'].isin(ticker_filter)]
    if action_filter:
        filtered_df = filtered_df[filtered_df['action'].isin(action_filter)]
    if type_filter:
        filtered_df = filtered_df[filtered_df['option_type'].isin(type_filter)]
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['expiry'] >= start_date) & (filtered_df['expiry'] <= end_date)]
    
    st.write(f"**Showing {len(filtered_df)} of {len(df)} trades**")
    
    if not filtered_df.empty:
        # Header row
        header_cols = st.columns([1.2, 0.8, 0.8, 1, 1, 0.8, 1.2, 1.1, 1.1, 0.6, 0.6])
        headers = ["Ticker", "Action", "Type", "Strike", "Premium", "Contracts", "Commission", "Total Value", "Expiry", "Edit", "Delete"]
        for col, label in zip(header_cols, headers):
            col.markdown(f"**{label}**")

        st.markdown("---")

        # Entries
        for idx, (_, trade) in enumerate(filtered_df.iterrows()):
            cols = st.columns([1.2, 0.8, 0.8, 1, 1, 0.8, 1.2, 1.1, 1.1, 0.6, 0.6])
            
            cols[0].write(f"**{trade['ticker']}**")
            cols[1].markdown("BUY" if trade['action']=="BUY" else "SELL")
            cols[2].markdown("CALL" if trade['option_type']=="CALL" else "PUT")
            cols[3].write(f"${trade['strike_price']:.2f}")
            cols[4].write(f"${trade['premium']:.2f}")
            cols[5].write(str(trade['contracts']))
            cols[6].write(f"${trade['commission']:.2f}")
            cols[7].write(f"${trade['total_value']:,.2f}")
            cols[8].write(str(trade['expiry']))
            
            if cols[9].button("✏️", key=f"edit_{trade['id']}", help=f"Edit {trade['ticker']} trade"):
                st.session_state["edit_trade_id"] = trade['id']
                st.session_state["show_new_trade"] = False
                st.rerun()
            
            if cols[10].button("🗑️", key=f"delete_{trade['id']}", help=f"Delete {trade['ticker']} trade"):
                try:
                    supabase.table("trades").delete().eq("id", trade['id']).execute()
                    st.success("✅ Trade deleted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to delete trade: {str(e)}")
            
            if idx < len(filtered_df) - 1:
                st.markdown("---")
    else:
        st.info("No trades match the selected filters.")


def display_transactions_table(transactions_data):
    """Display account transactions in a filterable table"""
    if not transactions_data:
        st.info("💰 No account transactions recorded yet. Add your first transaction to get started!")
        return
    
    # Convert to dataframe and format
    df = pd.DataFrame(transactions_data)
    df['amount'] = df['amount'].astype(float)
    df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601').dt.date    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        type_filter = st.multiselect("Filter by Type", options=['DEPOSIT', 'WITHDRAWAL', 'PROFIT'], default=[], key="transaction_type_filter")
    with col2:
        if not df.empty:
            min_date = df['created_at'].min()
            max_date = df['created_at'].max()
            date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="transaction_date_filter")
    
    # Apply filters
    filtered_df = df.copy()
    if type_filter:
        filtered_df = filtered_df[filtered_df['transaction_type'].isin(type_filter)]
    if 'date_range' in locals() and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['created_at'] >= start_date) & (filtered_df['created_at'] <= end_date)]
    
    st.write(f"**Showing {len(filtered_df)} of {len(df)} transactions**")
    
    if not filtered_df.empty:
        # Header row
        header_cols = st.columns([1.5, 1.5, 2, 1.5, 0.6, 0.6])
        headers = ["Date", "Type", "Description", "Amount", "Edit", "Delete"]
        for col, label in zip(header_cols, headers):
            col.markdown(f"**{label}**")

        st.markdown("---")

        # Entries
        for idx, (_, transaction) in enumerate(filtered_df.iterrows()):
            cols = st.columns([1.5, 1.5, 2, 1.5, 0.6, 0.6])
            
            cols[0].write(str(transaction['created_at']))
            
            # Color-code transaction types
            trans_type = transaction['transaction_type']
            if trans_type == "DEPOSIT":
                cols[1].markdown("🟢 DEPOSIT")
            elif trans_type == "WITHDRAWAL":
                cols[1].markdown("🔴 WITHDRAWAL")
            else:  # PROFIT
                cols[1].markdown("🔵 PROFIT")
            
            cols[2].write(transaction.get('description', '-'))
            
            # Format amount with appropriate sign
            amount = transaction['amount']
            if trans_type == "WITHDRAWAL":
                cols[3].write(f"-${amount:,.2f}")
            else:
                cols[3].write(f"+${amount:,.2f}")
            
            if cols[4].button("✏️", key=f"edit_transaction_{transaction['id']}", help="Edit transaction"):
                st.session_state["edit_transaction_id"] = transaction['id']
                st.session_state["show_new_transaction"] = False
                st.rerun()
            
            if cols[5].button("🗑️", key=f"delete_transaction_{transaction['id']}", help="Delete transaction"):
                try:
                    supabase.table("account_transactions").delete().eq("id", transaction['id']).execute()
                    st.success("✅ Transaction deleted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to delete transaction: {str(e)}")
            
            if idx < len(filtered_df) - 1:
                st.markdown("---")
    else:
        st.info("No transactions match the selected filters.")


def calculate_account_summary(trades_data, transactions_data):
    """Calculate comprehensive account summary"""
    # Calculate from trades
    trade_cost = 0
    trade_credit = 0
    if trades_data:
        for trade in trades_data:
            value = float(trade['premium']) * int(trade['contracts']) * 100
            commission = float(trade.get('commission', 0))
            if trade['action'] == 'BUY':
                trade_cost += value + commission  
            else:
                trade_credit += value - commission

    
    net_trade_premium = trade_credit - trade_cost
    
    # Calculate from transactions
    deposits = sum(float(t['amount']) for t in transactions_data if t['transaction_type'] == 'DEPOSIT')
    withdrawals = sum(float(t['amount']) for t in transactions_data if t['transaction_type'] == 'WITHDRAWAL')
    recorded_profits = sum(float(t['amount']) for t in transactions_data if t['transaction_type'] == 'PROFIT')
    
    net_cash_flow = deposits - withdrawals
    account_balance = net_cash_flow + net_trade_premium + recorded_profits
    
    return {
        'deposits': deposits,
        'withdrawals': withdrawals,
        'net_cash_flow': net_cash_flow,
        'trade_cost': trade_cost,
        'trade_credit': trade_credit,
        'net_trade_premium': net_trade_premium,
        'recorded_profits': recorded_profits,
        'account_balance': account_balance
    }

def build_balance_timeline(trades_data, transactions_data):
    """Reconstruct account balance over time (using trade expiry dates)"""
    events = []

    # Add transactions as events (still use created_at, since no expiry concept)
    for t in transactions_data:
        amount = float(t['amount'])
        if t['transaction_type'] == "WITHDRAWAL":
            amount = -amount
        elif t['transaction_type'] == "PROFIT":
            amount = amount  # keep as is
        events.append({
            "date": pd.to_datetime(t['created_at']).date(),
            "amount": amount,
            "source": "transaction"
        })

    # Add trades as events (using expiry instead of created_at)
    for trade in trades_data:
        value = float(trade['premium']) * int(trade['contracts']) * 100
        commission = float(trade.get('commission', 0))
        if trade['action'] == "BUY":
            net = -(value + commission)
        else:  # SELL
            net = (value - commission)
        events.append({
            "date": pd.to_datetime(trade['expiry']).date(),  # ✅ use expiry here
            "amount": net,
            "source": "trade"
        })

    if not events:
        return pd.DataFrame()

    # Build DataFrame
    df = pd.DataFrame(events)
    df = df.groupby("date")["amount"].sum().reset_index()
    df = df.sort_values("date")

    # Cumulative balance
    df["balance"] = df["amount"].cumsum()
    return df


def main_dashboard():
    """Main trading dashboard with tabs"""
    # Ensure authentication
    if not authenticate_session():
        clear_session()
        st.rerun()
        return
    
    # Header with user info and logout
    col1, col2, col3 = st.columns([6, 2, 1])
    with col1:
        st.title("🥕🥕🥕 Options Trading Tracker")
        if "user" in st.session_state:
            st.caption(f"👤 Logged in as: {st.session_state['user'].email}")
    
    with col3:
        if st.button("➜] Logout", use_container_width=True):
            try:
                supabase.auth.sign_out()
            except:
                pass
            clear_session()
            st.rerun()

    user_id = st.session_state["user"].id

    # Load data
    try:
        trades_response = supabase.table("trades").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        trades_data = trades_response.data
        
        transactions_response = supabase.table("account_transactions").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        transactions_data = transactions_response.data
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        trades_data = []
        transactions_data = []

    # Calculate account summary
    summary = calculate_account_summary(trades_data, transactions_data)

    # Account Summary
    st.write("### 📋 Account Summary (USD)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Account Balance", f"${summary['account_balance']:,.2f}")
    with col2:
        st.metric("Deposits", f"${summary['deposits']:,.2f}")
    with col3:
        st.metric("Withdrawals", f"${summary['withdrawals']:,.2f}")
    with col4:
        st.metric("Net Trade Premium", f"${summary['net_trade_premium']:,.2f}")
    with col5:
        st.metric("Recorded Profits", f"${summary['recorded_profits']:,.2f}")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Trades", "💰 Account Transactions", "📊 Analytics"])

    with tab1:
        # Initialize session state for trades
        if "show_new_trade" not in st.session_state:
            st.session_state.show_new_trade = False

        # Header with Add button
        header_col1, header_col2 = st.columns([6, 1])
        with header_col1:
            st.write("### 📈 Your Trades")
        with header_col2:
            if st.button("➕ Add Trade", use_container_width=True, key="add_trade_btn"):
                st.session_state.show_new_trade = True
                if "edit_trade_id" in st.session_state:
                    del st.session_state["edit_trade_id"]
                st.rerun()

        # Show add form
        if st.session_state.show_new_trade:
            st.write("#### ➕ New Trade")
            new_trade_data = trade_form()
            if new_trade_data:
                try:
                    new_trade_data["user_id"] = user_id
                    supabase.table("trades").insert(new_trade_data).execute()
                    st.success("✅ Trade saved successfully!")
                    st.session_state.show_new_trade = False
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save trade: {str(e)}")

        # Edit trade section
        if "edit_trade_id" in st.session_state:
            st.write("#### ✏️ Edit Trade")
            trade_to_edit = next((t for t in trades_data if t['id'] == st.session_state["edit_trade_id"]), None)
            if trade_to_edit:
                updated_trade_data = trade_form(trade_to_edit, is_edit=True)
                if updated_trade_data:
                    try:
                        supabase.table("trades").update(updated_trade_data).eq("id", st.session_state["edit_trade_id"]).execute()
                        st.success("✅ Trade updated successfully!")
                        del st.session_state["edit_trade_id"]
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to update trade: {str(e)}")

        # Display trades table
        display_trades_table(trades_data)

    with tab2:
        # Initialize session state for transactions
        if "show_new_transaction" not in st.session_state:
            st.session_state.show_new_transaction = False

        # Header with Add button
        header_col1, header_col2 = st.columns([6, 1])
        with header_col1:
            st.write("### 💰 Account Transactions")
        with header_col2:
            if st.button("➕ Add Transaction", use_container_width=True, key="add_transaction_btn"):
                st.session_state.show_new_transaction = True
                if "edit_transaction_id" in st.session_state:
                    del st.session_state["edit_transaction_id"]
                st.rerun()

        # Show add form
        if st.session_state.show_new_transaction:
            st.write("#### ➕ New Transaction")
            new_transaction_data = transaction_form()
            if new_transaction_data:
                try:
                    new_transaction_data["user_id"] = user_id
                    supabase.table("account_transactions").insert(new_transaction_data).execute()
                    st.success("✅ Transaction saved successfully!")
                    st.session_state.show_new_transaction = False
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save transaction: {str(e)}")

        # Edit transaction section
        if "edit_transaction_id" in st.session_state:
            st.write("#### ✏️ Edit Transaction")
            transaction_to_edit = next((t for t in transactions_data if t['id'] == st.session_state["edit_transaction_id"]), None)
            if transaction_to_edit:
                updated_transaction_data = transaction_form(transaction_to_edit, is_edit=True)
                if updated_transaction_data:
                    try:
                        supabase.table("account_transactions").update(updated_transaction_data).eq("id", st.session_state["edit_transaction_id"]).execute()
                        st.success("✅ Transaction updated successfully!")
                        del st.session_state["edit_transaction_id"]
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to update transaction: {str(e)}")

        # Display transactions table
        display_transactions_table(transactions_data)

    with tab3:
        st.write("### 📊 Analytics & Performance")
        
        if not trades_data and not transactions_data:
            st.info("📈 Start trading and recording transactions to see analytics!")
            return

        # Date Range Selector
        date_filter_option = st.radio(
            "Time Period",
            ["Inception", "YTD", "1Y", "Custom"],
            horizontal=True
        )

        # Default: full inception range
        start_date = pd.to_datetime("1900-01-01").date()
        end_date = pd.to_datetime("today").date()

        if trades_data or transactions_data:
            all_dates = []
            if trades_data:
                all_dates.extend([pd.to_datetime(t['expiry']).date() for t in trades_data])
            if transactions_data:
                all_dates.extend([pd.to_datetime(t['created_at']).date() for t in transactions_data])
            if all_dates:
                start_date = min(all_dates)
                end_date = max(all_dates)

        # Apply filter logic
        today = date.today()
        if date_filter_option == "YTD":
            start_date = date(today.year, 1, 1)
            end_date = today
        elif date_filter_option == "1Y":
            start_date = today.replace(year=today.year - 1)
            end_date = today
        elif date_filter_option == "Custom":
            start_date, end_date = st.date_input(
                "Select custom date range",
                value=(start_date, end_date),
                min_value=start_date,
                max_value=end_date
            )

        # Filter trades & transactions by chosen range
        filtered_trades = [
            t for t in trades_data 
            if start_date <= pd.to_datetime(t['expiry']).date() <= end_date
        ]
        filtered_transactions = [
            tr for tr in transactions_data
            if start_date <= pd.to_datetime(tr['created_at']).date() <= end_date
        ]
        st.caption(f"Showing data from {start_date} to {end_date}")

        # ✅ Recalculate summary for filtered data
        filtered_summary = calculate_account_summary(filtered_trades, filtered_transactions)
            
        # Performance metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_trades = len(filtered_trades)
        buy_trades = len([t for t in filtered_trades if t['action'] == 'BUY'])
        sell_trades = len([t for t in filtered_trades if t['action'] == 'SELL'])
        with col1:
            st.metric("Total Trades", total_trades)
            st.metric("Total Deposits", f"${filtered_summary['deposits']:,.2f}")
        with col2:
            st.metric("Buy Trades", buy_trades)
            st.metric("Total Withdrawals", f"${filtered_summary['withdrawals']:,.2f}")
        with col3:
            st.metric("Sell Trades", sell_trades)
            st.metric("Net Cash Flow", f"${filtered_summary['net_cash_flow']:,.2f}")
        with col4:
            st.metric("Trade Cost", f"${filtered_summary['trade_cost']:,.2f}")
            st.metric("Recorded Profits", f"${filtered_summary['recorded_profits']:,.2f}")
        with col5:
            st.metric("Trade Credit", f"${filtered_summary['trade_credit']:,.2f}")
            if filtered_summary['deposits'] > 0:
                roi = ((filtered_summary['account_balance'] - filtered_summary['deposits']) / 
                       filtered_summary['deposits']) * 100
                st.metric("ROI", f"{roi:.2f}%")


        # Charts and visualizations
        if filtered_trades:
            st.write("#### 📊 Cash Flow Breakdown")

            # Collect contributions
            flow_data = []

            # Trades
            for t in filtered_trades:
                value = float(t['premium']) * int(t['contracts']) * 100
                commission = float(t.get('commission', 0))
                if t['action'] == "BUY":
                    net = -(value + commission)
                else:  # SELL
                    net = (value - commission)
                label = f"{t['action']} {t['option_type']}"
                flow_data.append({"category": label, "amount": net})

            # Transactions
            for tr in filtered_transactions:
                amt = float(tr['amount'])
                if tr['transaction_type'] == "WITHDRAWAL":
                    amt = -amt
                flow_data.append({"category": tr['transaction_type'], "amount": amt})

            df_flow = pd.DataFrame(flow_data)

            if not df_flow.empty:
                agg = df_flow.groupby("category")["amount"].sum().reset_index()

                # --- Pie chart (absolute contribution %) ---
                fig_pie = go.Figure(
                    data=[go.Pie(
                        labels=agg["category"],
                        values=agg["amount"].abs(),  # absolute for percentages
                        textinfo="label+percent",
                        hovertemplate="%{label}: %{value:$,.2f}<extra></extra>"
                    )]
                )
                fig_pie.update_layout(title="Cash Flow Contribution by Category")
                st.plotly_chart(fig_pie, use_container_width=True)

        st.write("#### 📈 Account Balance Over Time")

        df_balance = build_balance_timeline(filtered_trades, filtered_transactions)

        if df_balance.empty:
            st.info("No data available to plot account balance.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_balance['date'], 
                y=df_balance['balance'],
                mode='lines+markers',
                name='Account Balance'
            ))
            fig.update_layout(
                title="Account Balance Over Time",
                xaxis_title="Date",
                yaxis_title="Balance ($)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)



def main():
    # Check if we have a valid session
    if "session" in st.session_state and "user" in st.session_state:
        if authenticate_session():
            main_dashboard()
        else:
            clear_session()
            login_page()
    else:
        login_page()


if __name__ == "__main__":
    main()
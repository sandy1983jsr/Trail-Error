import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from itertools import permutations
import io
import json

# Page configuration
st.set_page_config(
    page_title="Pharma Production Optimizer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e8f4f8 0%, #b8e0f0 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'product_details' not in st.session_state:
    st.session_state.product_details = None
if 'changeover_data' not in st.session_state:
    st.session_state.changeover_data = None
if 'production_schedule' not in st.session_state:
    st.session_state.production_schedule = None
if 'emission_factors' not in st.session_state:
    st.session_state.emission_factors = {
        'electricity': 0.7,  # tCO2e per kWh
        'steam': 0.5  # tCO2e per kg
    }
if 'calculation_results' not in st.session_state:
    st.session_state.calculation_results = None
if 'optimized_schedule' not in st.session_state:
    st.session_state.optimized_schedule = None


class PharmaOptimizer:
    """Main optimizer class for pharma production scheduling"""
    
    def __init__(self, product_details, changeover_data, emission_factors):
        self.product_details = product_details
        self.changeover_data = changeover_data
        self.emission_factors = emission_factors
        
    def calculate_batch_emissions(self, product_name, num_batches=1):
        """Calculate emissions for a given number of batches of a product"""
        
        # Filter product details
        product_data = self.product_details[
            self.product_details['Product Name'].str.strip() == product_name.strip()
        ]
        
        if product_data.empty:
            return None
        
        # Calculate electricity emissions
        total_electricity = product_data['Electricity consumption'].sum() * num_batches
        electricity_emissions = total_electricity * self.emission_factors['electricity']
        
        # Calculate steam emissions
        total_steam = product_data['Total steam consumption'].sum() * num_batches
        steam_emissions = total_steam * self.emission_factors['steam']
        
        # Total emissions
        total_emissions = electricity_emissions + steam_emissions
        
        # Calculate total time
        total_machine_hours = product_data['Machine Hrs'].sum() * num_batches
        
        return {
            'product': product_name,
            'batches': num_batches,
            'total_electricity_kwh': total_electricity,
            'total_steam_kg': total_steam,
            'electricity_emissions_tco2e': electricity_emissions,
            'steam_emissions_tco2e': steam_emissions,
            'total_emissions_tco2e': total_emissions,
            'total_machine_hours': total_machine_hours
        }
    
    def calculate_changeover_time(self, from_product, to_product, same_product=False):
        """Calculate changeover time between products"""
        
        if same_product:
            # Batch-to-batch cleaning for same product
            changeover_data = self.changeover_data[
                self.changeover_data['Product name'].str.strip() == from_product.strip()
            ]
            if not changeover_data.empty:
                return changeover_data['Batch-to-batch clean'].sum()
            return 0
        else:
            # Product changeover cleaning
            changeover_data = self.changeover_data[
                self.changeover_data['Product name'].str.strip() == to_product.strip()
            ]
            if not changeover_data.empty:
                # Handle both 'Product Cleaning Tim' and 'Product Cleaning Time' column names
                if 'Product Cleaning Time' in changeover_data.columns:
                    return changeover_data['Product Cleaning Time'].sum()
                elif 'Product Cleaning Tim' in changeover_data.columns:
                    return changeover_data['Product Cleaning Tim'].sum()
            return 0
    
    def calculate_schedule_emissions(self, schedule_df):
        """Calculate total emissions for a production schedule"""
        
        results = []
        total_emissions = 0
        total_time = 0
        cumulative_time = 0
        
        prev_product = None
        
        for idx, row in schedule_df.iterrows():
            product = row['Product']
            batches = row['Batches']
            
            # Calculate production emissions
            batch_result = self.calculate_batch_emissions(product, batches)
            
            if batch_result:
                # Calculate changeover time
                if prev_product is None:
                    changeover_time = 0
                elif prev_product == product:
                    changeover_time = self.calculate_changeover_time(prev_product, product, same_product=True)
                else:
                    changeover_time = self.calculate_changeover_time(prev_product, product, same_product=False)
                
                batch_result['changeover_time_hours'] = changeover_time
                batch_result['sequence_order'] = idx + 1
                batch_result['start_time_hours'] = cumulative_time
                
                cumulative_time += batch_result['total_machine_hours'] + changeover_time
                batch_result['end_time_hours'] = cumulative_time
                
                results.append(batch_result)
                total_emissions += batch_result['total_emissions_tco2e']
                total_time += batch_result['total_machine_hours'] + changeover_time
                
                prev_product = product
        
        results_df = pd.DataFrame(results)
        
        summary = {
            'total_emissions_tco2e': total_emissions,
            'total_time_hours': total_time,
            'total_electricity_kwh': results_df['total_electricity_kwh'].sum() if not results_df.empty else 0,
            'total_steam_kg': results_df['total_steam_kg'].sum() if not results_df.empty else 0,
            'total_changeover_time': results_df['changeover_time_hours'].sum() if not results_df.empty else 0,
            'number_of_batches': results_df['batches'].sum() if not results_df.empty else 0
        }
        
        return results_df, summary
    
    def optimize_schedule(self, schedule_df, max_iterations=1000):
        """Optimize production schedule to minimize GHG emissions"""
        
        # For large schedules, use heuristic approach
        # For small schedules, try different permutations
        
        if len(schedule_df) <= 8:
            return self._optimize_by_permutation(schedule_df)
        else:
            return self._optimize_by_heuristic(schedule_df, max_iterations)
    
    def _optimize_by_permutation(self, schedule_df):
        """Optimize by trying all permutations (for small schedules)"""
        
        products = schedule_df['Product'].tolist()
        batches = schedule_df['Batches'].tolist()
        
        best_emissions = float('inf')
        best_order = None
        best_results = None
        
        # Try all permutations
        for perm in permutations(range(len(products))):
            test_schedule = pd.DataFrame({
                'Product': [products[i] for i in perm],
                'Batches': [batches[i] for i in perm]
            })
            
            results_df, summary = self.calculate_schedule_emissions(test_schedule)
            
            if summary['total_emissions_tco2e'] < best_emissions:
                best_emissions = summary['total_emissions_tco2e']
                best_order = perm
                best_results = (results_df, summary)
        
        optimized_schedule = pd.DataFrame({
            'Product': [products[i] for i in best_order],
            'Batches': [batches[i] for i in best_order]
        })
        
        return optimized_schedule, best_results[0], best_results[1]
    
    def _optimize_by_heuristic(self, schedule_df, max_iterations):
        """Optimize using heuristic approach (for larger schedules)"""
        
        # Group same products together to minimize changeovers
        schedule_df_sorted = schedule_df.sort_values('Product').reset_index(drop=True)
        
        # Calculate emissions for this arrangement
        results_df, summary = self.calculate_schedule_emissions(schedule_df_sorted)
        
        best_schedule = schedule_df_sorted.copy()
        best_emissions = summary['total_emissions_tco2e']
        best_results = (results_df, summary)
        
        # Try random swaps to improve
        for _ in range(max_iterations):
            test_schedule = best_schedule.copy()
            
            # Random swap
            idx1, idx2 = np.random.choice(len(test_schedule), 2, replace=False)
            test_schedule.iloc[idx1], test_schedule.iloc[idx2] = test_schedule.iloc[idx2].copy(), test_schedule.iloc[idx1].copy()
            
            results_df, summary = self.calculate_schedule_emissions(test_schedule)
            
            if summary['total_emissions_tco2e'] < best_emissions:
                best_emissions = summary['total_emissions_tco2e']
                best_schedule = test_schedule.copy()
                best_results = (results_df, summary)
        
        return best_schedule, best_results[0], best_results[1]


# Main App
def main():
    st.markdown('<div class="main-header">🏭 Pharma Production Schedule Optimizer</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/pharmaceutical-industry.png", width=80)
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["📊 Dashboard", "📁 Data Upload", "⚙️ Configuration", "🔬 Calculate Emissions", "🎯 Optimize Schedule"]
        )
        
        st.markdown("---")
        st.info("**About:** This tool optimizes pharmaceutical production schedules to minimize GHG emissions.")
    
    # Page routing
    if page == "📁 Data Upload":
        data_upload_page()
    elif page == "⚙️ Configuration":
        configuration_page()
    elif page == "🔬 Calculate Emissions":
        calculate_emissions_page()
    elif page == "🎯 Optimize Schedule":
        optimize_schedule_page()
    else:
        dashboard_page()


def dashboard_page():
    """Main dashboard overview"""
    
    st.header("📊 Dashboard Overview")
    
    # Check data status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.product_details is not None:
            st.success("✅ Product Details Loaded")
            st.metric("Products", len(st.session_state.product_details['Product Name'].unique()))
        else:
            st.warning("⚠️ Product Details Not Loaded")
    
    with col2:
        if st.session_state.changeover_data is not None:
            st.success("✅ Changeover Data Loaded")
            st.metric("Changeover Records", len(st.session_state.changeover_data))
        else:
            st.warning("⚠️ Changeover Data Not Loaded")
    
    with col3:
        if st.session_state.production_schedule is not None:
            st.success("✅ Production Schedule Loaded")
            st.metric("Scheduled Items", len(st.session_state.production_schedule))
        else:
            st.warning("⚠️ Production Schedule Not Loaded")
    
    st.markdown("---")
    
    # Display results if available
    if st.session_state.calculation_results:
        st.subheader("📈 Latest Calculation Results")
        
        results_df = st.session_state.calculation_results['details']
        summary = st.session_state.calculation_results['summary']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total GHG Emissions",
                f"{summary['total_emissions_tco2e']:.2f} tCO2e",
                delta=None
            )
        
        with col2:
            st.metric(
                "Total Production Time",
                f"{summary['total_time_hours']:.1f} hrs",
                delta=None
            )
        
        with col3:
            st.metric(
                "Electricity Consumed",
                f"{summary['total_electricity_kwh']:.1f} kWh",
                delta=None
            )
        
        with col4:
            st.metric(
                "Steam Consumed",
                f"{summary['total_steam_kg']:.1f} kg",
                delta=None
            )
        
        # Visualization
        st.subheader("📊 Emissions Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Emissions by product
            fig = px.bar(
                results_df,
                x='product',
                y='total_emissions_tco2e',
                title='GHG Emissions by Product',
                labels={'product': 'Product', 'total_emissions_tco2e': 'Emissions (tCO2e)'},
                color='total_emissions_tco2e',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Emissions breakdown
            emission_breakdown = pd.DataFrame({
                'Source': ['Electricity', 'Steam'],
                'Emissions (tCO2e)': [
                    results_df['electricity_emissions_tco2e'].sum(),
                    results_df['steam_emissions_tco2e'].sum()
                ]
            })
            
            fig = px.pie(
                emission_breakdown,
                values='Emissions (tCO2e)',
                names='Source',
                title='Emissions by Source',
                color_discrete_sequence=['#ff7f0e', '#1f77b4']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline
        st.subheader("⏱️ Production Timeline")
        
        fig = go.Figure()
        
        for idx, row in results_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['end_time_hours'] - row['start_time_hours']],
                y=[row['product']],
                orientation='h',
                name=f"{row['product']} (Batch {row['batches']})",
                text=f"{row['total_emissions_tco2e']:.2f} tCO2e",
                textposition='inside',
                base=[row['start_time_hours']],
                hovertemplate=f"<b>{row['product']}</b><br>Start: {row['start_time_hours']:.1f}h<br>End: {row['end_time_hours']:.1f}h<br>Emissions: {row['total_emissions_tco2e']:.2f} tCO2e"
            ))
        
        fig.update_layout(
            title='Production Schedule Timeline',
            xaxis_title='Time (hours)',
            yaxis_title='Product',
            showlegend=False,
            height=max(400, len(results_df) * 40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with optimized schedule
        if st.session_state.optimized_schedule:
            st.subheader("🎯 Optimization Results")
            
            opt_summary = st.session_state.optimized_schedule['summary']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                reduction = summary['total_emissions_tco2e'] - opt_summary['total_emissions_tco2e']
                reduction_pct = (reduction / summary['total_emissions_tco2e']) * 100
                st.metric(
                    "Emission Reduction",
                    f"{reduction:.2f} tCO2e",
                    delta=f"{reduction_pct:.1f}%",
                    delta_color="inverse"
                )
            
            with col2:
                time_reduction = summary['total_time_hours'] - opt_summary['total_time_hours']
                st.metric(
                    "Time Saved",
                    f"{time_reduction:.1f} hrs",
                    delta=f"{(time_reduction/summary['total_time_hours']*100):.1f}%",
                    delta_color="normal"
                )
            
            with col3:
                changeover_reduction = summary['total_changeover_time'] - opt_summary['total_changeover_time']
                st.metric(
                    "Changeover Time Reduction",
                    f"{changeover_reduction:.1f} hrs",
                    delta=f"{(changeover_reduction/summary['total_changeover_time']*100):.1f}%" if summary['total_changeover_time'] > 0 else "N/A"
                )
    else:
        st.info("👈 Upload data and calculate emissions to see results here!")
        
        # Quick start guide
        st.subheader("🚀 Quick Start Guide")
        
        st.markdown("""
        1. **📁 Data Upload**: Upload your product details, changeover data, and production schedule
        2. **⚙️ Configuration**: Set emission factors for electricity and steam
        3. **🔬 Calculate Emissions**: Calculate GHG emissions for your current schedule
        4. **🎯 Optimize Schedule**: Find the optimal production sequence to minimize emissions
        """)


def data_upload_page():
    """Data upload interface"""
    
    st.header("📁 Data Upload")
    
    st.markdown("""
    Upload the required data files to get started. You can upload Excel files containing:
    - **Product Details**: Product information, processes, equipment, and resource consumption
    - **Changeover Data**: Cleaning times for batch-to-batch and product changeovers
    - **Production Schedule**: Monthly production schedule with products and batch quantities
    """)
    
    st.markdown("---")
    
    # Product Details Upload
    st.subheader("1️⃣ Product Details")
    
    product_file = st.file_uploader(
        "Upload Product Details Excel File",
        type=['xlsx', 'xls'],
        key='product_upload',
        help="File should contain: Product Name, Base Qty, UoM, Operation Description, Equipment, Machine Hrs, Electricity consumption, Machine Hrs for Steam, Total steam consumption"
    )
    
    if product_file:
        try:
            df = pd.read_excel(product_file)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace('\xa0', ' ')
            
            # Standardize column names
            column_mapping = {
                'Product Name': 'Product Name',
                'Base Qty': 'Base Qty',
                'UoM': 'UoM',
                'Operation Descriptio': 'Operation Description',
                'Equipment': 'Equipment',
                'Machine Hrs': 'Machine Hrs',
                'Electricity consumption': 'Electricity consumption',
                'Machine Hrs for Steam': 'Machine Hrs for Steam',
                'Total steam consumption': 'Total steam consumption'
            }
            
            df = df.rename(columns=column_mapping)
            df['Product Name'] = df['Product Name'].str.strip()
            
            st.session_state.product_details = df
            st.success(f"✅ Product details loaded successfully! ({len(df)} records)")
            
            with st.expander("Preview Data"):
                st.dataframe(df.head(20), use_container_width=True)
                
                st.write(f"**Total Products:** {df['Product Name'].nunique()}")
                st.write(f"**Total Records:** {len(df)}")
                
        except Exception as e:
            st.error(f"Error loading product details: {str(e)}")
    
    st.markdown("---")
    
    # Changeover Data Upload
    st.subheader("2️⃣ Changeover Data")
    
    changeover_file = st.file_uploader(
        "Upload Changeover Data Excel File",
        type=['xlsx', 'xls'],
        key='changeover_upload',
        help="File should contain: Product name, Operation Description, Batch-to-batch clean, Product Cleaning Time"
    )
    
    if changeover_file:
        try:
            df = pd.read_excel(changeover_file)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace('\xa0', ' ')
            
            # Standardize column names
            column_mapping = {
                'Product name': 'Product name',
                'Operation Descriptio': 'Operation Description',
                'Batch-to-batch clean': 'Batch-to-batch clean',
                'Product Cleaning Tim': 'Product Cleaning Time'
            }
            
            df = df.rename(columns=column_mapping)
            df['Product name'] = df['Product name'].str.strip()
            
            st.session_state.changeover_data = df
            st.success(f"✅ Changeover data loaded successfully! ({len(df)} records)")
            
            with st.expander("Preview Data"):
                st.dataframe(df.head(20), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading changeover data: {str(e)}")
    
    st.markdown("---")
    
    # Production Schedule Upload
    st.subheader("3️⃣ Production Schedule")
    
    st.info("📝 **Schedule Format:** Your Excel file should have columns: 'Product' and 'Batches'")
    
    schedule_file = st.file_uploader(
        "Upload Production Schedule Excel File",
        type=['xlsx', 'xls'],
        key='schedule_upload',
        help="File should contain: Product, Batches"
    )
    
    if schedule_file:
        try:
            df = pd.read_excel(schedule_file)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace('\xa0', ' ')
            df['Product'] = df['Product'].str.strip()
            
            st.session_state.production_schedule = df
            st.success(f"✅ Production schedule loaded successfully! ({len(df)} items)")
            
            with st.expander("Preview Schedule"):
                st.dataframe(df, use_container_width=True)
                
                st.write(f"**Total Items:** {len(df)}")
                st.write(f"**Total Batches:** {df['Batches'].sum()}")
                
        except Exception as e:
            st.error(f"Error loading production schedule: {str(e)}")
    
    # Manual Schedule Creation
    st.markdown("---")
    st.subheader("Or Create Schedule Manually")
    
    if st.session_state.product_details is not None:
        available_products = sorted(st.session_state.product_details['Product Name'].unique())
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_product = st.selectbox("Select Product", available_products)
        
        with col2:
            num_batches = st.number_input("Batches", min_value=1, value=1, step=1)
        
        if st.button("Add to Schedule"):
            if st.session_state.production_schedule is None:
                st.session_state.production_schedule = pd.DataFrame(columns=['Product', 'Batches'])
            
            new_row = pd.DataFrame({'Product': [selected_product], 'Batches': [num_batches]})
            st.session_state.production_schedule = pd.concat([st.session_state.production_schedule, new_row], ignore_index=True)
            st.success(f"Added {selected_product} ({num_batches} batches) to schedule!")
            st.rerun()
        
        if st.session_state.production_schedule is not None and len(st.session_state.production_schedule) > 0:
            st.write("**Current Schedule:**")
            st.dataframe(st.session_state.production_schedule, use_container_width=True)
            
            if st.button("Clear Schedule"):
                st.session_state.production_schedule = None
                st.rerun()
    else:
        st.warning("Please upload product details first to create a manual schedule.")
    
    # Download template
    st.markdown("---")
    st.subheader("📥 Download Template")
    
    template_df = pd.DataFrame({
        'Product': ['Product A', 'Product B', 'Product C'],
        'Batches': [2, 1, 3]
    })
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Schedule')
    
    st.download_button(
        label="Download Schedule Template",
        data=buffer.getvalue(),
        file_name="production_schedule_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def configuration_page():
    """Configuration settings"""
    
    st.header("⚙️ Configuration")
    
    st.markdown("""
    Configure emission factors and other parameters for GHG calculations.
    """)
    
    st.markdown("---")
    
    st.subheader("🌍 Emission Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        electricity_factor = st.number_input(
            "Electricity Emission Factor (tCO2e per kWh)",
            min_value=0.0,
            value=st.session_state.emission_factors['electricity'],
            step=0.01,
            format="%.3f",
            help="Default: 0.7 tCO2e per kWh"
        )
    
    with col2:
        steam_factor = st.number_input(
            "Steam Emission Factor (tCO2e per kg)",
            min_value=0.0,
            value=st.session_state.emission_factors['steam'],
            step=0.01,
            format="%.3f",
            help="Default: 0.5 tCO2e per kg"
        )
    
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.emission_factors = {
            'electricity': electricity_factor,
            'steam': steam_factor
        }
        st.success("✅ Configuration saved successfully!")
    
    st.markdown("---")
    
    st.subheader("📊 Current Configuration")
    
    config_df = pd.DataFrame({
        'Parameter': ['Electricity Emission Factor', 'Steam Emission Factor'],
        'Value': [
            f"{st.session_state.emission_factors['electricity']:.3f} tCO2e/kWh",
            f"{st.session_state.emission_factors['steam']:.3f} tCO2e/kg"
        ]
    })
    
    st.table(config_df)
    
    # Reset to defaults
    if st.button("🔄 Reset to Defaults"):
        st.session_state.emission_factors = {
            'electricity': 0.7,
            'steam': 0.5
        }
        st.success("Configuration reset to default values!")
        st.rerun()


def calculate_emissions_page():
    """Calculate emissions for current schedule"""
    
    st.header("🔬 Calculate GHG Emissions")
    
    # Check if all required data is loaded
    if st.session_state.product_details is None:
        st.error("❌ Product details not loaded. Please upload in the Data Upload page.")
        return
    
    if st.session_state.changeover_data is None:
        st.error("❌ Changeover data not loaded. Please upload in the Data Upload page.")
        return
    
    if st.session_state.production_schedule is None:
        st.error("❌ Production schedule not loaded. Please upload or create in the Data Upload page.")
        return
    
    st.success("✅ All required data loaded!")
    
    st.markdown("---")
    
    # Display current schedule
    st.subheader("📋 Current Production Schedule")
    st.dataframe(st.session_state.production_schedule, use_container_width=True)
    
    st.markdown("---")
    
    # Calculate button
    if st.button("🚀 Calculate Emissions", type="primary"):
        with st.spinner("Calculating emissions..."):
            try:
                # Initialize optimizer
                optimizer = PharmaOptimizer(
                    st.session_state.product_details,
                    st.session_state.changeover_data,
                    st.session_state.emission_factors
                )
                
                # Calculate emissions
                results_df, summary = optimizer.calculate_schedule_emissions(
                    st.session_state.production_schedule
                )
                
                # Store results
                st.session_state.calculation_results = {
                    'details': results_df,
                    'summary': summary,
                    'timestamp': datetime.now()
                }
                
                st.success("✅ Emissions calculated successfully!")
                
            except Exception as e:
                st.error(f"Error calculating emissions: {str(e)}")
                return
    
    # Display results
    if st.session_state.calculation_results:
        results_df = st.session_state.calculation_results['details']
        summary = st.session_state.calculation_results['summary']
        
        st.markdown("---")
        st.subheader("📊 Results Summary")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total GHG Emissions",
                f"{summary['total_emissions_tco2e']:.2f}",
                delta=None,
                help="Total greenhouse gas emissions in tonnes of CO2 equivalent"
            )
            st.caption("tCO2e")
        
        with col2:
            st.metric(
                "Production Time",
                f"{summary['total_time_hours']:.1f}",
                delta=None,
                help="Total production time including changeovers"
            )
            st.caption("hours")
        
        with col3:
            st.metric(
                "Electricity",
                f"{summary['total_electricity_kwh']:.1f}",
                delta=None,
                help="Total electricity consumption"
            )
            st.caption("kWh")
        
        with col4:
            st.metric(
                "Steam",
                f"{summary['total_steam_kg']:.1f}",
                delta=None,
                help="Total steam consumption"
            )
            st.caption("kg")
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Batches", int(summary['number_of_batches']))
        
        with col2:
            st.metric("Changeover Time", f"{summary['total_changeover_time']:.1f} hrs")
        
        with col3:
            avg_emissions = summary['total_emissions_tco2e'] / summary['number_of_batches']
            st.metric("Avg Emissions per Batch", f"{avg_emissions:.2f} tCO2e")
        
        st.markdown("---")
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        display_df = results_df[[
            'sequence_order', 'product', 'batches', 
            'total_emissions_tco2e', 'electricity_emissions_tco2e', 'steam_emissions_tco2e',
            'total_machine_hours', 'changeover_time_hours'
        ]].copy()
        
        display_df.columns = [
            'Order', 'Product', 'Batches',
            'Total Emissions (tCO2e)', 'Electricity Emissions', 'Steam Emissions',
            'Production Hours', 'Changeover Hours'
        ]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Emissions Breakdown", "Resource Consumption", "Timeline"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Emissions by product
                fig = px.bar(
                    results_df,
                    x='product',
                    y='total_emissions_tco2e',
                    title='GHG Emissions by Product',
                    labels={'product': 'Product', 'total_emissions_tco2e': 'Emissions (tCO2e)'},
                    color='total_emissions_tco2e',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Electricity vs Steam emissions
                emission_data = results_df[['product', 'electricity_emissions_tco2e', 'steam_emissions_tco2e']].copy()
                emission_data_melted = emission_data.melt(
                    id_vars='product',
                    value_vars=['electricity_emissions_tco2e', 'steam_emissions_tco2e'],
                    var_name='Source',
                    value_name='Emissions'
                )
                emission_data_melted['Source'] = emission_data_melted['Source'].map({
                    'electricity_emissions_tco2e': 'Electricity',
                    'steam_emissions_tco2e': 'Steam'
                })
                
                fig = px.bar(
                    emission_data_melted,
                    x='product',
                    y='Emissions',
                    color='Source',
                    title='Emissions by Source',
                    labels={'product': 'Product', 'Emissions': 'Emissions (tCO2e)'},
                    barmode='stack',
                    color_discrete_map={'Electricity': '#ff7f0e', 'Steam': '#1f77b4'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Electricity consumption
                fig = px.bar(
                    results_df,
                    x='product',
                    y='total_electricity_kwh',
                    title='Electricity Consumption by Product',
                    labels={'product': 'Product', 'total_electricity_kwh': 'Electricity (kWh)'},
                    color='total_electricity_kwh',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Steam consumption
                fig = px.bar(
                    results_df,
                    x='product',
                    y='total_steam_kg',
                    title='Steam Consumption by Product',
                    labels={'product': 'Product', 'total_steam_kg': 'Steam (kg)'},
                    color='total_steam_kg',
                    color_continuous_scale='Oranges'
                )
                fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Timeline Gantt chart
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set3
            
            for idx, row in results_df.iterrows():
                color_idx = idx % len(colors)
                
                fig.add_trace(go.Bar(
                    x=[row['end_time_hours'] - row['start_time_hours']],
                    y=[f"{row['product']} (x{int(row['batches'])})"],
                    orientation='h',
                    name=row['product'],
                    text=f"{row['total_emissions_tco2e']:.2f} tCO2e",
                    textposition='inside',
                    base=[row['start_time_hours']],
                    marker_color=colors[color_idx],
                    hovertemplate=(
                        f"<b>{row['product']}</b><br>"
                        f"Batches: {int(row['batches'])}<br>"
                        f"Start: {row['start_time_hours']:.1f}h<br>"
                        f"End: {row['end_time_hours']:.1f}h<br>"
                        f"Duration: {row['total_machine_hours']:.1f}h<br>"
                        f"Changeover: {row['changeover_time_hours']:.1f}h<br>"
                        f"Emissions: {row['total_emissions_tco2e']:.2f} tCO2e<br>"
                        "<extra></extra>"
                    )
                ))
            
            fig.update_layout(
                title='Production Schedule Timeline',
                xaxis_title='Time (hours)',
                yaxis_title='Product',
                showlegend=False,
                height=max(400, len(results_df) * 50),
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Export results
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export to Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Detailed Results')
                pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name='Summary')
            
            st.download_button(
                label="📥 Download Excel Report",
                data=buffer.getvalue(),
                file_name=f"emissions_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # Export to CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"emissions_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def optimize_schedule_page():
    """Optimize production schedule"""
    
    st.header("🎯 Optimize Production Schedule")
    
    st.markdown("""
    This tool will optimize your production schedule to minimize GHG emissions by:
    - Grouping similar products together to reduce changeover times
    - Sequencing batches to minimize cleaning requirements
    - Finding the optimal production order through intelligent algorithms
    """)
    
    # Check if all required data is loaded
    if st.session_state.product_details is None:
        st.error("❌ Product details not loaded. Please upload in the Data Upload page.")
        return
    
    if st.session_state.changeover_data is None:
        st.error("❌ Changeover data not loaded. Please upload in the Data Upload page.")
        return
    
    if st.session_state.production_schedule is None:
        st.error("❌ Production schedule not loaded. Please upload or create in the Data Upload page.")
        return
    
    if st.session_state.calculation_results is None:
        st.warning("⚠️ Please calculate emissions for the current schedule first.")
        if st.button("Go to Calculate Emissions"):
            st.session_state.page = "🔬 Calculate Emissions"
            st.rerun()
        return
    
    st.success("✅ All required data loaded!")
    
    st.markdown("---")
    
    # Display current schedule performance
    st.subheader("📊 Current Schedule Performance")
    
    current_summary = st.session_state.calculation_results['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Emissions", f"{current_summary['total_emissions_tco2e']:.2f} tCO2e")
    
    with col2:
        st.metric("Total Time", f"{current_summary['total_time_hours']:.1f} hrs")
    
    with col3:
        st.metric("Changeover Time", f"{current_summary['total_changeover_time']:.1f} hrs")
    
    with col4:
        st.metric("Total Batches", int(current_summary['number_of_batches']))
    
    st.markdown("---")
    
    # Optimization settings
    st.subheader("⚙️ Optimization Settings")
    
    schedule_size = len(st.session_state.production_schedule)
    
    if schedule_size <= 8:
        st.info("🔍 **Optimization Method:** Exhaustive search (trying all possible sequences)")
        max_iterations = None
    else:
        st.info("🔍 **Optimization Method:** Heuristic optimization (intelligent random search)")
        max_iterations = st.slider(
            "Maximum Iterations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="More iterations may find better solutions but take longer"
        )
    
    st.markdown("---")
    
    # Optimize button
    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Optimizing schedule... This may take a moment."):
            try:
                # Initialize optimizer
                optimizer = PharmaOptimizer(
                    st.session_state.product_details,
                    st.session_state.changeover_data,
                    st.session_state.emission_factors
                )
                
                # Run optimization
                if schedule_size <= 8:
                    optimized_schedule, results_df, summary = optimizer.optimize_schedule(
                        st.session_state.production_schedule
                    )
                else:
                    optimized_schedule, results_df, summary = optimizer.optimize_schedule(
                        st.session_state.production_schedule,
                        max_iterations=max_iterations
                    )
                
                # Store optimized results
                st.session_state.optimized_schedule = {
                    'schedule': optimized_schedule,
                    'details': results_df,
                    'summary': summary,
                    'timestamp': datetime.now()
                }
                
                st.success("✅ Optimization completed!")
                
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                return
    
    # Display optimization results
    if st.session_state.optimized_schedule:
        opt_schedule = st.session_state.optimized_schedule['schedule']
        opt_results = st.session_state.optimized_schedule['details']
        opt_summary = st.session_state.optimized_schedule['summary']
        
        st.markdown("---")
        st.subheader("✨ Optimization Results")
        
        # Comparison metrics
        st.write("### 📈 Improvements")
        
        col1, col2, col3, col4 = st.columns(4)
        
        emission_reduction = current_summary['total_emissions_tco2e'] - opt_summary['total_emissions_tco2e']
        emission_reduction_pct = (emission_reduction / current_summary['total_emissions_tco2e']) * 100
        
        time_reduction = current_summary['total_time_hours'] - opt_summary['total_time_hours']
        time_reduction_pct = (time_reduction / current_summary['total_time_hours']) * 100
        
        changeover_reduction = current_summary['total_changeover_time'] - opt_summary['total_changeover_time']
        changeover_reduction_pct = (changeover_reduction / current_summary['total_changeover_time']) * 100 if current_summary['total_changeover_time'] > 0 else 0
        
        with col1:
            st.metric(
                "Emission Reduction",
                f"{opt_summary['total_emissions_tco2e']:.2f} tCO2e",
                delta=f"-{emission_reduction:.2f} ({emission_reduction_pct:.1f}%)",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Total Time",
                f"{opt_summary['total_time_hours']:.1f} hrs",
                delta=f"-{time_reduction:.1f} ({time_reduction_pct:.1f}%)" if time_reduction > 0 else f"+{abs(time_reduction):.1f}",
                delta_color="normal" if time_reduction > 0 else "inverse"
            )
        
        with col3:
            st.metric(
                "Changeover Time",
                f"{opt_summary['total_changeover_time']:.1f} hrs",
                delta=f"-{changeover_reduction:.1f} ({changeover_reduction_pct:.1f}%)" if changeover_reduction > 0 else "No change"
            )
        
        with col4:
            electricity_reduction = current_summary['total_electricity_kwh'] - opt_summary['total_electricity_kwh']
            st.metric(
                "Electricity Saved",
                f"{electricity_reduction:.1f} kWh",
                delta=None
            )
        
        # Highlight box for significant improvement
        if emission_reduction_pct > 5:
            st.success(f"🎉 Great! The optimized schedule reduces emissions by {emission_reduction_pct:.1f}%, saving {emission_reduction:.2f} tCO2e!")
        elif emission_reduction_pct > 0:
            st.info(f"✓ The optimized schedule provides a modest improvement of {emission_reduction_pct:.1f}%.")
        else:
            st.warning("The current schedule is already well-optimized. Minimal improvement found.")
        
        st.markdown("---")
        
        # Side-by-side comparison
        st.subheader("🔄 Schedule Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Schedule**")
            current_display = st.session_state.production_schedule.copy()
            current_display['Order'] = range(1, len(current_display) + 1)
            current_display = current_display[['Order', 'Product', 'Batches']]
            st.dataframe(current_display, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Optimized Schedule**")
            opt_display = opt_schedule.copy()
            opt_display['Order'] = range(1, len(opt_display) + 1)
            opt_display = opt_display[['Order', 'Product', 'Batches']]
            st.dataframe(opt_display, use_container_width=True, hide_index=True)
        
        # Detailed comparison
        st.markdown("---")
        st.subheader("📊 Detailed Comparison")
        
        tab1, tab2, tab3 = st.tabs(["Emissions Comparison", "Timeline Comparison", "Resource Comparison"])
        
        with tab1:
            # Emissions comparison chart
            current_results = st.session_state.calculation_results['details']
            
            comparison_data = []
            for _, row in current_results.iterrows():
                comparison_data.append({
                    'Product': row['product'],
                    'Schedule': 'Current',
                    'Emissions': row['total_emissions_tco2e']
                })
            
            for _, row in opt_results.iterrows():
                comparison_data.append({
                    'Product': row['product'],
                    'Schedule': 'Optimized',
                    'Emissions': row['total_emissions_tco2e']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comparison_df,
                x='Product',
                y='Emissions',
                color='Schedule',
                barmode='group',
                title='Emissions Comparison by Product',
                labels={'Emissions': 'Emissions (tCO2e)'},
                color_discrete_map={'Current': '#ff7f0e', 'Optimized': '#2ca02c'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Timeline comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Timeline**")
                fig = go.Figure()
                
                for idx, row in current_results.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row['end_time_hours'] - row['start_time_hours']],
                        y=[row['product']],
                        orientation='h',
                        name=row['product'],
                        text=f"{row['total_emissions_tco2e']:.1f}",
                        textposition='inside',
                        base=[row['start_time_hours']],
                        showlegend=False
                    ))
                
                fig.update_layout(
                    xaxis_title='Time (hours)',
                    height=max(300, len(current_results) * 40),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Optimized Timeline**")
                fig = go.Figure()
                
                for idx, row in opt_results.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row['end_time_hours'] - row['start_time_hours']],
                        y=[row['product']],
                        orientation='h',
                        name=row['product'],
                        text=f"{row['total_emissions_tco2e']:.1f}",
                        textposition='inside',
                        base=[row['start_time_hours']],
                        showlegend=False
                    ))
                
                fig.update_layout(
                    xaxis_title='Time (hours)',
                    height=max(300, len(opt_results) * 40),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Resource comparison
            col1, col2 = st.columns(2)
            
            with col1:
                resource_comparison = pd.DataFrame({
                    'Schedule': ['Current', 'Optimized'],
                    'Electricity (kWh)': [
                        current_summary['total_electricity_kwh'],
                        opt_summary['total_electricity_kwh']
                    ],
                    'Steam (kg)': [
                        current_summary['total_steam_kg'],
                        opt_summary['total_steam_kg']
                    ]
                })
                
                fig = px.bar(
                    resource_comparison,
                    x='Schedule',
                    y='Electricity (kWh)',
                    title='Electricity Consumption',
                    color='Schedule',
                    color_discrete_map={'Current': '#ff7f0e', 'Optimized': '#2ca02c'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    resource_comparison,
                    x='Schedule',
                    y='Steam (kg)',
                    title='Steam Consumption',
                    color='Schedule',
                    color_discrete_map={'Current': '#ff7f0e', 'Optimized': '#2ca02c'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Export optimized schedule
        st.markdown("---")
        st.subheader("💾 Export Optimized Schedule")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export optimized schedule
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                opt_schedule.to_excel(writer, index=False, sheet_name='Optimized Schedule')
                opt_results.to_excel(writer, index=False, sheet_name='Detailed Results')
                
                comparison = pd.DataFrame({
                    'Metric': ['Total Emissions (tCO2e)', 'Total Time (hrs)', 'Changeover Time (hrs)', 
                              'Electricity (kWh)', 'Steam (kg)'],
                    'Current': [
                        current_summary['total_emissions_tco2e'],
                        current_summary['total_time_hours'],
                        current_summary['total_changeover_time'],
                        current_summary['total_electricity_kwh'],
                        current_summary['total_steam_kg']
                    ],
                    'Optimized': [
                        opt_summary['total_emissions_tco2e'],
                        opt_summary['total_time_hours'],
                        opt_summary['total_changeover_time'],
                        opt_summary['total_electricity_kwh'],
                        opt_summary['total_steam_kg']
                    ],
                    'Improvement': [
                        f"{emission_reduction_pct:.1f}%",
                        f"{time_reduction_pct:.1f}%",
                        f"{changeover_reduction_pct:.1f}%",
                        f"{(electricity_reduction/current_summary['total_electricity_kwh']*100):.1f}%",
                        f"{((current_summary['total_steam_kg']-opt_summary['total_steam_kg'])/current_summary['total_steam_kg']*100):.1f}%"
                    ]
                })
                comparison.to_excel(writer, index=False, sheet_name='Comparison')
            
            st.download_button(
                label="📥 Download Optimized Schedule (Excel)",
                data=buffer.getvalue(),
                file_name=f"optimized_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # Apply optimized schedule
            if st.button("✅ Apply Optimized Schedule", type="primary"):
                st.session_state.production_schedule = opt_schedule.copy()
                st.session_state.calculation_results = {
                    'details': opt_results,
                    'summary': opt_summary,
                    'timestamp': datetime.now()
                }
                st.success("✅ Optimized schedule has been applied as the current schedule!")
                st.balloons()


if __name__ == "__main__":
    main()

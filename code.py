import pandas as pd
import numpy as np

df_emissions = pd.read_csv('/Users/kunwuxue/Desktop/ECO225/archive/GCB2022v27_MtCO2_flat.csv')
df_energy = pd.read_excel('/Users/kunwuxue/Desktop/ECO225/archive/owid-energy-data.xlsx')
df_loss = pd.read_excel('/Users/kunwuxue/Desktop/ECO225/archive/API_EG.ELC.LOSS.ZS_DS2_en_excel_v2_22.xlsm')

df_emissions = df_emissions.rename(columns={'ISO 3166-1 alpha-3': 'iso_code', 'Year': 'year'})
year_cols = [str(y) for y in range(1960, 2025) if str(y) in df_loss.columns]
cols_to_melt = ['Country Code'] + year_cols
df_loss_clean = df_loss[cols_to_melt].melt(
    id_vars=['Country Code'], 
    var_name='year', 
    value_name='transmission_loss_pct'
)

df_loss_clean = df_loss_clean.rename(columns={'Country Code': 'iso_code'})
df_loss_clean['year'] = pd.to_numeric(df_loss_clean['year'])

merged_df = pd.merge(
    df_emissions,
    df_energy,
    on=['iso_code', 'year'],
    how='inner',
)
merged_df = pd.merge(
    merged_df,
    df_loss_clean,
    on=['iso_code', 'year'],
    how='left'
)
cols_to_keep = [
    'iso_code', 'year', 'country',                
    'Total',                  
    'Coal', 'Gas',                    
    'electricity_generation',      
    'hydro_electricity',      
    'solar_electricity',      
    'wind_electricity',       
    'carbon_intensity_elec',
    'gdp',                        
    'primary_energy_consumption', 
    'population',
    'transmission_loss_pct'
]
valid_cols = [c for c in cols_to_keep if c in merged_df.columns]
df_clean = merged_df[valid_cols].copy()
df_clean = df_clean.dropna(subset=['iso_code'])

df_clean = df_clean.rename(columns={
    'Total': 'co2_total',
    'Coal': 'co2_coal',
    'Gas': 'co2_gas',
    'electricity_generation': 'total_generation_twh',
    'hydro_electricity': 'hydro_generation_twh',
    'solar_electricity': 'solar_generation_twh',
    'wind_electricity': 'wind_generation_twh',
    'carbon_intensity_elec': 'carbon_intensity',
    'primary_energy_consumption': 'total_energy_twh'
}) 

developed_isos = [
    'USA', 'CAN', 'GBR', 'FRA', 'DEU', 'ITA', 'JPN', 'AUS', 'NZL', 
    'AUT', 'BEL', 'CHE', 'DNK', 'ESP', 'FIN', 'GRC', 'HKG', 'IRL', 
    'ISL', 'ISR', 'KOR', 'LUX', 'NLD', 'NOR', 'PRT', 'SGP', 'SWE', 'TWN'
]

def classify_country(iso):
    if iso in developed_isos:
        return 'Developed'
    return 'Developing'

df_clean['status'] = df_clean['iso_code'].apply(classify_country)

df_final = df_clean[(df_clean['year'] >= 2000) & (df_clean['year'] <= 2021)].copy()

if 'gdp' in df_final.columns:
    df_final['co2_per_gdp_kg'] = (df_final['co2_total'] * 1e9) / df_final['gdp']

if 'total_energy_twh' in df_final.columns:
    df_final['electricity_share_energy'] = (df_final['total_generation_twh'] / df_final['total_energy_twh']) * 100
df_final['electricity_share_energy'] = df_final['electricity_share_energy'].replace([np.inf, -np.inf], np.nan)

df_final['hydro_share_pct'] = (df_final['hydro_generation_twh'] / df_final['total_generation_twh']) * 100
df_final['solar_generation_twh'] = df_final['solar_generation_twh'].fillna(0)
df_final['wind_generation_twh'] = df_final['wind_generation_twh'].fillna(0)

df_final['solar_wind_share_pct'] = (
    (df_final['solar_generation_twh'] + df_final['wind_generation_twh']) / df_final['total_generation_twh']
) * 100
df_final['gdp_per_capita'] = df_final['gdp'] / df_final['population']

df_final = df_final[df_final['carbon_intensity'] > 0]
df_final.to_csv('/Users/kunwuxue/Desktop/ECO225/archive/my_merged_data.csv', index=False)

from IPython.display import display, HTML

cols_to_keep = [
    'carbon_intensity', 
    'co2_per_gdp_kg',
    'hydro_share_pct', 
    'transmission_loss_pct', 
    'gdp_per_capita', 
    'electricity_share_energy', 
    'solar_wind_share_pct'
]

labels = {
    'carbon_intensity': 'Grid Carbon Intensity (gCO2/kWh)',
    'co2_per_gdp_kg': 'Economic Carbon Intensity (kgCO2/$)',
    'hydro_share_pct': 'Hydro Share (%)',
    'transmission_loss_pct': 'Grid Losses (%)',
    'gdp_per_capita': 'GDP per Capita ($)',
    'electricity_share_energy': 'Electricity Share of Energy (%)',
    'solar_wind_share_pct': 'Solar+Wind Share (%)'
}

def create_summary_table(df, status_name):
    subset = df[df['status'] == status_name][cols_to_keep].rename(columns=labels)
    
    stats = subset.describe().T[['count', '50%', 'mean', 'std', 'min', 'max']]
    stats.columns = ['N', 'Median','Mean', 'Std. Dev.', 'Min', 'Max']
    
    html_style = stats.style.format("{:.2f}") \
        .set_caption(f"Table: Summary Statistics - {status_name} Nations") \
        .set_table_styles([{
            'selector': 'caption',
            'props': [
                ('color', 'black'), 
                ('font-weight', 'bold'), 
                ('font-size', '16px'),
                ('text-align', 'center'),
                ('padding-bottom', '10px')
            ]
        }]) \
        .to_html()
    
    return html_style

html_developed = create_summary_table(df_final, 'Developed')
html_developing = create_summary_table(df_final, 'Developing')

display(HTML(html_developed))
display(HTML("<br>")) 
display(HTML(html_developing))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = df_final.dropna(subset=['hydro_share_pct', 'carbon_intensity', 'transmission_loss_pct', 'status']).copy()
colors = {'Developed': '#1f77b4', 'Developing': '#ff7f0e'} 

fig1, ax1 = plt.subplots(figsize=(10, 6))

trend_data = df.groupby(['year', 'status'])['carbon_intensity'].mean().unstack()

ax1.plot(trend_data.index, trend_data['Developed'], 
         color=colors['Developed'], linewidth=3, label='Developed')

ax1.plot(trend_data.index, trend_data['Developing'], 
         color=colors['Developing'], linewidth=3, label='Developing')

ax1.set_title('Viz 1: Carbon Intensity Trends (2000-2021)')
ax1.set_ylabel('Average Carbon Intensity (gCO2/kWh)')
ax1.set_xlabel('Year')

start_year = int(trend_data.index.min())
end_year = int(trend_data.index.max())
ax1.set_xticks(np.arange(start_year, end_year + 1, 5)) 
ax1.set_xlim(start_year, end_year)

ax1.legend()
ax1.grid(True, alpha=0.3)

fig2, ax2 = plt.subplots(figsize=(10, 6))
line_styles = {'Developed': ':','Developing': '--'}

for status in ['Developed', 'Developing']:
    subset = df[df['status'] == status]
    # Plot Dots (No label here, we add it manually later)
    ax2.scatter(subset['hydro_share_pct'], subset['carbon_intensity'], 
                color=colors[status], alpha=0.4, s=30)
    
    # Plot Trendline (No label here)
    z = np.polyfit(subset['hydro_share_pct'], subset['carbon_intensity'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(subset['hydro_share_pct'].min(), subset['hydro_share_pct'].max(), 100)
    ax2.plot(x_range, p(x_range), color='#333333', linewidth=3, linestyle=line_styles[status])

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Developed (Data)',
           markerfacecolor=colors['Developed'], markersize=9),
    Line2D([0], [0], marker='o', color='w', label='Developing (Data)',
           markerfacecolor=colors['Developing'], markersize=9),
    Line2D([0], [0], color='#333333', lw=2, linestyle=':', label='Developed Trend'),
    Line2D([0], [0], color='#333333', lw=2, linestyle='--', label='Developing Trend')
]
ax2.legend(handles=legend_elements, loc='upper right')

ax2.set_title('Viz 2: The Paradox (Developed nations decarbonize faster with Hydro)')
ax2.set_xlabel('Hydro Share (%)')
ax2.set_ylabel('Carbon Intensity (gCO2/kWh)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig3, ax3 = plt.subplots(figsize=(10, 6))

df['Hydro_Tier'] = pd.cut(df['hydro_share_pct'], 
                          bins=[-1, 10, 50, 100], 
                          labels=['Low Hydro (<10%)', 'Med Hydro (10-50%)', 'High Hydro (>50%)'])

median_loss = df['transmission_loss_pct'].median()
df['Grid_Type'] = np.where(df['transmission_loss_pct'] < median_loss, 'Efficient Grid', 'Inefficient Grid')

bar_data = df.groupby(['Hydro_Tier', 'Grid_Type'], observed=False)['carbon_intensity'].mean().unstack()

bar_data.plot(kind='bar', ax=ax3, color=['green', 'red'], alpha=0.8)

ax3.set_title('Viz 3: The Efficiency Gap (Inefficient grids are dirtier at EVERY level)')
ax3.set_ylabel('Average Carbon Intensity (gCO2/kWh)')
ax3.set_xlabel('Hydro Dependency Tier')
ax3.grid(axis='y', alpha=0.3)
ax3.legend(title='Grid Quality')

plt.xticks(rotation=0) 
plt.tight_layout()
plt.show()

fig4, ax4 = plt.subplots(figsize=(10, 6))

hydro_rich = df[df['hydro_share_pct'] > 40]

ax4.scatter(hydro_rich['transmission_loss_pct'], hydro_rich['carbon_intensity'], 
            color='purple', alpha=0.6, s=50)

z = np.polyfit(hydro_rich['transmission_loss_pct'], hydro_rich['carbon_intensity'], 1)
p = np.poly1d(z)
x_range = np.linspace(hydro_rich['transmission_loss_pct'].min(), hydro_rich['transmission_loss_pct'].max(), 100)
ax4.plot(x_range, p(x_range), color='black', linestyle='--', linewidth=2)

ax4.set_title('Viz 4: The "Leakage" Effect (Among Hydro-Rich Nations)')
ax4.set_xlabel('Transmission Losses (%)')
ax4.set_ylabel('Carbon Intensity (gCO2/kWh)')
ax4.text(x=hydro_rich['transmission_loss_pct'].min(), 
         y=hydro_rich['carbon_intensity'].max(), 
         s="Data Filtered to Hydro Share > 40%", 
         bbox=dict(facecolor='white', alpha=0.8))
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#this is only for testing

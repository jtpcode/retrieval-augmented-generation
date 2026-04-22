import pandas as pd
import os

# Create natural language text files of the CSV data


def transaction_to_sentence(row):
    # Format dates like this: January 1, 2020
    order_date = row['Order Date'].strftime('%B %-d, %Y')
    ship_date = row['Ship Date'].strftime('%B %-d, %Y')
    ship_mode = row['Ship Mode']
    customer_name = row['Customer Name']
    segment = row['Segment']
    country = row['Country']
    city = row['City']
    state = row['State']
    region = row['Region']
    category = row['Category']
    sub_category = row['Sub-Category']
    product_name = row['Product Name']
    sales = row['Sales']
    qty = row['Quantity']
    discount = row['Discount']
    profit = row['Profit']

    return (f"On {order_date}, {customer_name} ({segment}) ordered {qty} unit(s) of {product_name} " +
            f"(in product category {category}, subcategory {sub_category}) from {city}, {state} ({region} region), {country}. " +
            f"Shipped on {ship_date} via {ship_mode}. Sales: ${sales:.2f}, discount: {discount*100:.0f}%, " +
            f"{'profit' if profit >= 0 else 'loss'}: ${abs(profit):.2f}.")


df = pd.read_csv('data/superstore.csv',
                 parse_dates=['Order Date', 'Ship Date'],
                 encoding='latin-1')

# Convert singular transactions to natural language sentences
os.makedirs('text_files', exist_ok=True)
with open('text_files/csv_rows_as_sentences.txt', 'w') as f:
    for index, row in df.iterrows():
        sentence = transaction_to_sentence(row)
        f.write(sentence + '\n')

# Summaries by time, category, region etc.

# Create a text file with monthly summaries of sales as natural language sentences
df_monthly = df.groupby(df['Order Date'].dt.to_period('M')).agg({
    'Sales': 'sum',
    'Profit': 'sum'
})

with open('text_files/monthly_summaries.txt', 'w') as f:
    for period, row in df_monthly.iterrows():
        # Format the period: for example '2014-01' -> 'January 2014'
        month_and_year = period.strftime('%B %Y')
        sales = row['Sales']
        profit = row['Profit']

        sentence = f"In {month_and_year}, total sales were ${sales:,.2f} and total {'profit' if profit >= 0 else 'loss'} was ${abs(profit):,.2f}."
        f.write(sentence + '\n')

# Create a text file with category performance summaries as natural language sentences
df_category = df.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum'
})

with open('text_files/category_summaries.txt', 'w') as f:
    for category, row in df_category.iterrows():
        sales = row['Sales']
        profit = row['Profit']

        sentence = f"In the {category} category, total sales were ${sales:,.2f} and total {'profit' if profit >= 0 else 'loss'} was ${abs(profit):,.2f}."
        f.write(sentence + '\n')

# Create a text file with subcategory performance summaries as natural language sentences
df_sub_category = df.groupby(['Category', 'Sub-Category']).agg({
    'Sales': 'sum',
    'Profit': 'sum'
})

with open('text_files/sub_category_summaries.txt', 'w') as f:
    for (category, sub_category), row in df_sub_category.iterrows():
        sales = row['Sales']
        profit = row['Profit']

        sentence = f"In the {category} category, subcategory {sub_category}, total sales were ${sales:,.2f} and total {'profit' if profit >= 0 else 'loss'} was ${abs(profit):,.2f}."
        f.write(sentence + '\n')

# Create a text file with regional performance summaries as natural language sentences
df_region = df.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum'
})

with open('text_files/region_summaries.txt', 'w') as f:
    for region, row in df_region.iterrows():
        sales = row['Sales']
        profit = row['Profit']

        sentence = f"In the {region} region, total sales were ${sales:,.2f} and total {'profit' if profit >= 0 else 'loss'} was ${abs(profit):,.2f}."
        f.write(sentence + '\n')

# Create a text file with state performance summaries as natural language sentences
df_state = df.groupby('State').agg({
    'Sales': 'sum',
    'Profit': 'sum'
})

with open('text_files/state_summaries.txt', 'w') as f:
    for state, row in df_state.iterrows():
        sales = row['Sales']
        profit = row['Profit']

        sentence = f"In {state} state, total sales were ${sales:,.2f} and total {'profit' if profit >= 0 else 'loss'} was ${abs(profit):,.2f}."
        f.write(sentence + '\n')

# Create a text file with city performance summaries as natural language sentences
df_city = df.groupby('City').agg({
    'Sales': 'sum',
    'Profit': 'sum'
})

with open('text_files/city_summaries.txt', 'w') as f:
    for city, row in df_city.iterrows():
        sales = row['Sales']
        profit = row['Profit']

        sentence = f"In {city} city, total sales were ${sales:,.2f} and total {'profit' if profit >= 0 else 'loss'} was ${abs(profit):,.2f}."
        f.write(sentence + '\n')

# Statistical summaries
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

with open('text_files/statistical_summaries.txt', 'w') as f:
    # Yearly totals
    yearly_sales = df.groupby('Year')['Sales'].sum()
    for year, sales in yearly_sales.items():
        f.write(f"In {year}, the total sales were ${sales:,.2f}.\n")

    yearly_profit = df.groupby('Year')['Profit'].sum()
    for year, profit in yearly_profit.items():
        f.write(f"In {year}, the overall {'profit' if profit >= 0 else 'loss'} was ${abs(profit):,.2f}.\n")

    # Means and medians
    mean_sales = df['Sales'].mean()
    median_sales = df['Sales'].median()
    mean_profit = df['Profit'].mean()
    median_profit = df['Profit'].median()
    f.write(f"\nOverall mean sales: ${mean_sales:,.2f}, median sales: ${median_sales:,.2f}.\n")
    f.write(f"Overall mean profit: ${mean_profit:,.2f}, median profit: ${median_profit:,.2f}.\n")

    # Top/bottom performers - Products
    product_sales = df.groupby('Product Name')['Sales'].sum()
    top_product = product_sales.idxmax()
    top_product_sales = product_sales.max()
    bottom_product = product_sales.idxmin()
    bottom_product_sales = product_sales.min()
    f.write(f"\nTop product by sales: {top_product} (${top_product_sales:,.2f}).\n")
    f.write(f"Lowest selling product: {bottom_product} (${bottom_product_sales:,.2f}).\n")

    product_profit = df.groupby('Product Name')['Profit'].sum()
    top_product_profit = product_profit.idxmax()
    top_product_profit_val = product_profit.max()
    bottom_product_profit = product_profit.idxmin()
    bottom_product_profit_val = product_profit.min()
    f.write(f"Top product by profit: {top_product_profit} (${top_product_profit_val:,.2f}).\n")
    f.write(f"Lowest profit product: {bottom_product_profit} (${bottom_product_profit_val:,.2f}).\n")

    # Top/bottom performers - Cities
    city_sales = df.groupby('City')['Sales'].sum()
    top_city = city_sales.idxmax()
    top_city_sales = city_sales.max()
    bottom_city = city_sales.idxmin()
    bottom_city_sales = city_sales.min()
    f.write(f"\nTop city by sales: {top_city} (${top_city_sales:,.2f}).\n")
    f.write(f"Lowest city by sales: {bottom_city} (${bottom_city_sales:,.2f}).\n")

    city_profit = df.groupby('City')['Profit'].sum()
    top_city_profit = city_profit.idxmax()
    top_city_profit_val = city_profit.max()
    bottom_city_profit = city_profit.idxmin()
    bottom_city_profit_val = city_profit.min()
    f.write(f"Top city by profit: {top_city_profit} (${top_city_profit_val:,.2f}).\n")
    f.write(f"Lowest profit city: {bottom_city_profit} (${bottom_city_profit_val:,.2f}).\n")

    # Top/bottom performers - States
    state_sales = df.groupby('State')['Sales'].sum()
    top_state = state_sales.idxmax()
    top_state_sales = state_sales.max()
    bottom_state = state_sales.idxmin()
    bottom_state_sales = state_sales.min()
    f.write(f"\nTop state by sales: {top_state} (${top_state_sales:,.2f}).\n")
    f.write(f"Lowest state by sales: {bottom_state} (${bottom_state_sales:,.2f}).\n")

    state_profit = df.groupby('State')['Profit'].sum()
    top_state_profit = state_profit.idxmax()
    top_state_profit_val = state_profit.max()
    bottom_state_profit = state_profit.idxmin()
    bottom_state_profit_val = state_profit.min()
    f.write(f"Top state by profit: {top_state_profit} (${top_state_profit_val:,.2f}).\n")
    f.write(f"Lowest profit state: {bottom_state_profit} (${bottom_state_profit_val:,.2f}).\n")

    # Top/bottom performers - Regions
    region_sales = df.groupby('Region')['Sales'].sum()
    top_region = region_sales.idxmax()
    top_region_sales = region_sales.max()
    bottom_region = region_sales.idxmin()
    bottom_region_sales = region_sales.min()
    f.write(f"\nTop region by sales: {top_region} (${top_region_sales:,.2f}).\n")
    f.write(f"Lowest region by sales: {bottom_region} (${bottom_region_sales:,.2f}).\n")

    region_profit = df.groupby('Region')['Profit'].sum()
    top_region_profit = region_profit.idxmax()
    top_region_profit_val = region_profit.max()
    bottom_region_profit = region_profit.idxmin()
    bottom_region_profit_val = region_profit.min()
    f.write(f"Top region by profit: {top_region_profit} (${top_region_profit_val:,.2f}).\n")
    f.write(f"Lowest profit region: {bottom_region_profit} (${bottom_region_profit_val:,.2f}).\n")

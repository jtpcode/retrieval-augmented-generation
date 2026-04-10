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
    yearly_sales = df.groupby('Year')['Sales'].sum()
    for year, sales in yearly_sales.items():
        f.write(f"In {year}, the total sales were ${sales:,.2f}.\n")

    yearly_profit = df.groupby('Year')['Profit'].sum()
    for year, profit in yearly_profit.items():
        f.write(f"In {year}, the overall {'profit' if profit >= 0 else 'loss'} was ${abs(profit):,.2f}.\n")

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

        sentence = (
            f"From 2014 to 2017, in the {category} category, total sales were ${sales:,.2f} "
            f"and total {'profit' if profit >= 0 else 'loss'} was ${abs(profit):,.2f}."
        )
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

    # Top/bottom performers - Products
    product_sales = df.groupby('Product Name')['Sales'].sum()
    top5_products_sales = product_sales.sort_values(ascending=False).head(5)
    bottom5_products_sales = product_sales.sort_values(ascending=True).head(5)
    f.write("\nTop 5 products by sales:\n")
    for product, sales in top5_products_sales.items():
        f.write(f"  {product}: ${sales:,.2f}\n")
    f.write("Lowest 5 products by sales:\n")
    for product, sales in bottom5_products_sales.items():
        f.write(f"  {product}: ${sales:,.2f}\n")

    product_profit = df.groupby('Product Name')['Profit'].sum()
    top5_products_profit = product_profit.sort_values(ascending=False).head(5)
    bottom5_products_profit = product_profit.sort_values(ascending=True).head(5)
    f.write("Top 5 products by profit:\n")
    for product, profit in top5_products_profit.items():
        f.write(f"  {product}: ${profit:,.2f}\n")
    f.write("Lowest 5 products by profit:\n")
    for product, profit in bottom5_products_profit.items():
        f.write(f"  {product}: ${profit:,.2f}\n")

    # Top/bottom performers - Cities
    city_sales = df.groupby('City')['Sales'].sum()
    top5_cities_sales = city_sales.sort_values(ascending=False).head(5)
    bottom5_cities_sales = city_sales.sort_values(ascending=True).head(5)
    f.write("\nTop 5 cities by sales:\n")
    for city, sales in top5_cities_sales.items():
        f.write(f"  {city}: ${sales:,.2f}\n")
    f.write("Lowest 5 cities by sales:\n")
    for city, sales in bottom5_cities_sales.items():
        f.write(f"  {city}: ${sales:,.2f}\n")

    city_profit = df.groupby('City')['Profit'].sum()
    top5_cities_profit = city_profit.sort_values(ascending=False).head(5)
    bottom5_cities_profit = city_profit.sort_values(ascending=True).head(5)
    f.write("Top 5 cities by profit:\n")
    for city, profit in top5_cities_profit.items():
        f.write(f"  {city}: ${profit:,.2f}\n")
    f.write("Lowest 5 cities by profit:\n")
    for city, profit in bottom5_cities_profit.items():
        f.write(f"  {city}: ${profit:,.2f}\n")

    # Top/bottom performers - States
    state_sales = df.groupby('State')['Sales'].sum()
    top5_states_sales = state_sales.sort_values(ascending=False).head(5)
    bottom5_states_sales = state_sales.sort_values(ascending=True).head(5)
    f.write("\nTop 5 states by sales:\n")
    for state, sales in top5_states_sales.items():
        f.write(f"  {state}: ${sales:,.2f}\n")
    f.write("Lowest 5 states by sales:\n")
    for state, sales in bottom5_states_sales.items():
        f.write(f"  {state}: ${sales:,.2f}\n")

    state_profit = df.groupby('State')['Profit'].sum()
    top5_states_profit = state_profit.sort_values(ascending=False).head(5)
    bottom5_states_profit = state_profit.sort_values(ascending=True).head(5)
    f.write("Top 5 states by profit:\n")
    for state, profit in top5_states_profit.items():
        f.write(f"  {state}: ${profit:,.2f}\n")
    f.write("Lowest 5 states by profit:\n")
    for state, profit in bottom5_states_profit.items():
        f.write(f"  {state}: ${profit:,.2f}\n")

# Profit margin summaries (profit as % of sales)
with open('text_files/profit_margin_summaries.txt', 'w') as f:
    # Yearly profit margins
    yearly = df.groupby('Year')[['Sales', 'Profit']].sum()
    for year, row in yearly.iterrows():
        margin = (row['Profit'] / row['Sales']) * 100
        f.write(
            f"In {year}, across all Superstore transactions, the profit margin was {margin:.1f}% "
            f"(total sales: ${row['Sales']:,.2f}, total profit: ${row['Profit']:,.2f}).\n"
        )

    f.write('\n')

    # Category profit margins
    cat = df.groupby('Category')[['Sales', 'Profit']].sum()
    for category, row in cat.iterrows():
        margin = (row['Profit'] / row['Sales']) * 100
        f.write(
            f"From 2014 to 2017, the {category} category had a profit margin of {margin:.1f}% "
            f"(sales: ${row['Sales']:,.2f}, profit: ${row['Profit']:,.2f}).\n"
        )

    f.write('\n')

    # Sub-category profit margins, sorted highest to lowest
    subcat = df.groupby(['Category', 'Sub-Category'])[['Sales', 'Profit']].sum()
    subcat['Margin'] = (subcat['Profit'] / subcat['Sales']) * 100
    subcat_sorted = subcat.sort_values('Margin', ascending=False)
    for (category, sub_category), row in subcat_sorted.iterrows():
        f.write(
            f"From 2014 to 2017, in the {category} category, subcategory {sub_category} had a profit margin of "
            f"{row['Margin']:.1f}% (total sales: ${row['Sales']:,.2f}, total profit: ${row['Profit']:,.2f}).\n"
        )

# Category-by-year breakdown
with open('text_files/category_year_summaries.txt', 'w') as f:
    cat_year = df.groupby(['Year', 'Category'])[['Sales', 'Profit']].sum()
    for (year, category), row in cat_year.iterrows():
        margin = (row['Profit'] / row['Sales']) * 100
        f.write(
            f"In {year}, the {category} category had sales of ${row['Sales']:,.2f}, "
            f"profit of ${row['Profit']:,.2f}, and a profit margin of {margin:.1f}%.\n"
        )

# Seasonal patterns (average sales by month name across all years)
MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

with open('text_files/seasonal_summaries.txt', 'w') as f:
    seasonal = df.groupby('Month')[['Sales', 'Profit']].mean()
    for month_num, row in seasonal.iterrows():
        month_name = MONTH_NAMES[month_num - 1]
        f.write(
            f"On average across all years, {month_name} has monthly sales of "
            f"${row['Sales']:,.2f} and monthly profit of ${row['Profit']:,.2f}.\n"
        )

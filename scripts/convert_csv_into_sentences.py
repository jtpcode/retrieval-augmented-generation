import pandas as pd
import os

# Convert each transaction to a natural language description


def transaction_to_sentence(row):
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
            f"(from product category {category}, subcategory {sub_category}) from {city}, {state} ({region} region), {country}. " +
            f"Shipped on {ship_date} via {ship_mode}. Sales: ${sales:.2f}, discount: {discount*100:.0f}%, " +
            f"{'profit' if profit >= 0 else 'loss'}: ${abs(profit):.2f}.")


csv = pd.read_csv('data/superstore.csv',
                  parse_dates=['Order Date', 'Ship Date'],
                  encoding='latin-1')

os.makedirs('text_files', exist_ok=True)
with open('text_files/csv_as_sentences.txt', 'w') as f:
    for index, row in csv.iterrows():
        sentence = transaction_to_sentence(row)
        f.write(sentence + '\n')

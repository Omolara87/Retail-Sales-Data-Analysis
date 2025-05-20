import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime

# Load data from CSV
sales = pd.read_csv("sales_data.csv")
products = pd.read_csv("product_data.csv")
customers = pd.read_csv("customer_data.csv")

# 1. Convert date column to datetime
sales['Date'] = pd.to_datetime(sales['Date'])

# 2. Extract date features
sales['Month'] = sales['Date'].dt.month
sales['Day'] = sales['Date'].dt.day
sales['Weekday'] = sales['Date'].dt.day_name()

# 3. Merge datasets
df = sales.merge(products, on='Product_ID', how='left')
df = df.merge(customers, on='Customer_ID', how='left')

# 4. Handle missing values
df.fillna(method='ffill', inplace=True)

# 5. Handle outliers (clip units sold and unit price)
df['Units_Sold'] = df['Units_Sold'].clip(lower=1, upper=20)
df['Unit_Price'] = df['Unit_Price'].clip(lower=10, upper=500)

# 6. Calculate total sales
df['Total_Sales'] = df['Units_Sold'] * df['Unit_Price']

# 7. Total sales per product, category, and region
sales_by_product = df.groupby('Product_Name')['Total_Sales'].sum()
sales_by_category = df.groupby('Category')['Total_Sales'].sum()
sales_by_region = df.groupby('Region_x')['Total_Sales'].sum()

# 8. Top-selling products and categories
top_products = sales_by_product.sort_values(ascending=False).head(5)
top_categories = sales_by_category.sort_values(ascending=False)

# 9. Analyze sales trends
monthly_sales = df.groupby('Month')['Total_Sales'].sum()

# 10. Customer segmentation (basic RFM approach)
rfm = df.groupby('Customer_ID').agg({
    'Date': lambda x: (df['Date'].max() - x.max()).days,
    'Sale_ID': 'count',
    'Total_Sales': 'sum'
}).rename(columns={'Date': 'Recency', 'Sale_ID': 'Frequency', 'Total_Sales': 'Monetary'})

# 11. Calculate CLV (simple: frequency × avg monetary)
rfm['CLV'] = rfm['Frequency'] * (rfm['Monetary'] / rfm['Frequency'])

# 12. Line plot for sales trend
plt.figure(figsize=(10, 4))
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.ylabel("Total Sales")
plt.xlabel("Month")
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_sales_trend.png")
plt.close()

# 13. Bar charts for top products and categories
top_products.plot(kind='bar', title="Top Selling Products", ylabel="Total Sales")
plt.tight_layout()
plt.savefig("top_products.png")
plt.close()

top_categories.plot(kind='bar', color='orange', title="Top Categories")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("top_categories.png")
plt.close()

# 14. Heatmap: correlation between numeric variables by region
corr = df[['Units_Sold', 'Unit_Price', 'Total_Sales']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Sales Feature Correlation")
plt.tight_layout()
plt.savefig("sales_correlation_heatmap.png")
plt.close()

# 15. Promotions impact analysis
promo_impact = df.groupby('Promotion_Applied')['Total_Sales'].mean()

# 16. Products with declining sales
sales_by_month_product = df.groupby(['Month', 'Product_Name'])['Total_Sales'].sum().unstack()
declining_products = sales_by_month_product.pct_change().mean().sort_values()

# 17. Inventory adjustment recommendation (based on avg sales)
inventory_suggestions = df.groupby('Product_Name')['Units_Sold'].mean().sort_values(ascending=False)

# 18. Export analysis to CSV
sales_by_product.to_csv("sales_by_product.csv")
rfm.to_csv("customer_rfm.csv")
inventory_suggestions.to_csv("inventory_suggestions.csv")

# 19. Connect to SQL database (demo connection, assume SQLite)
conn = sqlite3.connect(":memory:")
df.to_sql("sales_data", conn, index=False, if_exists="replace")

# 20. Document findings (printed here, can be written to a file)
print("Top Products:\n", top_products)
print("\nTop Categories:\n", top_categories)
print("\nMonthly Sales:\n", monthly_sales)
print("\nPromotion Impact:\n", promo_impact)
print("\nProducts with Declining Sales:\n", declining_products.head(3))
print("\nInventory Recommendations:\n", inventory_suggestions)

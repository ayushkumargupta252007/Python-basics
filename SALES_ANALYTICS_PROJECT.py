#===================================================================
# PROJECT   : Sales Analytics and Business Decision Support System
# LANGUAGE  : Python (Beginner Friendly, No OOP, No Lambda)
# LIBRARIES : pandas, numpy, matplotlib, scikit-learn
# PURPOSE   : Analyze historical sales data and use Machine Learning
#             to help a company make future business decisions.
#===================================================================

# We import pandas to read, store and analyze tabular data (like Excel tables)
import pandas as pd

# We import numpy to perform fast numerical calculations
import numpy as np

# We import matplotlib to draw graphs and charts
import matplotlib.pyplot as plt

# We import LinearRegression to predict numbers like future Sales and Profit
from sklearn.linear_model import LinearRegression

# We import KMeans to group regions/products into performance clusters
from sklearn.cluster import KMeans

# We import RandomForestClassifier to classify performance categories
# (a Random Forest trains many small decision trees together and combines
# their votes, which usually gives more reliable predictions than one
# single decision tree)
from sklearn.ensemble import RandomForestClassifier


#===================================================================
# SECTION 1 : GLOBAL VARIABLES
#===================================================================
# In this beginner-friendly version we do NOT use classes.
# Instead we use simple global variables that every function can use.

# This variable will store the historical sales data after loading
sales_data = None

# This flag tells the program whether the dataset has been loaded yet
dataset_loaded = False

# This flag tells the program whether the ML models have been trained yet
models_trained = False

# These variables will hold our trained Machine Learning models
sales_regression_model = None
profit_regression_model = None
region_cluster_model = None
performance_forest_model = None

# These dictionaries convert text (Region, Category) into numbers
# because Machine Learning models only understand numbers, not text
region_number_map = {}
category_number_map = {}

# After Menu Option 12 runs, the predicted future data is stored here so
# that Business Suggestions (Menu Option 13) can generate suggestions
# from it as well as from the historical data
predicted_data = None
predicted_data_available = False

# A simple dictionary to turn a month number (1-12) into its name.
# This avoids needing an extra import just to display month names.
month_name_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# This list stores every business recommendation created during the session
# so it can be written into a saved report file at the end
report_lines = []

# This is the folder where all graphs will be saved
# All graph images are saved directly in the current folder using this
# prefix on the file name (no separate folder needed, so we don't need
# the "os" module at all)
graph_file_prefix = "graph_"


#===================================================================
# SECTION 2 : HELPER FUNCTIONS (used by many menu options)
#===================================================================

def add_line_to_report(text_line):
    # This function adds one line of text to our report list
    # We use this everywhere so the final saved report has everything
    report_lines.append(text_line)


def check_dataset_is_loaded():
    # This function checks if the dataset has been loaded
    # Many menu options need this check before they can run
    if dataset_loaded == False:
        print("\nNo dataset loaded yet.")
        print("Please choose Menu Option 1 (Load Dataset) first.\n")
        return False
    else:
        return True


#===================================================================
# SECTION 3 : MENU OPTION 1 - LOAD DATASET
#===================================================================

# These are the column names every menu option in this program expects
# to find in the CSV file. If you want to load your OWN csv file
# (instead of the sample sales_data.csv), just make sure it has these
# same column names, in any order.
required_columns_list = [
    "Order_ID", "Order_Date", "Region", "Category", "Product",
    "Customer_Name", "Quantity", "Discount", "Sales", "Profit"
]


def load_dataset():
    # This function reads the sales data CSV file into a pandas DataFrame
    global sales_data
    global dataset_loaded

    print("\n--- LOAD DATASET ---")
    file_path = input("Enter the CSV file name (example: sales_data.csv): ")

    # Remove extra spaces, and remove quote characters in case the user
    # copy-pasted a path that came wrapped in quotes (this happens a lot
    # when copying a file path on Windows)
    file_path = file_path.strip()
    file_path = file_path.strip('"')
    file_path = file_path.strip("'")

    # We try to read the file directly instead of checking os.path.exists()
    # first. This avoids needing the "os" module: if the file is missing,
    # pandas raises a FileNotFoundError, which we simply catch below.
    try:
        new_sales_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found:", file_path)
        print("Things to check:")
        print("  1. Is the CSV file saved in the SAME folder as this .py file?")
        print("  2. Did you type the extension correctly? (Windows sometimes")
        print("     hides the real extension, so 'data.csv' might actually")
        print("     be named 'data.csv.csv' on disk.)")
        print("  3. If the file is in a different folder, type the FULL path,")
        print("     for example: C:/Users/YourName/Desktop/sales_data.csv")
        return
    except Exception as error_message:
        print("The file was found but could not be read as a CSV.")
        print("Error details:", error_message)
        return

    # Check that every column this program needs is actually present.
    # This is what lets the program work with ANY csv file the user
    # provides, as long as the column names match.
    missing_columns_list = []
    for column_name in required_columns_list:
        if column_name not in new_sales_data.columns:
            missing_columns_list.append(column_name)

    if len(missing_columns_list) > 0:
        print("\nThis CSV file is missing the following required column(s):")
        print(missing_columns_list)
        print("\nColumns found in your file:")
        print(list(new_sales_data.columns))
        print("\nPlease rename your columns to match, or use sales_data.csv,")
        print("and try loading again.\n")
        return

    # Only now, after every check has passed, do we replace the real
    # sales_data variable, so a bad file never overwrites good data.
    sales_data = new_sales_data

    # pd.to_datetime() converts the text date column into a real date type
    # so that we can extract Month and Year for trend analysis
    sales_data["Order_Date"] = pd.to_datetime(sales_data["Order_Date"])

    # We create two new columns: Month and Year
    # dt.month and dt.year pull out the month number and year number
    sales_data["Month"] = sales_data["Order_Date"].dt.month
    sales_data["Year"] = sales_data["Order_Date"].dt.year

    dataset_loaded = True

    total_rows = len(sales_data)
    print("Dataset loaded successfully.")
    print("Total records loaded:", total_rows)
    add_line_to_report("Dataset loaded with " + str(total_rows) + " records.")


#===================================================================
# SECTION 4 : MENU OPTION 2 - DATASET INFORMATION
#===================================================================

def show_dataset_information():
    # This function shows basic information about the loaded dataset
    if check_dataset_is_loaded() == False:
        return

    print("\n--- DATASET INFORMATION ---")

    # shape gives (number_of_rows, number_of_columns)
    print("Number of rows:", sales_data.shape[0])
    print("Number of columns:", sales_data.shape[1])

    print("\nColumn Names:")
    print(list(sales_data.columns))

    print("\nFirst 5 rows of data (head):")
    print(sales_data.head())

    print("\nMissing values in each column (isnull().sum()):")
    print(sales_data.isnull().sum())

    print("\nStatistical Summary (describe()):")
    print(sales_data.describe())


#===================================================================
# SECTION 5 : MENU OPTION 3 - OVERALL BUSINESS SUMMARY
#===================================================================

def show_overall_business_summary():
    # This function calculates the most important overall numbers
    if check_dataset_is_loaded() == False:
        return

    print("\n--- OVERALL BUSINESS SUMMARY ---")

    # sum() adds up every value in the Sales column
    total_sales = sales_data["Sales"].sum()

    # sum() adds up every value in the Profit column
    total_profit = sales_data["Profit"].sum()

    # mean() calculates the average value
    average_sales = sales_data["Sales"].mean()
    average_profit = sales_data["Profit"].mean()

    # count of unique customers, products and orders
    total_orders = sales_data["Order_ID"].nunique()
    total_customers = sales_data["Customer_Name"].nunique()
    total_products = sales_data["Product"].nunique()

    # average discount given across all orders
    average_discount = sales_data["Discount"].mean()

    print("Total Orders            :", total_orders)
    print("Total Unique Customers  :", total_customers)
    print("Total Unique Products   :", total_products)
    print("Total Sales Revenue     : Rs.", round(total_sales, 2))
    print("Total Profit            : Rs.", round(total_profit, 2))
    print("Average Sales per Order : Rs.", round(average_sales, 2))
    print("Average Profit per Order: Rs.", round(average_profit, 2))
    print("Average Discount Given  :", round(average_discount * 100, 2), "%")

    add_line_to_report("Overall Total Sales: Rs. " + str(round(total_sales, 2)))
    add_line_to_report("Overall Total Profit: Rs. " + str(round(total_profit, 2)))


#===================================================================
# SECTION 6 : MENU OPTION 4 - PRODUCT ANALYSIS
#===================================================================

def product_analysis():
    # This function analyzes performance of every product
    if check_dataset_is_loaded() == False:
        return

    print("\n--- PRODUCT ANALYSIS ---")

    # groupby() collects all rows for the same product together
    # sum() then adds up the Sales for each product group
    product_sales = sales_data.groupby("Product")["Sales"].sum()

    # sort_values() arranges the numbers from smallest to biggest
    # ascending=False means biggest number comes first
    product_sales_sorted = product_sales.sort_values(ascending=False)

    print("\nTop 10 Best Selling Products:")
    # head(10) shows only the first 10 rows (the top 10 here)
    print(product_sales_sorted.head(10))

    print("\nBottom 10 Weak Selling Products:")
    # tail(10) shows the last 10 rows (the bottom 10 here)
    print(product_sales_sorted.tail(10))

    # idxmax() gives the label (product name) of the maximum value
    best_product = product_sales.idxmax()
    # idxmin() gives the label (product name) of the minimum value
    worst_product = product_sales.idxmin()

    print("\nHighest Selling Product Overall:", best_product)
    print("Lowest Selling Product Overall :", worst_product)

    # We also check which product is ordered the MOST number of times
    product_order_count = sales_data.groupby("Product")["Order_ID"].count()
    most_ordered_product = product_order_count.idxmax()
    least_ordered_product = product_order_count.idxmin()

    print("\nMost Frequently Ordered Product :", most_ordered_product)
    print("Least Frequently Ordered Product:", least_ordered_product)

    add_line_to_report("Best Product: " + str(best_product))
    add_line_to_report("Worst Product: " + str(worst_product))

    # -----------------------------------------------------------
    # DETAILED SECTION 1 : Region-wise Category breakdown
    # -----------------------------------------------------------
    # This answers: "in which region does which category sell the
    # most, and which sells the least?"
    print("\nWould you like to see a detailed Region-wise Category")
    show_detail_choice = input("breakdown for products? (yes/no): ").strip().lower()

    if show_detail_choice == "yes":
        show_region_category_breakdown()

    # -----------------------------------------------------------
    # DETAILED SECTION 2 : Look up one specific combination
    # -----------------------------------------------------------
    print("\nWould you like to check a specific Product + Category")
    lookup_choice = input("+ Region combination? (yes/no): ").strip().lower()

    if lookup_choice == "yes":
        lookup_specific_product_category_region()


def show_region_category_breakdown():
    # This function shows, for every region, which category sold the
    # most and which sold the least - a detailed drill-down beyond the
    # simple overall Product Analysis totals above.
    print("\n--- REGION-WISE CATEGORY BREAKDOWN ---")

    # Group by TWO columns together: Region and Category.
    # This groups every row into a (Region, Category) pair before summing.
    region_category_sales = sales_data.groupby(["Region", "Category"])["Sales"].sum()

    unique_regions = sales_data["Region"].unique()

    # We build one row of a summary table for every region
    summary_rows = []
    for region_name in unique_regions:
        # .loc[region_name] pulls out just the categories for this region
        category_sales_in_region = region_category_sales.loc[region_name]

        best_category_in_region = category_sales_in_region.idxmax()
        best_category_sales_value = category_sales_in_region.max()

        worst_category_in_region = category_sales_in_region.idxmin()
        worst_category_sales_value = category_sales_in_region.min()

        summary_rows.append([
            region_name,
            best_category_in_region,
            round(best_category_sales_value, 2),
            worst_category_in_region,
            round(worst_category_sales_value, 2),
        ])

    # We turn our list of rows into a DataFrame so it prints as a clean table
    breakdown_table = pd.DataFrame(summary_rows, columns=[
        "Region", "Best Category", "Best Category Sales",
        "Weak Category", "Weak Category Sales"
    ])

    # index=False hides the default 0,1,2... row numbers for a cleaner look
    print(breakdown_table.to_string(index=False))


def lookup_specific_product_category_region():
    # This function lets the user type in one Product, Category and Region
    # and instantly see the sales performance of that exact combination
    print("\n--- SPECIFIC PRODUCT / CATEGORY / REGION LOOKUP ---")

    print("Available Regions   :", list(sales_data["Region"].unique()))
    print("Available Categories:", list(sales_data["Category"].unique()))

    region_input = input("Enter Region: ").strip()
    category_input = input("Enter Category: ").strip()
    product_input = input("Enter Product name: ").strip()

    # This filter keeps only the rows that match ALL THREE conditions
    filtered_rows = sales_data[
        (sales_data["Region"] == region_input) &
        (sales_data["Category"] == category_input) &
        (sales_data["Product"] == product_input)
    ]

    if len(filtered_rows) == 0:
        print("\nNo matching orders found for that Product + Category + Region.")
        print("Please check the spelling (names are case-sensitive).\n")
        return

    total_sales_value = filtered_rows["Sales"].sum()
    total_profit_value = filtered_rows["Profit"].sum()
    total_quantity_value = filtered_rows["Quantity"].sum()
    total_orders_value = len(filtered_rows)

    print("\nResults for", product_input, "(", category_input, ") in", region_input, "region:")
    print("Total Orders   :", total_orders_value)
    print("Total Quantity Sold:", total_quantity_value)
    print("Total Sales    : Rs.", round(total_sales_value, 2))
    print("Total Profit   : Rs.", round(total_profit_value, 2))


#===================================================================
# SECTION 7 : MENU OPTION 5 - CATEGORY ANALYSIS
#===================================================================

def category_analysis():
    # This function analyzes performance of every product category
    if check_dataset_is_loaded() == False:
        return

    print("\n--- CATEGORY ANALYSIS ---")

    category_sales = sales_data.groupby("Category")["Sales"].sum()
    category_profit = sales_data.groupby("Category")["Profit"].sum()

    category_sales_sorted = category_sales.sort_values(ascending=False)
    category_profit_sorted = category_profit.sort_values(ascending=False)

    print("\nCategory wise Total Sales (highest to lowest):")
    print(category_sales_sorted)

    print("\nCategory wise Total Profit (highest to lowest):")
    print(category_profit_sorted)

    best_sales_category = category_sales.idxmax()
    worst_sales_category = category_sales.idxmin()

    best_profit_category = category_profit.idxmax()
    worst_profit_category = category_profit.idxmin()

    print("\nHighest Sales Category  :", best_sales_category)
    print("Lowest Sales Category   :", worst_sales_category)
    print("Most Profitable Category:", best_profit_category)
    print("Least Profitable Category:", worst_profit_category)

    add_line_to_report("Best Category (Sales): " + str(best_sales_category))
    add_line_to_report("Weak Category (Sales): " + str(worst_sales_category))
    add_line_to_report("Category needing marketing focus: " + str(worst_sales_category))

    # -----------------------------------------------------------
    # DETAILED SECTION : for every Category, which Region sells the
    # most of it, and which Region sells the least of it?
    # -----------------------------------------------------------
    print("\n--- CATEGORY-WISE REGION BREAKDOWN ---")

    # Group by Category first, then Region inside each category
    category_region_sales = sales_data.groupby(["Category", "Region"])["Sales"].sum()

    unique_categories = sales_data["Category"].unique()

    summary_rows = []
    for category_name in unique_categories:
        region_sales_in_category = category_region_sales.loc[category_name]

        best_region_for_category = region_sales_in_category.idxmax()
        best_region_sales_value = region_sales_in_category.max()

        worst_region_for_category = region_sales_in_category.idxmin()
        worst_region_sales_value = region_sales_in_category.min()

        summary_rows.append([
            category_name,
            best_region_for_category,
            round(best_region_sales_value, 2),
            worst_region_for_category,
            round(worst_region_sales_value, 2),
        ])

    breakdown_table = pd.DataFrame(summary_rows, columns=[
        "Category", "Best Region", "Best Region Sales",
        "Weak Region", "Weak Region Sales"
    ])

    print(breakdown_table.to_string(index=False))


#===================================================================
# SECTION 8 : MENU OPTION 6 - REGION ANALYSIS
#===================================================================

def region_analysis():
    # This function analyzes performance of every region
    if check_dataset_is_loaded() == False:
        return

    print("\n--- REGION ANALYSIS ---")

    region_sales = sales_data.groupby("Region")["Sales"].sum()
    region_profit = sales_data.groupby("Region")["Profit"].sum()

    region_sales_sorted = region_sales.sort_values(ascending=False)
    region_profit_sorted = region_profit.sort_values(ascending=False)

    print("\nRegion wise Total Sales (highest to lowest):")
    print(region_sales_sorted)

    print("\nRegion wise Total Profit (highest to lowest):")
    print(region_profit_sorted)

    best_region = region_sales.idxmax()
    worst_region = region_sales.idxmin()

    best_profit_region = region_profit.idxmax()
    worst_profit_region = region_profit.idxmin()

    print("\nBest Performing Region (Sales) :", best_region)
    print("Weakest Performing Region (Sales):", worst_region)
    print("Highest Profit Region           :", best_profit_region)
    print("Lowest Profit Region            :", worst_profit_region)

    print("\nBusiness Interpretation:")
    print("-> Region", best_region, "should receive more stock/inventory.")
    print("-> Region", worst_region, "may need marketing investment.")

    add_line_to_report("Best Region: " + str(best_region) + " (needs more stock)")
    add_line_to_report("Weak Region: " + str(worst_region) + " (needs marketing investment)")

    # -----------------------------------------------------------
    # DETAILED SECTION : for every Region, which Category and which
    # Product performs the best?
    # -----------------------------------------------------------
    print("\n--- REGION-WISE TOP CATEGORY AND TOP PRODUCT ---")

    region_category_sales = sales_data.groupby(["Region", "Category"])["Sales"].sum()
    region_product_sales = sales_data.groupby(["Region", "Product"])["Sales"].sum()

    unique_regions = sales_data["Region"].unique()

    summary_rows = []
    for region_name in unique_regions:
        top_category_in_region = region_category_sales.loc[region_name].idxmax()
        top_product_in_region = region_product_sales.loc[region_name].idxmax()

        summary_rows.append([region_name, top_category_in_region, top_product_in_region])

    breakdown_table = pd.DataFrame(summary_rows, columns=[
        "Region", "Top Category", "Top Product"
    ])

    print(breakdown_table.to_string(index=False))


#===================================================================
# SECTION 9 : MENU OPTION 7 - CUSTOMER ANALYSIS
#===================================================================

def customer_analysis():
    # This function finds the best and worst customers by revenue
    if check_dataset_is_loaded() == False:
        return

    print("\n--- CUSTOMER ANALYSIS ---")

    customer_sales = sales_data.groupby("Customer_Name")["Sales"].sum()
    customer_sales_sorted = customer_sales.sort_values(ascending=False)

    print("\nTop 10 Customers by Total Sales:")
    print(customer_sales_sorted.head(10))

    best_customer = customer_sales.idxmax()
    worst_customer = customer_sales.idxmin()

    # Calculate what percentage of total revenue the best customer contributes
    total_sales_all = sales_data["Sales"].sum()
    best_customer_sales_value = customer_sales.max()
    best_customer_percentage = (best_customer_sales_value / total_sales_all) * 100

    print("\nBest Customer :", best_customer)
    print("Contribution of Best Customer to Total Revenue:",
          round(best_customer_percentage, 2), "%")
    print("Weakest Customer (lowest revenue):", worst_customer)

    add_line_to_report("Best Customer: " + str(best_customer) + " contributes " +
                        str(round(best_customer_percentage, 2)) + "% of revenue.")

    # -----------------------------------------------------------
    # DETAILED SECTION : which customers of which region are
    # interested in which category, shown briefly as a table
    # -----------------------------------------------------------
    print("\n--- CUSTOMER INTEREST BY REGION (Detailed Table) ---")

    region_category_sales = sales_data.groupby(["Region", "Category"])["Sales"].sum()
    region_customer_sales = sales_data.groupby(["Region", "Customer_Name"])["Sales"].sum()

    unique_regions = sales_data["Region"].unique()

    summary_rows = []
    for region_name in unique_regions:
        top_category_in_region = region_category_sales.loc[region_name].idxmax()
        top_customer_in_region = region_customer_sales.loc[region_name].idxmax()
        top_customer_sales_value = region_customer_sales.loc[region_name].max()

        summary_rows.append([
            region_name,
            top_category_in_region,
            top_customer_in_region,
            round(top_customer_sales_value, 2),
        ])

    interest_table = pd.DataFrame(summary_rows, columns=[
        "Region", "Most Popular Category", "Top Customer", "Top Customer Sales"
    ])

    print(interest_table.to_string(index=False))
    print("\n(This table shows, for each region, the category customers")
    print("buy the most, and the single customer spending the most there.)")


#===================================================================
# SECTION 10 : MENU OPTION 8 - MONTHLY SALES TREND
#===================================================================

def monthly_sales_trend():
    # This function shows which months have strong or weak sales
    if check_dataset_is_loaded() == False:
        return

    print("\n--- MONTHLY SALES TREND ---")

    # Group by Month number (1 to 12) and sum the Sales
    monthly_sales = sales_data.groupby("Month")["Sales"].sum()

    # Build a small table that shows the Month NAME (not just the number)
    # next to its total Sales, using our month_name_map dictionary
    month_number_list = []
    month_name_list = []
    month_sales_list = []
    for month_number in monthly_sales.index:
        month_number_list.append(month_number)
        month_name_list.append(month_name_map[month_number])
        month_sales_list.append(round(monthly_sales[month_number], 2))

    monthly_table = pd.DataFrame({
        "Month": month_name_list,
        "Total Sales": month_sales_list
    })
    print("\nMonth wise Total Sales:")
    print(monthly_table.to_string(index=False))

    best_month_number = monthly_sales.idxmax()
    worst_month_number = monthly_sales.idxmin()
    best_month_name = month_name_map[best_month_number]
    worst_month_name = month_name_map[worst_month_number]

    print("\nStrongest Sales Month:", best_month_name)
    print("Weakest Sales Month  :", worst_month_name)

    # Also show a Year wise trend if more than one year exists
    yearly_sales = sales_data.groupby("Year")["Sales"].sum()
    print("\nYear wise Total Sales:")
    print(yearly_sales)

    add_line_to_report("Strongest sales month: " + str(best_month_name))
    add_line_to_report("Weakest sales month: " + str(worst_month_name))

    # -----------------------------------------------------------
    # DETAILED SECTION 1 : which Category (and Product) gained the
    # most sales in each month?
    # -----------------------------------------------------------
    print("\n--- MONTH-WISE TOP CATEGORY AND TOP PRODUCT ---")

    month_category_sales = sales_data.groupby(["Month", "Category"])["Sales"].sum()
    month_category_product_sales = sales_data.groupby(["Month", "Category", "Product"])["Sales"].sum()

    summary_rows = []
    for month_number in monthly_sales.index:
        top_category_in_month = month_category_sales.loc[month_number].idxmax()

        # Now find the top PRODUCT inside that month's top category
        product_sales_in_month_category = month_category_product_sales.loc[month_number, top_category_in_month]
        top_product_in_month = product_sales_in_month_category.idxmax()

        summary_rows.append([
            month_name_map[month_number],
            top_category_in_month,
            top_product_in_month,
        ])

    month_category_table = pd.DataFrame(summary_rows, columns=[
        "Month", "Top Category", "Top Product in that Category"
    ])
    print(month_category_table.to_string(index=False))

    # -----------------------------------------------------------
    # DETAILED SECTION 2 : which category do customers in each
    # region seem to "like" the most (highest sales)?
    # -----------------------------------------------------------
    print("\n--- CUSTOMER CATEGORY PREFERENCE BY REGION ---")

    region_category_sales = sales_data.groupby(["Region", "Category"])["Sales"].sum()
    unique_regions = sales_data["Region"].unique()

    preference_rows = []
    for region_name in unique_regions:
        preferred_category = region_category_sales.loc[region_name].idxmax()
        preference_rows.append([region_name, preferred_category])

    preference_table = pd.DataFrame(preference_rows, columns=[
        "Region", "Most Liked Category (by Sales)"
    ])
    print(preference_table.to_string(index=False))


#===================================================================
# SECTION 11 : MENU OPTION 9 - PROFIT ANALYSIS
#===================================================================

def profit_analysis():
    # This function focuses purely on Profit numbers, separate from Sales
    if check_dataset_is_loaded() == False:
        return

    print("\n--- PROFIT ANALYSIS ---")

    # -----------------------------------------------------------
    # FIRST: show which Region and which Category has the highest
    # and lowest PROFIT (not just Sales)
    # -----------------------------------------------------------
    region_profit = sales_data.groupby("Region")["Profit"].sum()
    best_profit_region = region_profit.idxmax()
    worst_profit_region = region_profit.idxmin()

    category_profit = sales_data.groupby("Category")["Profit"].sum()
    best_profit_category = category_profit.idxmax()
    worst_profit_category = category_profit.idxmin()

    print("Highest Profit Region  :", best_profit_region,
          "( Rs.", round(region_profit.max(), 2), ")")
    print("Lowest Profit Region   :", worst_profit_region,
          "( Rs.", round(region_profit.min(), 2), ")")
    print("Highest Profit Category:", best_profit_category,
          "( Rs.", round(category_profit.max(), 2), ")")
    print("Lowest Profit Category :", worst_profit_category,
          "( Rs.", round(category_profit.min(), 2), ")")

    add_line_to_report("Highest Profit Region: " + str(best_profit_region))
    add_line_to_report("Lowest Profit Region: " + str(worst_profit_region))
    add_line_to_report("Highest Profit Category: " + str(best_profit_category))
    add_line_to_report("Lowest Profit Category: " + str(worst_profit_category))

    total_profit = sales_data["Profit"].sum()
    average_profit = sales_data["Profit"].mean()

    # Calculate a simple profit margin percentage for the whole business
    total_sales = sales_data["Sales"].sum()
    overall_profit_margin = (total_profit / total_sales) * 100

    print("Total Profit         : Rs.", round(total_profit, 2))
    print("Average Profit/Order : Rs.", round(average_profit, 2))
    print("Overall Profit Margin:", round(overall_profit_margin, 2), "%")

    product_profit = sales_data.groupby("Product")["Profit"].sum()
    product_profit_sorted = product_profit.sort_values(ascending=False)

    print("\nTop 5 Most Profitable Products:")
    print(product_profit_sorted.head(5))

    print("\nBottom 5 Least Profitable Products:")
    print(product_profit_sorted.tail(5))

    add_line_to_report("Overall Profit Margin: " + str(round(overall_profit_margin, 2)) + "%")


#===================================================================
# SECTION 12 : MENU OPTION 10 - VISUAL GRAPH REPORTS
#===================================================================

def visual_graph_reports():
    # This function creates and saves five different types of graphs
    if check_dataset_is_loaded() == False:
        return

    print("\n--- VISUAL GRAPH REPORTS ---")

    # -------------------- BAR CHART --------------------
    # Bar chart is best for comparing totals between categories
    region_sales = sales_data.groupby("Region")["Sales"].sum()

    plt.figure(figsize=(8, 5))
    plt.bar(region_sales.index, region_sales.values, color="steelblue")
    plt.title("Total Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Sales")
    plt.savefig(graph_file_prefix + "bar_chart_region_sales.png")
    plt.close()
    print("Bar Chart saved: Total Sales by Region")

    # -------------------- LINE CHART --------------------
    # Line chart is best for showing a trend over time (months)
    monthly_sales = sales_data.groupby("Month")["Sales"].sum()

    plt.figure(figsize=(8, 5))
    plt.plot(monthly_sales.index, monthly_sales.values, marker="o", color="darkorange")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month Number")
    plt.ylabel("Total Sales")
    plt.savefig(graph_file_prefix + "line_chart_monthly_trend.png")
    plt.close()
    print("Line Chart saved: Monthly Sales Trend")

    # -------------------- PIE CHART --------------------
    # Pie chart is best for showing share/percentage of a whole
    category_sales = sales_data.groupby("Category")["Sales"].sum()

    plt.figure(figsize=(7, 7))
    plt.pie(category_sales.values, labels=category_sales.index, autopct="%1.1f%%")
    plt.title("Category wise Sales Share")
    plt.savefig(graph_file_prefix + "pie_chart_category_share.png")
    plt.close()
    print("Pie Chart saved: Category wise Sales Share")

    # -------------------- HISTOGRAM --------------------
    # Histogram is best for showing how order values are distributed
    plt.figure(figsize=(8, 5))
    plt.hist(sales_data["Sales"], bins=20, color="mediumseagreen")
    plt.title("Distribution of Sales Values")
    plt.xlabel("Sales Value")
    plt.ylabel("Number of Orders")
    plt.savefig(graph_file_prefix + "histogram_sales_distribution.png")
    plt.close()
    print("Histogram saved: Distribution of Sales Values")

    # -------------------- SCATTER PLOT --------------------
    # Scatter plot is best for showing relationship between two numbers
    plt.figure(figsize=(8, 5))
    plt.scatter(sales_data["Sales"], sales_data["Profit"], alpha=0.4, color="indianred")
    plt.title("Sales vs Profit Relationship")
    plt.xlabel("Sales")
    plt.ylabel("Profit")
    plt.savefig(graph_file_prefix + "scatter_sales_vs_profit.png")
    plt.close()
    print("Scatter Plot saved: Sales vs Profit Relationship")

    print("\nAll graphs saved successfully in the current folder,")
    print("as files starting with:", graph_file_prefix)


#===================================================================
# SECTION 13 : MENU OPTION 11 - MACHINE LEARNING ANALYSIS (TRAINING)
#===================================================================

def prepare_encoding_maps():
    # This function converts Region and Category text into numbers
    # Machine Learning models cannot understand text, only numbers
    global region_number_map
    global category_number_map

    unique_regions = sales_data["Region"].unique()
    unique_categories = sales_data["Category"].unique()

    region_number_map = {}
    counter = 0
    for region_name in unique_regions:
        region_number_map[region_name] = counter
        counter = counter + 1

    category_number_map = {}
    counter = 0
    for category_name in unique_categories:
        category_number_map[category_name] = counter
        counter = counter + 1


def machine_learning_analysis():
    # This function trains all Machine Learning models using historical data
    global sales_regression_model
    global profit_regression_model
    global region_cluster_model
    global performance_forest_model
    global models_trained

    if check_dataset_is_loaded() == False:
        return

    print("\n--- MACHINE LEARNING ANALYSIS (TRAINING MODELS) ---")

    # Step 1: Create numeric encoding for Region and Category
    prepare_encoding_maps()

    # We create new numeric columns using our encoding maps
    # map() replaces each text value with its matching number
    sales_data["Region_Code"] = sales_data["Region"].map(region_number_map)
    sales_data["Category_Code"] = sales_data["Category"].map(category_number_map)

    # -----------------------------------------------------------
    # MODEL 1 : LINEAR REGRESSION to predict SALES
    # -----------------------------------------------------------
    # Features (input columns) used to predict Sales
    input_features = sales_data[["Quantity", "Discount", "Region_Code", "Category_Code"]]
    target_sales = sales_data["Sales"]

    sales_regression_model = LinearRegression()
    # fit() is the training step, the model learns patterns from past data
    sales_regression_model.fit(input_features, target_sales)
    print("Linear Regression model trained to predict SALES.")

    # -----------------------------------------------------------
    # MODEL 2 : LINEAR REGRESSION to predict PROFIT
    # -----------------------------------------------------------
    target_profit = sales_data["Profit"]

    profit_regression_model = LinearRegression()
    profit_regression_model.fit(input_features, target_profit)
    print("Linear Regression model trained to predict PROFIT.")

    # -----------------------------------------------------------
    # MODEL 3 : KMEANS CLUSTERING for Region Performance Grouping
    # -----------------------------------------------------------
    # We group regions into 3 clusters: High, Medium, Low performance
    region_summary = sales_data.groupby("Region")[["Sales", "Profit"]].sum()

    region_cluster_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    region_cluster_model.fit(region_summary)

    region_summary["Cluster_Number"] = region_cluster_model.labels_
    print("\nKMeans Clustering Result (Region Performance Groups):")
    print(region_summary)

    # We label clusters as High/Medium/Low based on average Sales in that cluster
    cluster_average_sales = region_summary.groupby("Cluster_Number")["Sales"].mean()
    cluster_average_sales_sorted = cluster_average_sales.sort_values(ascending=False)

    cluster_label_map = {}
    cluster_names_in_order = ["High Performance", "Medium Performance", "Low Performance"]
    position = 0
    for cluster_number in cluster_average_sales_sorted.index:
        cluster_label_map[cluster_number] = cluster_names_in_order[position]
        position = position + 1

    print("\nRegion Performance Labels:")
    for region_name in region_summary.index:
        cluster_number_of_region = region_summary.loc[region_name, "Cluster_Number"]
        performance_label = cluster_label_map[cluster_number_of_region]
        print(region_name, "->", performance_label)

    # -----------------------------------------------------------
    # MODEL 4 : RANDOM FOREST to classify Order Performance
    # -----------------------------------------------------------
    # We create a simple target column: is this order "High Value" or "Low Value"?
    # We use the median Sales value as the cutoff point
    median_sales_value = sales_data["Sales"].median()

    # np.where() checks a condition and assigns one of two values
    sales_data["Performance_Label"] = np.where(
        sales_data["Sales"] >= median_sales_value, 1, 0
    )
    # Here 1 means "High Value Order" and 0 means "Low Value Order"

    forest_input_features = sales_data[["Quantity", "Discount", "Region_Code", "Category_Code"]]
    forest_target = sales_data["Performance_Label"]

    # A Random Forest builds many small decision trees (n_estimators=100
    # means 100 trees), each trained on a slightly different random slice
    # of the data, and then takes a majority vote of all trees' answers.
    # This usually gives a more reliable result than trusting one single
    # decision tree, while still being easy to explain: "many small trees
    # vote, and the majority answer wins".
    performance_forest_model = RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )
    performance_forest_model.fit(forest_input_features, forest_target)
    print("\nRandom Forest model trained to classify High/Low value orders.")

    models_trained = True
    print("\nAll Machine Learning models are now trained and ready for prediction.")
    add_line_to_report("Machine Learning models trained successfully on historical data.")


#===================================================================
# SECTION 14 : MENU OPTION 12 - PREDICT FUTURE SALES FROM NEW CSV
#===================================================================

def predict_future_sales_from_csv():
    # This function loads a NEW csv file and predicts Sales and Profit
    # using the models we already trained on historical data
    global models_trained

    if models_trained == False:
        print("\nPlease train the Machine Learning models first.")
        print("Choose Menu Option 11 (Machine Learning Analysis).\n")
        return

    print("\n--- PREDICT FUTURE SALES FROM NEW CSV ---")
    file_path = input("Enter the new CSV file name (example: future_sales_data.csv): ")

    # Clean up the typed path the same way we do in Load Dataset
    file_path = file_path.strip()
    file_path = file_path.strip('"')
    file_path = file_path.strip("'")

    # Try to read the file directly instead of using os.path.exists().
    # A FileNotFoundError is caught here instead of crashing the program.
    try:
        new_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found:", file_path)
        print("Make sure this CSV is in the same folder as the program,")
        print("or type the full file path.\n")
        return

    # This new/future file only needs these columns (no Sales/Profit,
    # since that is exactly what we are about to predict)
    required_future_columns = ["Region", "Category", "Product", "Quantity", "Discount"]
    missing_future_columns = []
    for column_name in required_future_columns:
        if column_name not in new_data.columns:
            missing_future_columns.append(column_name)

    if len(missing_future_columns) > 0:
        print("\nThis CSV file is missing the following required column(s):")
        print(missing_future_columns)
        print("Columns found in your file:", list(new_data.columns))
        print("")
        return

    # Convert Region and Category text into numbers using the SAME
    # encoding maps that were created during training
    new_data["Region_Code"] = new_data["Region"].map(region_number_map)
    new_data["Category_Code"] = new_data["Category"].map(category_number_map)

    # If a Region or Category was never seen during training, fillna()
    # replaces any missing encoded value with 0 so the program does not crash
    new_data["Region_Code"] = new_data["Region_Code"].fillna(0)
    new_data["Category_Code"] = new_data["Category_Code"].fillna(0)

    new_input_features = new_data[["Quantity", "Discount", "Region_Code", "Category_Code"]]

    # predict() uses the trained model to estimate output values
    predicted_sales = sales_regression_model.predict(new_input_features)
    predicted_profit = profit_regression_model.predict(new_input_features)
    predicted_performance = performance_forest_model.predict(new_input_features)

    new_data["Predicted_Sales"] = predicted_sales
    new_data["Predicted_Profit"] = predicted_profit
    new_data["Predicted_Performance"] = predicted_performance

    print("\nPrediction completed. Showing first 10 predicted rows:")
    print(new_data[["Region", "Category", "Product", "Predicted_Sales", "Predicted_Profit"]].head(10))

    # Business level summary from the predictions
    predicted_region_sales = new_data.groupby("Region")["Predicted_Sales"].sum()
    predicted_region_sales_sorted = predicted_region_sales.sort_values(ascending=False)

    predicted_category_sales = new_data.groupby("Category")["Predicted_Sales"].sum()
    predicted_category_sales_sorted = predicted_category_sales.sort_values(ascending=False)

    expected_best_region = predicted_region_sales.idxmax()
    expected_weak_region = predicted_region_sales.idxmin()
    expected_best_category = predicted_category_sales.idxmax()
    expected_weak_category = predicted_category_sales.idxmin()

    # ---------------------------------------------------------------
    # Find the top PRODUCT (and its Category) inside the best/weak
    # region, and the top PRODUCT inside the best/weak category, so
    # every prediction comes with a concrete product-level detail.
    # ---------------------------------------------------------------
    best_region_rows = new_data[new_data["Region"] == expected_best_region]
    best_region_product_sales = best_region_rows.groupby("Product")["Predicted_Sales"].sum()
    best_region_top_product = best_region_product_sales.idxmax()
    best_region_top_product_category = best_region_rows[
        best_region_rows["Product"] == best_region_top_product
    ]["Category"].iloc[0]

    weak_region_rows = new_data[new_data["Region"] == expected_weak_region]
    weak_region_product_sales = weak_region_rows.groupby("Product")["Predicted_Sales"].sum()
    weak_region_top_product = weak_region_product_sales.idxmin()
    weak_region_top_product_category = weak_region_rows[
        weak_region_rows["Product"] == weak_region_top_product
    ]["Category"].iloc[0]

    best_category_rows = new_data[new_data["Category"] == expected_best_category]
    best_category_product_sales = best_category_rows.groupby("Product")["Predicted_Sales"].sum()
    best_category_top_product = best_category_product_sales.idxmax()

    weak_category_rows = new_data[new_data["Category"] == expected_weak_category]
    weak_category_product_sales = weak_category_rows.groupby("Product")["Predicted_Sales"].sum()
    weak_category_top_product = weak_category_product_sales.idxmin()

    print("\nExpected High Performing Region:", expected_best_region)
    print("   -> Leading Product there   :", best_region_top_product,
          "(Category:", best_region_top_product_category, ")")
    print("Expected Weak Region            :", expected_weak_region)
    print("   -> Weakest Product there   :", weak_region_top_product,
          "(Category:", weak_region_top_product_category, ")")
    print("Expected High Selling Category  :", expected_best_category)
    print("   -> Leading Product in it   :", best_category_top_product)
    print("Expected Weak Category          :", expected_weak_category)
    print("   -> Weakest Product in it   :", weak_category_top_product)

    # Save the predictions into a new CSV file
    output_file_name = "predicted_future_sales.csv"
    new_data.to_csv(output_file_name, index=False)
    print("\nPredicted results saved to file:", output_file_name)

    # Create a simple bar graph of predicted sales by region
    plt.figure(figsize=(8, 5))
    plt.bar(predicted_region_sales_sorted.index, predicted_region_sales_sorted.values, color="purple")
    plt.title("Predicted Future Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Predicted Sales")
    plt.savefig(graph_file_prefix + "predicted_sales_by_region.png")
    plt.close()
    print("Graph saved: Predicted Future Sales by Region")

    add_line_to_report("Prediction from new CSV: Expected strong region = " +
                        str(expected_best_region) + ", Expected weak region = " +
                        str(expected_weak_region))

    # ---------------------------------------------------------------
    # Store this predicted DataFrame in a global variable so that
    # Business Suggestions (later menu option) can generate
    # recommendations from the NEW/predicted data too, not only the
    # historical data.
    # ---------------------------------------------------------------
    global predicted_data
    global predicted_data_available
    predicted_data = new_data
    predicted_data_available = True


#===================================================================
# SECTION 15 : MENU OPTION 13 - BUSINESS SUGGESTIONS
#===================================================================

def generate_suggestions_for_dataset(data_frame, sales_column_name, dataset_label):
    # This function contains the actual suggestion-building logic.
    # It works on ANY dataframe passed to it (historical OR predicted),
    # as long as it has Region, Category, Product and a sales-like column.
    print("\n" + "=" * 60)
    print("SUGGESTIONS BASED ON:", dataset_label)
    print("=" * 60)

    region_sales = data_frame.groupby("Region")[sales_column_name].sum()
    best_region = region_sales.idxmax()
    worst_region = region_sales.idxmin()

    category_sales = data_frame.groupby("Category")[sales_column_name].sum()
    best_category = category_sales.idxmax()
    worst_category = category_sales.idxmin()

    # These cross-breakdowns let us explain WHY a region/category is
    # strong or weak, instead of just stating the fact by itself
    region_category_sales = data_frame.groupby(["Region", "Category"])[sales_column_name].sum()
    region_product_sales = data_frame.groupby(["Region", "Product"])[sales_column_name].sum()
    category_product_sales = data_frame.groupby(["Category", "Product"])[sales_column_name].sum()

    best_region_top_category = region_category_sales.loc[best_region].idxmax()
    best_region_top_product = region_product_sales.loc[best_region].idxmax()

    worst_region_top_category = region_category_sales.loc[worst_region].idxmax()
    worst_region_weak_product = region_product_sales.loc[worst_region].idxmin()

    best_category_top_product = category_product_sales.loc[best_category].idxmax()
    worst_category_weak_product = category_product_sales.loc[worst_category].idxmin()

    print("-" * 60)
    print("Region", best_region, "has the highest", sales_column_name, "performance,")
    print("led mainly by the", best_region_top_category, "category and product '" +
          str(best_region_top_product) + "'.")
    print("Recommendation: Increase inventory and stock allocation in", best_region,
          "for", best_region_top_category, "products.")
    print("-" * 60)

    print("Region", worst_region, "has the weakest", sales_column_name, "performance.")
    print("Even its best-selling category there (", worst_region_top_category,
          ") lags behind other regions, and product '" + str(worst_region_weak_product) +
          "' is barely selling at all.")
    print("Recommendation: Increase marketing campaigns in", worst_region,
          ", focused on the", worst_region_top_category, "category.")
    print("-" * 60)

    print("Category", best_category, "generates the highest revenue overall,")
    print("driven strongly by product '" + str(best_category_top_product) + "'.")
    print("Recommendation: Expand this category with more products similar to '" +
          str(best_category_top_product) + "'.")
    print("-" * 60)

    print("Category", worst_category, "has poor performance,")
    print("with product '" + str(worst_category_weak_product) + "' being its weakest seller.")
    print("Recommendation: Review pricing strategy, or consider discontinuing weak")
    print("products like '" + str(worst_category_weak_product) + "'.")
    print("-" * 60)

    # Customer-level suggestion only if this dataframe actually has
    # customer information (the future/new CSV does not always need it)
    if "Customer_Name" in data_frame.columns:
        customer_sales = data_frame.groupby("Customer_Name")[sales_column_name].sum()
        best_customer = customer_sales.idxmax()
        total_sales_all = data_frame[sales_column_name].sum()
        best_customer_percentage = (customer_sales.max() / total_sales_all) * 100

        customer_region_sales = data_frame.groupby(["Customer_Name", "Region"])[sales_column_name].sum()
        customer_category_sales = data_frame.groupby(["Customer_Name", "Category"])[sales_column_name].sum()
        best_customer_top_region = customer_region_sales.loc[best_customer].idxmax()
        best_customer_top_category = customer_category_sales.loc[best_customer].idxmax()

        print("Customer", best_customer, "contributes",
              round(best_customer_percentage, 2), "% of total", sales_column_name + ",")
        print("mostly buying", best_customer_top_category, "products from the",
              best_customer_top_region, "region.")
        print("Recommendation: Create a loyalty program targeted at this customer profile.")
        print("-" * 60)

        add_line_to_report("[" + dataset_label + "] Loyalty program for customer " + str(best_customer))

    add_line_to_report("[" + dataset_label + "] Increase inventory in " + str(best_region))
    add_line_to_report("[" + dataset_label + "] Increase marketing in " + str(worst_region))
    add_line_to_report("[" + dataset_label + "] Expand category " + str(best_category))
    add_line_to_report("[" + dataset_label + "] Review pricing for category " + str(worst_category))


def business_suggestions():
    # This function lets the user choose whether suggestions should come
    # from the HISTORICAL dataset (Menu Option 1), the PREDICTED/FUTURE
    # dataset (Menu Option 12), or both - and clearly labels which is which.
    print("\n--- AUTOMATIC BUSINESS SUGGESTIONS ---")
    print("Which dataset should the suggestions be based on?")
    print("1. Historical Dataset (loaded in Menu Option 1)")
    print("2. Predicted / Future Dataset (created in Menu Option 12)")
    print("3. Both")

    dataset_choice = input("Enter your choice (1/2/3): ").strip()

    if dataset_choice == "1":
        if check_dataset_is_loaded() == False:
            return
        generate_suggestions_for_dataset(sales_data, "Sales", "HISTORICAL DATA")

    elif dataset_choice == "2":
        if predicted_data_available == False:
            print("\nNo predicted data available yet.")
            print("Please run Menu Option 12 (Predict Future Sales from New CSV) first.\n")
            return
        generate_suggestions_for_dataset(predicted_data, "Predicted_Sales", "PREDICTED / FUTURE DATA")

    elif dataset_choice == "3":
        if check_dataset_is_loaded() == True:
            generate_suggestions_for_dataset(sales_data, "Sales", "HISTORICAL DATA")
        else:
            print("\nHistorical dataset not loaded yet, skipping that part.")

        if predicted_data_available == True:
            generate_suggestions_for_dataset(predicted_data, "Predicted_Sales", "PREDICTED / FUTURE DATA")
        else:
            print("\nNo predicted data available yet (run Menu Option 12 first), skipping that part.")

    else:
        print("\nInvalid choice. Please enter 1, 2, or 3.\n")


#===================================================================
# SECTION 17 : MENU OPTION 15 - SAVE REPORT
#===================================================================

def save_report():
    # This function writes every collected insight into a text file
    print("\n--- SAVE REPORT ---")

    if len(report_lines) == 0:
        print("No report data available yet. Please run some analysis options first.\n")
        return

    report_file_name = "business_report.txt"

    # open() with mode "w" creates (or overwrites) a text file for writing
    report_file = open(report_file_name, "w")

    report_file.write("SALES ANALYTICS BUSINESS REPORT\n")
    report_file.write("=" * 40 + "\n\n")

    for single_line in report_lines:
        report_file.write(single_line + "\n")

    report_file.close()

    print("Report saved successfully as:", report_file_name)


#===================================================================
# SECTION 18 : MAIN MENU DISPLAY
#===================================================================

def print_main_menu():
    # This function prints the menu options on the screen
    print("\n===============================")
    print(" SALES ANALYTICS SYSTEM")
    print("===============================")
    print(" 1.  Load Dataset")
    print(" 2.  Dataset Information")
    print(" 3.  Overall Business Summary")
    print(" 4.  Product Analysis")
    print(" 5.  Category Analysis")
    print(" 6.  Region Analysis")
    print(" 7.  Customer Analysis")
    print(" 8.  Monthly Sales Trend")
    print(" 9.  Profit Analysis")
    print("10.  Visual Graph Reports")
    print("11.  Machine Learning Analysis")
    print("12.  Predict Future Sales from New CSV")
    print("13.  Business Suggestions")
    print("14.  Save Report")
    print("15.  Exit")
    print("===============================")


#===================================================================
# SECTION 19 : MAIN PROGRAM LOOP
#===================================================================

def main():
    # This is the starting point of the whole program
    # A while loop keeps showing the menu until the user chooses Exit
    program_running = True

    print("Welcome to the Sales Analytics and Business Decision Support System")

    while program_running == True:
        print_main_menu()
        user_choice = input("Enter your choice (1-15): ")

        if user_choice == "1":
            load_dataset()
        elif user_choice == "2":
            show_dataset_information()
        elif user_choice == "3":
            show_overall_business_summary()
        elif user_choice == "4":
            product_analysis()
        elif user_choice == "5":
            category_analysis()
        elif user_choice == "6":
            region_analysis()
        elif user_choice == "7":
            customer_analysis()
        elif user_choice == "8":
            monthly_sales_trend()
        elif user_choice == "9":
            profit_analysis()
        elif user_choice == "10":
            visual_graph_reports()
        elif user_choice == "11":
            machine_learning_analysis()
        elif user_choice == "12":
            predict_future_sales_from_csv()
        elif user_choice == "13":
            business_suggestions()
        elif user_choice == "14":
            save_report()
        elif user_choice == "15":
            print("\nThank you for using the Sales Analytics System. Goodbye!")
            program_running = False
        else:
            print("\nInvalid choice. Please enter a number between 1 and 15.\n")


# This special check makes sure main() only runs when this file is
# executed directly, not when it is imported into another file
if __name__ == "__main__":
    main()

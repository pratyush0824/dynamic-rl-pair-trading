# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")

# List of file names
files = ["btcusd", "ethusd", "xrpusd", "ltcusd", "ftmusd"]

# Define the start and end timestamps (in milliseconds)
start_ts = 1656633600000  # 1st July 2022
end_ts = 1688169600000  # 1st July 2023

def create_preprocessed_data():
    # Loop through each file to process
    for file in files:
        # Read the CSV file
        df = pd.read_csv(f"data/{file}.csv")

        # Convert the 'time' column from milliseconds to datetime
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        # Sort the DataFrame by time
        df = df.sort_values(by="time")

        # Filter the DataFrame to include only data within the desired period
        df = df[(df["time"] >= pd.to_datetime(start_ts, unit="ms")) &
                (df["time"] <= pd.to_datetime(end_ts, unit="ms"))]

        # Set the datetime as the index
        df.set_index("time", inplace=True)

        # Create a complete minute-wise date range for the period
        all_minutes = pd.date_range(start=df.index.min(), end=df.index.max(), freq="T")

        # Reindex the DataFrame so that every minute is represented
        df_full = df.reindex(all_minutes)

        # Forward-fill price columns ('open', 'high', 'low', 'close')
        price_cols = ['open', 'high', 'low', 'close']
        df_full[price_cols] = df_full[price_cols].ffill()

        # Fill missing volume values with 0
        df_full["volume"] = df_full["volume"].fillna(0)

        # Reset index to have 'time' as a column
        df_full.reset_index(inplace=True)
        df_full.rename(columns={"index": "time"}, inplace=True)

        # Save the preprocessed DataFrame to a new CSV file
        output_filename = f"data/{file}_preprocessed.csv"
        df_full.to_csv(output_filename, index=False)
        print(f"Preprocessed data saved to {output_filename}")

    print("All files have been processed and preprocessed!")

def verify_rows():
    row_counts = {}

    for file in files:
        # Read the filtered CSV file
        filtered_df = pd.read_csv(f"data/{file}_preprocessed.csv")

        # Get number of rows
        row_counts[file] = len(filtered_df)

    # Print row counts
    for file, count in row_counts.items():
        print(f"Filtered {file}_preprocessed.csv has {count} rows")

def create_merged_df():
    # Dictionary to store DataFrames
    dfs = {}

    # Read each CSV file and store in dictionary
    for file in files:
        df = pd.read_csv(f"data/{file}_preprocessed.csv")

        # Extract the crypto prefix (e.g., "btc" from "btcusd")
        prefix = file.replace("usd", "")

        # Create renaming dictionary
        rename_dict = {
            "open": f"open_{prefix}",
            "close": f"close_{prefix}",
            "high": f"high_{prefix}",
            "low": f"low_{prefix}",
            "volume": f"volume_{prefix}"
        }

        # Rename columns
        df.rename(columns=rename_dict, inplace=True)

        # Store in dictionary
        dfs[file] = df

    # Merge DataFrames on 'time' using an outer join
    merged_df = dfs[files[0]]  # Start with the first DataFrame
    for file in files[1:]:  # Merge remaining DataFrames
        merged_df = pd.merge(merged_df, dfs[file], on="time", how="outer")

    # 4. Convert 'time' to datetime
    merged_df["time"] = pd.to_datetime(merged_df["time"])

    # Sort by 'time' and reset the index
    merged_df.sort_values(by="time", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df

def get_correlation_matrix(merged_df):
    # Define the columns of interest
    price_cols = ["close_btc", "close_eth", "close_ltc", "close_ftm", "close_xrp"]

    # Compute correlation matrix
    corr_matrix = merged_df[price_cols].corr()

    print("Correlation matrix among closing prices:")
    return corr_matrix

def create_heat_map(corr_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm_r",  # Lighter colors
        vmin=-1, vmax=1,
        linewidths=0.5,  # Adds subtle gridlines
        cbar=True,
        center=0,
        annot_kws={"size": 10},  # Adjust annotation size for clarity
        fmt=".2f"  # Limits decimal places for readability
    )

    plt.gca().set_facecolor("white")
    plt.title("Correlation Heatmap of Closing Prices", fontsize=14)
    plt.show()

def create_plots(merged_df):
    # Define pairs of cryptos for plotting
    pairs = [
        ("btc", "eth"),
        ("btc", "ltc"),
        ("btc", "xrp"),
        ("eth", "ltc"),
        ("eth", "ftm")
    ]

    # Loop through each pair and create a dual y-axis plot
    for base, quote in pairs:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Primary Y-Axis (Left)
        ax1.set_xlabel("Time")
        ax1.set_ylabel(f"{base.upper()} Price", color="blue")
        ax1.plot(merged_df["time"], merged_df[f"close_{base}"], label=f"{base.upper()}", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Secondary Y-Axis (Right)
        ax2 = ax1.twinx()
        ax2.set_ylabel(f"{quote.upper()} Price", color="brown")
        ax2.plot(merged_df["time"], merged_df[f"close_{quote}"], label=f"{quote.upper()}", color="brown")
        ax2.tick_params(axis="y", labelcolor="brown")

        # Title and Labels
        plt.title(f"{base.upper()} and {quote.upper()} Prices")

        # Legend
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        # Show plot
        plt.show()



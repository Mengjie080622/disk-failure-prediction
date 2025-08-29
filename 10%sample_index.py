import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import duckdb

# --- Config ---
label_path = 'sampled_failure_label_weighted.csv'
log_folder = 'smartlog2019ssd'
output_path = 'sample_index.csv'
positive_lookback_days = 10
label_1_days = 7
neg_ratio = 2 

# --- Load failure label file ---
failure_df = pd.read_csv(label_path, parse_dates=['failure_time'])
failure_df['disk_id'] = failure_df['disk_id'].astype(str)
failure_df['failure_date'] = failure_df['failure_time'].dt.date
failure_records = failure_df.to_dict('records')

# --- Use DuckDB to read one disk from one CSV day ---
def get_log_filtered_by_disk(date_str, target_disk_id):
    file_path = os.path.join(log_folder, f"{date_str}.csv")
    if not os.path.exists(file_path):
        return None
    try:
        query = f"""
            SELECT * FROM read_csv_auto('{file_path}', AUTO_DETECT=TRUE)
            WHERE disk_id = '{target_disk_id}'
        """
        df = duckdb.query(query).to_df()
        return df
    except Exception as e:
        return None

# --- Extract positive samples per disk ---
def extract_positive(row):
    disk_id = row['disk_id']
    failure_date = row['failure_date']
    pos_list = []
    for offset in range(1, positive_lookback_days + 1):
        current_date = failure_date - timedelta(days=offset)
        date_str = current_date.strftime('%Y%m%d')
        df = get_log_filtered_by_disk(date_str, disk_id)
        if df is None or df.empty:
            continue

        if df.drop(columns=['model', 'disk_id'], errors='ignore').nunique().sum() == 0:
            continue

        df['model'] = df.get('model', 'Unknown')
        df['date'] = current_date
        df['label'] = 1 if offset <= label_1_days else 0
        pos_list.append(df[['disk_id', 'model', 'date', 'label']])
    return pos_list

# --- Main Function ---
def main():
    print("Extracting positive samples...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(extract_positive, failure_records), total=len(failure_records)))

    positive_samples = [s for sublist in results for s in sublist if s is not None]
    positive_df = pd.concat(positive_samples, ignore_index=True)
    positive_df.drop_duplicates(subset=['disk_id', 'date'], inplace=True)

    positive_ids = set(positive_df['disk_id'].unique())
    positive_dates = set(zip(positive_df['disk_id'], pd.to_datetime(positive_df['date']).dt.date))

    print("Extracting negative samples...")
    log_files = sorted([f for f in os.listdir(log_folder) if f.endswith('.csv')])
    seen_disks = set()
    sampled_negatives = []

    for file_name in tqdm(log_files, desc="Scanning logs"):
        date_str = file_name.replace('.csv', '')
        file_path = os.path.join(log_folder, file_name)
        try:
            query = f"""
                SELECT disk_id, model FROM read_csv_auto('{file_path}', AUTO_DETECT=TRUE)
            """
            df = duckdb.query(query).to_df()
            df['disk_id'] = df['disk_id'].astype(str)
            df['date'] = pd.to_datetime(date_str, format='%Y%m%d').date()

            # Exclude the (disk_id, date) that has appeared in the positive sample.
            df = df[~df['disk_id'].isin(positive_ids)]
            df = df[~df.apply(lambda row: (row['disk_id'], row['date']) in positive_dates, axis=1)]

            for _, row in df.iterrows():
                key = (row['disk_id'])
                if key in seen_disks:
                    continue
                seen_disks.add(key)
                sampled_negatives.append({
                    'disk_id': row['disk_id'],
                    'model': row['model'],
                    'date': row['date'],
                    'label': 0
                })

        except:
            continue

    neg_df_all = pd.DataFrame(sampled_negatives)

    # Negative sample size control
    num_pos = int((positive_df['label'] == 1).sum())  
    sampled_neg_df = neg_df_all.sample(
             n=min(num_pos * neg_ratio, len(neg_df_all)), random_state=42
    )




    # Save
    all_df = pd.concat([positive_df, sampled_neg_df], ignore_index=True)
    all_df = all_df.sort_values(by=['disk_id', 'date'])
    all_df[['disk_id', 'model', 'date', 'label']].to_csv(output_path, index=False)

    print(f"\nDone. Total samples: {len(all_df)} (Positive: {len(positive_df)}, Negative: {len(sampled_neg_df)}).")
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    main()

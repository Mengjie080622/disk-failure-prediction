import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ==== Configuration ====
raw_log_dir = 'smartlog2019ssd'
cleaned_log_dir = 'cleaned_logs'
sample_index_path = 'sample_index.csv'
chunk_size = 50000
missing_threshold = 0.5  # Drop columns with >50% missing

# === Load sample index ===
sample_index = pd.read_csv(sample_index_path, parse_dates=['date'])
sample_index['disk_id'] = sample_index['disk_id'].astype(str)
sample_index['date_str'] = sample_index['date'].dt.strftime('%Y%m%d')
dates_to_process = sorted(sample_index['date_str'].unique())
os.makedirs(cleaned_log_dir, exist_ok=True)

# === Clean a single day's log ===
def clean_single_log(date_str):
    filename = f"{date_str}.csv"
    file_path = os.path.join(raw_log_dir, filename)
    output_path = os.path.join(cleaned_log_dir, filename)

    if not os.path.exists(file_path):
        return f"{filename} not found."

    try:
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        cleaned_chunks = []

        for chunk in chunks:
            chunk['disk_id'] = chunk['disk_id'].astype(str)
            relevant_ids = sample_index[sample_index['date_str'] == date_str]['disk_id'].unique()
            chunk = chunk[chunk['disk_id'].isin(relevant_ids)]

            if chunk.empty:
                continue

            # Drop columns with too many missing values
            chunk = chunk.loc[:, chunk.isnull().mean() < missing_threshold]


            # Remove duplicates
            chunk = chunk.drop_duplicates()

            # Remove negative numeric values
            num_cols = chunk.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                chunk = chunk[chunk[col] >= 0]

            cleaned_chunks.append(chunk)

        if cleaned_chunks:
            final_df = pd.concat(cleaned_chunks, ignore_index=True)
            final_df.to_csv(output_path, index=False)
            return f"{filename} cleaned, {len(final_df)} rows"
        else:
            return f"{filename} has no relevant data."

    except Exception as e:
        return f"{filename} failed: {str(e)}"

# === Main block ===
if __name__ == '__main__':
    print("Starting multi-process SMART log cleaning...")

    num_workers = min(cpu_count(), 8)
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(clean_single_log, dates_to_process), total=len(dates_to_process), desc="Processing logs"))

    for r in results[:10]:
        print(r)

    print("All cleaning tasks completed.")

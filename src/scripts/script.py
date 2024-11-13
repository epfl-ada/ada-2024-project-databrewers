import pandas as pd

def format_data(data, id_type="review_id", chunk_size=1600):
    processed_data = []  # Temporary list to hold processed data

    for start in range(0, len(data), chunk_size):
        # Extract a chunk of data
        chunk = data.iloc[start:start + chunk_size].copy()

        # Add id_type based on the 16-row structure
        chunk[id_type] = chunk.index // 16

        # Split key-value pairs
        split_data = chunk['info'].str.split(': ', n=1, expand=True)
        chunk = chunk[split_data[1].notna()]  # Keep rows with valid key-value pairs

        # Assign key and value columns
        chunk[['key', 'value']] = split_data

        # Pivot the chunk to convert key to columns
        chunk_pivoted = chunk.pivot(index=id_type, columns='key', values='value').reset_index(drop=True)
        processed_data.append(chunk_pivoted)

    processed_data = pd.concat(processed_data, ignore_index=True)

    return processed_data
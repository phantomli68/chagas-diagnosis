import os
import shutil
import csv
from collections import Counter
import random
from tqdm import tqdm
import re
def organize_and_rename_files(original_data_folder='original_data', target_data_folder='data'):
    """
    Copies .dat and .hea files from PTB-XL and SaMi-Trop folders to a new data folder,
    renaming them according to the specified format (prefix 'p'/'s' + 7-digit zero-padded original filename).
    """
    os.makedirs(target_data_folder, exist_ok=True)
    # Process PTB-XL files
    ptb_xl_path = os.path.join(original_data_folder, 'PTB-XL')
    if os.path.exists(ptb_xl_path):
        print(f"Processing files from {ptb_xl_path}...")
        for filename in os.listdir(ptb_xl_path):
            if filename.endswith('.dat') or filename.endswith('.hea'):
                original_filepath = os.path.join(ptb_xl_path, filename)

                base_name, ext = os.path.splitext(filename)
                try:
                    padded_base_name = f"{int(base_name):07d}"
                except ValueError:

                    padded_base_name = base_name.zfill(7)

                new_filename = f"p{padded_base_name}{ext}"

                destination_filepath = os.path.join(target_data_folder, new_filename)
                shutil.copy2(original_filepath, destination_filepath)
    else:
        print(f"Warning: {ptb_xl_path} not found. Skipping PTB-XL file processing.")
    # Process SaMi-Trop files
    sami_trop_path = os.path.join(original_data_folder, 'SaMi-Trop')
    if os.path.exists(sami_trop_path):
        print(f"Processing files from {sami_trop_path}...")
        for filename in os.listdir(sami_trop_path):
            if filename.endswith('.dat') or filename.endswith('.hea'):
                original_filepath = os.path.join(sami_trop_path, filename)

                base_name, ext = os.path.splitext(filename)

                try:
                    padded_base_name = f"{int(base_name):07d}"
                except ValueError:
                    padded_base_name = base_name.zfill(7)

                new_filename = f"s{padded_base_name}{ext}"

                destination_filepath = os.path.join(target_data_folder, new_filename)
                shutil.copy2(original_filepath, destination_filepath)
    else:
        print(f"Warning: {sami_trop_path} not found. Skipping SaMi-Trop file processing.")

    print(f"\nAll files processed and moved to '{target_data_folder}' folder.")

def update_hea_filenames(directory):
    for fname in os.listdir(directory):
        if fname.endswith('.hea'):
            file_path = os.path.join(directory, fname)
            new_id = os.path.splitext(fname)[0]

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            updated_lines = []
            for i, line in enumerate(lines):
                if i == 0:
                    line = re.sub(r'^\S+', new_id, line)
                elif line.strip().endswith(tuple(['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])):
                    line = re.sub(r'^\S+\.dat', f'{new_id}.dat', line)
                updated_lines.append(line)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            print(f'Updated: {fname}')

def gen_label_csv(data_folder, output_csv):
    csv_headers = ['id', 'Age', 'Sex', 'label', 'Source', 'fold']
    records = []
    label_counter = Counter()

    hea_files = [f for f in os.listdir(data_folder) if f.endswith('.hea')]

    for filename in tqdm(hea_files, desc="Processing .hea files"):
        file_path = os.path.join(data_folder, filename)
        record_id = filename.replace('.hea', '')
        age = ''
        sex = ''
        label = ''
        source = ''

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# Age:'):
                    age = line.replace('# Age:', '').strip()
                elif line.startswith('# Sex:'):
                    sex = line.replace('# Sex:', '').strip()
                elif line.startswith('# Chagas label:'):
                    label_str = line.replace('# Chagas label:', '').strip()
                    label = '1' if label_str.lower() == 'true' else '0'
                elif line.startswith('# Source:'):
                    source = line.replace('# Source:', '').strip()

        records.append([record_id, age, sex, label, source])
        label_counter[label] += 1

    random.shuffle(records)

    n = len(records)
    fold_sizes = [n // 10] * 10
    for i in range(n % 10):
        fold_sizes[i] += 1

    new_records = []
    idx = 0
    for fold_id, size in enumerate(fold_sizes):
        for _ in range(size):
            rec = records[idx]
            rec_with_fold = rec + [fold_id]
            new_records.append(rec_with_fold)
            idx += 1

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(new_records)

    print(f'\nDone. Saved {len(records)} records to {output_csv}')
    print("Label counts:")
    for lbl, count in label_counter.items():
        print(f"  label={lbl}: {count} samples")

if __name__ == "__main__":
    original_data_folder = 'original_data'
    target_data_folder = 'data'
    output_csv_file = os.path.join(target_data_folder, 'label.csv')

    # Step 1: Organize and rename files
    organize_and_rename_files(original_data_folder, target_data_folder)

    # Step 2: Update .hea filenames
    update_hea_filenames(target_data_folder)

    # Step 2: Generate label.csv
    gen_label_csv(target_data_folder, output_csv_file)
#!/usr/bin/env python3
"""
Download datasets for semi-synthetic training.

Downloads from UCI ML Repository and Kaggle, saves to /mnt/data/lacuna/raw/

Usage:
    pip install ucimlrepo kaggle
    # For Kaggle: set up ~/.kaggle/kaggle.json with your API key
    python scripts/download_datasets.py
    python scripts/download_datasets.py --skip-kaggle  # UCI only
    python scripts/download_datasets.py --skip-uci     # Kaggle only
"""

import argparse
import sys
import os
import zipfile
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

# Target directory
RAW_DIR = Path("/mnt/data/lacuna/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_uci_datasets():
    """Download datasets from UCI ML Repository."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("Please install ucimlrepo: pip install ucimlrepo")
        return [], []
    
    # UCI dataset IDs - all numeric, complete or mostly complete
    uci_datasets = [
        # (name, id, description)
        ("wine_quality_red", 186, "Wine quality (red) - 1599 samples, 11 features"),
        ("parkinsons", 174, "Parkinsons - 195 samples, 22 features"),
        ("glass", 42, "Glass identification - 214 samples, 9 features"),
        ("ionosphere", 52, "Ionosphere - 351 samples, 34 features"),
        ("vehicle", 149, "Vehicle silhouettes - 846 samples, 18 features"),
        ("segmentation", 50, "Image segmentation - 2310 samples, 19 features"),
        ("spambase", 94, "Spambase - 4601 samples, 57 features"),
        ("banknote", 267, "Banknote authentication - 1372 samples, 4 features"),
        ("yeast", 110, "Yeast - 1484 samples, 8 features"),
        ("ecoli", 39, "E. coli - 336 samples, 7 features"),
        ("letter_recognition", 59, "Letter recognition - 20000 samples, 16 features"),
        ("optical_digits", 80, "Optical digits - 5620 samples, 64 features"),
        ("pendigits", 81, "Pen digits - 10992 samples, 16 features"),
        ("satellite", 146, "Satellite (Statlog) - 6435 samples, 36 features"),
        ("page_blocks", 78, "Page blocks - 5473 samples, 10 features"),
        ("steel_plates", 198, "Steel plates faults - 1941 samples, 27 features"),
        ("cardiotocography", 193, "Cardiotocography - 2126 samples, 21 features"),
        ("magic_telescope", 159, "MAGIC telescope - 19020 samples, 10 features"),
        # Replacements for seeds (236) and waveform (108) which aren't available via API
        ("wifi_localization", 422, "WiFi localization - 2000 samples, 7 features"),
        ("concrete", 165, "Concrete compressive strength - 1030 samples, 8 features"),
    ]
    
    successful = []
    failed = []
    
    for name, dataset_id, description in uci_datasets:
        print(f"\nDownloading: {name} (ID={dataset_id})")
        print(f"  {description}")
        
        try:
            dataset = fetch_ucirepo(id=dataset_id)
            
            # Get features (X) - we only want numeric columns
            X = dataset.data.features
            
            # Convert to numeric, coercing errors
            X_numeric = X.select_dtypes(include=[np.number])
            
            if X_numeric.shape[1] == 0:
                print(f"  SKIPPED: No numeric columns")
                failed.append((name, "No numeric columns"))
                continue
            
            # Check for missing values
            missing_pct = X_numeric.isna().sum().sum() / X_numeric.size * 100
            
            if missing_pct > 5:
                print(f"  SKIPPED: {missing_pct:.1f}% missing values")
                failed.append((name, f"{missing_pct:.1f}% missing"))
                continue
            
            # Drop rows with any missing values (for complete data)
            X_clean = X_numeric.dropna()
            
            n_dropped = len(X_numeric) - len(X_clean)
            if n_dropped > 0:
                print(f"  Dropped {n_dropped} rows with missing values")
            
            # Save to CSV
            output_path = RAW_DIR / f"{name}.csv"
            X_clean.to_csv(output_path, index=False)
            
            print(f"  SAVED: {output_path}")
            print(f"  Shape: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
            
            successful.append((name, X_clean.shape))
            
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((name, str(e)))
    
    return successful, failed


def download_kaggle_datasets():
    """Download datasets from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Please install kaggle: pip install kaggle")
        print("And set up credentials via KAGGLE_API_TOKEN env var or ~/.kaggle/kaggle.json")
        print("See: https://www.kaggle.com/docs/api")
        return [], []
    
    # Check for credentials - support both env var and JSON file
    kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if kaggle_token:
        # Create kaggle.json from env var if it doesn't exist
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        if not kaggle_json.exists():
            # Token format is the full JSON string
            kaggle_json.write_text(kaggle_token)
            kaggle_json.chmod(0o600)
            print(f"Created {kaggle_json} from KAGGLE_API_TOKEN")
    
    if not kaggle_json.exists() and not kaggle_token:
        print("Kaggle credentials not found.")
        print("Either set KAGGLE_API_TOKEN environment variable:")
        print('  export KAGGLE_API_TOKEN=\'{"username":"...","key":"..."}\'')
        print(f"Or save your API key to {kaggle_json}")
        return [], []
    
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    
    # Kaggle datasets - (name, kaggle_path, csv_filename_in_zip, description)
    # csv_filename_in_zip can be None to auto-detect, or a specific filename
    kaggle_datasets = [
        (
            "dry_bean",
            "muratkokludataset/dry-bean-dataset",
            "Dry_Bean_Dataset.csv",
            "Dry Bean - 13611 samples, 16 features"
        ),
        (
            "rice",
            "muratkokludataset/rice-cammeo-and-osmancik",
            "Rice_Cammeo_Osmancik.csv",
            "Rice varieties - 3810 samples, 7 features"
        ),
        (
            "raisin",
            "muratkokludataset/raisin-dataset",
            "Raisin_Dataset.csv",
            "Raisin - 900 samples, 7 features"
        ),
        (
            "pumpkin_seeds",
            "muratkokludataset/pumpkin-seeds-dataset",
            "Pumpkin_Seeds_Dataset.csv",
            "Pumpkin seeds - 2500 samples, 12 features"
        ),
        (
            "pulsar_stars",
            "colearninglounge/predicting-pulsar-starintermediate",
            None,  # Auto-detect
            "Pulsar stars (HTRU2) - 17898 samples, 8 features"
        ),
        (
            "heart_disease",
            "cherngs/heart-disease-cleveland-uci",
            None,
            "Heart disease Cleveland - 303 samples, 13 features"
        ),
        (
            "credit_card_default",
            "uciml/default-of-credit-card-clients-dataset",
            None,
            "Credit card default - 30000 samples, 23 features"
        ),
        (
            "electrical_grid",
            "pcbreviern/smart-grid-stability",
            None,
            "Electrical grid stability - 10000 samples, 12 features"
        ),
        (
            "air_quality",
            "fedesoriano/air-quality-data-set",
            None,
            "Air quality - 9358 samples, 13 features"
        ),
        (
            "room_occupancy",
            "robmarkcole/occupancy-detection-data-set-uci",
            None,
            "Room occupancy - 20560 samples, 5 features"
        ),
        (
            "avocado_prices",
            "neuromusic/avocado-prices",
            None,
            "Avocado prices - 18249 samples, numeric features"
        ),
        (
            "superconductor",
            "munumbutt/superconductor-dataset",
            None,
            "Superconductor critical temp - 21263 samples, 81 features"
        ),
        (
            "bike_sharing",
            "lakshmi25npathi/bike-sharing-dataset",
            "hour.csv",
            "Bike sharing hourly - 17379 samples, numeric features"
        ),
    ]
    
    successful = []
    failed = []
    
    for name, kaggle_path, csv_filename, description in kaggle_datasets:
        print(f"\nDownloading: {name}")
        print(f"  Kaggle: {kaggle_path}")
        print(f"  {description}")
        
        try:
            # Download to temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                api.dataset_download_files(kaggle_path, path=tmpdir, unzip=True)
                
                # Find CSV file
                csv_files = list(Path(tmpdir).rglob("*.csv"))
                
                if len(csv_files) == 0:
                    # Try xlsx
                    xlsx_files = list(Path(tmpdir).rglob("*.xlsx"))
                    if xlsx_files:
                        print(f"  Found Excel file, converting...")
                        df = pd.read_excel(xlsx_files[0])
                    else:
                        print(f"  SKIPPED: No CSV or Excel files found")
                        failed.append((name, "No data files found"))
                        continue
                elif csv_filename:
                    # Use specified filename
                    matches = [f for f in csv_files if f.name == csv_filename]
                    if matches:
                        df = pd.read_csv(matches[0])
                    else:
                        # Try partial match
                        matches = [f for f in csv_files if csv_filename.lower() in f.name.lower()]
                        if matches:
                            df = pd.read_csv(matches[0])
                        else:
                            print(f"  SKIPPED: {csv_filename} not found")
                            print(f"  Available: {[f.name for f in csv_files]}")
                            failed.append((name, f"{csv_filename} not found"))
                            continue
                else:
                    # Auto-detect: use largest CSV
                    csv_files.sort(key=lambda f: f.stat().st_size, reverse=True)
                    df = pd.read_csv(csv_files[0])
                    print(f"  Using: {csv_files[0].name}")
                
                # Select only numeric columns
                df_numeric = df.select_dtypes(include=[np.number])
                
                if df_numeric.shape[1] == 0:
                    print(f"  SKIPPED: No numeric columns")
                    failed.append((name, "No numeric columns"))
                    continue
                
                # Check for missing values
                missing_pct = df_numeric.isna().sum().sum() / df_numeric.size * 100
                
                if missing_pct > 10:
                    print(f"  SKIPPED: {missing_pct:.1f}% missing values")
                    failed.append((name, f"{missing_pct:.1f}% missing"))
                    continue
                
                # Drop rows with missing values
                df_clean = df_numeric.dropna()
                
                n_dropped = len(df_numeric) - len(df_clean)
                if n_dropped > 0:
                    print(f"  Dropped {n_dropped} rows with missing values")
                
                # Skip if too small
                if df_clean.shape[0] < 100:
                    print(f"  SKIPPED: Only {df_clean.shape[0]} samples after cleaning")
                    failed.append((name, f"Only {df_clean.shape[0]} samples"))
                    continue
                
                # Save to CSV
                output_path = RAW_DIR / f"{name}.csv"
                df_clean.to_csv(output_path, index=False)
                
                print(f"  SAVED: {output_path}")
                print(f"  Shape: {df_clean.shape[0]} samples, {df_clean.shape[1]} features")
                
                successful.append((name, df_clean.shape))
                
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((name, str(e)[:50]))
    
    return successful, failed


def download_from_urls():
    """Download datasets from direct URLs."""
    
    url_datasets = [
        # (name, url, description, has_header)
        (
            "abalone",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
            "Abalone - 4177 samples, 7 numeric features",
            False
        ),
        (
            "wine_white",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
            "Wine quality (white) - 4898 samples, 11 features",
            True
        ),
    ]
    
    successful = []
    failed = []
    
    for name, url, description, has_header in url_datasets:
        print(f"\nDownloading: {name}")
        print(f"  {description}")
        
        try:
            header = 0 if has_header else None
            sep = ';' if 'winequality' in url else ','
            df = pd.read_csv(url, header=header, sep=sep)
            
            # Select only numeric columns
            df_numeric = df.select_dtypes(include=[np.number])
            
            if df_numeric.shape[1] == 0:
                print(f"  SKIPPED: No numeric columns")
                failed.append((name, "No numeric columns"))
                continue
            
            # Drop missing
            df_clean = df_numeric.dropna()
            
            # Add column names if needed
            if not has_header:
                df_clean.columns = [f"feature_{i}" for i in range(df_clean.shape[1])]
            
            # Save
            output_path = RAW_DIR / f"{name}.csv"
            df_clean.to_csv(output_path, index=False)
            
            print(f"  SAVED: {output_path}")
            print(f"  Shape: {df_clean.shape[0]} samples, {df_clean.shape[1]} features")
            
            successful.append((name, df_clean.shape))
            
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((name, str(e)))
    
    return successful, failed


def print_summary(all_successful, all_failed):
    """Print download summary."""
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    print(f"\nSuccessfully downloaded: {len(all_successful)}")
    
    # Group by size
    small = [(n, s) for n, s in all_successful if s[0] < 1000]
    medium = [(n, s) for n, s in all_successful if 1000 <= s[0] < 10000]
    large = [(n, s) for n, s in all_successful if s[0] >= 10000]
    
    if small:
        print(f"\n  Small (<1000 samples): {len(small)}")
        for name, shape in sorted(small, key=lambda x: x[1][0]):
            print(f"    {name}: {shape[0]} x {shape[1]}")
    
    if medium:
        print(f"\n  Medium (1000-10000 samples): {len(medium)}")
        for name, shape in sorted(medium, key=lambda x: x[1][0]):
            print(f"    {name}: {shape[0]} x {shape[1]}")
    
    if large:
        print(f"\n  Large (>10000 samples): {len(large)}")
        for name, shape in sorted(large, key=lambda x: x[1][0]):
            print(f"    {name}: {shape[0]} x {shape[1]}")
    
    if all_failed:
        print(f"\nFailed: {len(all_failed)}")
        for name, reason in all_failed:
            print(f"  {name}: {reason}")
    
    # Calculate totals
    total_samples = sum(s[0] for _, s in all_successful)
    feature_range = (
        min(s[1] for _, s in all_successful) if all_successful else 0,
        max(s[1] for _, s in all_successful) if all_successful else 0
    )
    
    print(f"\n" + "-" * 40)
    print(f"Total datasets: {len(all_successful)}")
    print(f"Total samples: {total_samples:,}")
    print(f"Feature range: {feature_range[0]} - {feature_range[1]}")
    
    # List files in directory
    print(f"\nFiles in {RAW_DIR}:")
    total_size = 0
    for f in sorted(RAW_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        total_size += size_kb
        print(f"  {f.name} ({size_kb:.1f} KB)")
    print(f"\nTotal size: {total_size/1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Project Lacuna")
    parser.add_argument("--skip-uci", action="store_true", help="Skip UCI downloads")
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip Kaggle downloads")
    parser.add_argument("--skip-urls", action="store_true", help="Skip URL downloads")
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATASET DOWNLOADER FOR PROJECT LACUNA")
    print("=" * 60)
    print(f"\nTarget directory: {RAW_DIR}")
    
    all_successful = []
    all_failed = []
    
    # Download from UCI
    if not args.skip_uci:
        print("\n" + "-" * 40)
        print("Downloading from UCI ML Repository...")
        print("-" * 40)
        successful, failed = download_uci_datasets()
        all_successful.extend(successful)
        all_failed.extend(failed)
    
    # Download from Kaggle
    if not args.skip_kaggle:
        print("\n" + "-" * 40)
        print("Downloading from Kaggle...")
        print("-" * 40)
        successful, failed = download_kaggle_datasets()
        all_successful.extend(successful)
        all_failed.extend(failed)
    
    # Download from URLs
    if not args.skip_urls:
        print("\n" + "-" * 40)
        print("Downloading from URLs...")
        print("-" * 40)
        successful, failed = download_from_urls()
        all_successful.extend(successful)
        all_failed.extend(failed)
    
    # Summary
    print_summary(all_successful, all_failed)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. The catalog will auto-discover CSVs when you call:
   catalog = create_default_catalog()
   catalog.scan_directory()

2. Update your config to use more datasets:
   train_datasets:
     - wine_quality_red
     - dry_bean
     - pendigits
     - ...

3. Or use all available datasets programmatically:
   catalog = create_default_catalog()
   all_datasets = catalog.list_datasets()
""")
    
    print("Done!")


if __name__ == "__main__":
    main()
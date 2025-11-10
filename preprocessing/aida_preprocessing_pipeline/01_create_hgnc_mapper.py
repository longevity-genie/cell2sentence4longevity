#!/usr/bin/env python3
"""
Step 1: Download HGNC gene mapping data and create mappers.

This script downloads official gene symbol mappings from HGNC to convert
Ensembl IDs to human-readable gene symbols.

Outputs:
- hgnc_mappers.pkl: Pickle file with bidirectional mappings
"""

import requests
import pandas as pd
import pickle
import io

def download_hgnc_data():
    """Download HGNC complete dataset."""
    print("📥 Downloading HGNC complete gene set...")

    urls = [
        "https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt",
        "http://ftp.ebi.ac.uk/pub/databases/genenames/new/tsv/hgnc_complete_set.txt",
    ]

    for i, url in enumerate(urls, 1):
        print(f"\n  Attempt {i}/{len(urls)}")
        print(f"  URL: {url}")

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            hgnc_df = pd.read_csv(io.StringIO(response.text), sep='\t')
            print(f"✅ Downloaded HGNC data: {len(hgnc_df):,} genes")

            hgnc_df.to_csv("hgnc_complete_set.tsv", sep='\t', index=False)
            print(f"💾 Saved to: hgnc_complete_set.tsv")

            return hgnc_df

        except Exception as e:
            print(f"❌ Failed: {e}")
            continue

    # Check if local file exists
    print("\n⚠️  All download attempts failed. Checking for local file...")
    try:
        hgnc_df = pd.read_csv("hgnc_complete_set.tsv", sep='\t')
        print(f"✅ Loaded from local file: {len(hgnc_df):,} genes")
        return hgnc_df
    except:
        print("❌ No local file found.")
        return None

def create_mappers(hgnc_df):
    """Create mapping dictionaries from HGNC data."""
    print(f"\n{'='*80}")
    print("CREATING GENE MAPPERS")
    print(f"{'='*80}\n")

    mappers = {}

    # Normalize column names (handle different HGNC formats)
    col_map = {
        'Ensembl gene ID': 'ensembl_gene_id',
        'Approved symbol': 'symbol',
        'Previous symbols': 'prev_symbol',
        'Alias symbols': 'alias_symbol'
    }

    for old_name, new_name in col_map.items():
        if old_name in hgnc_df.columns:
            hgnc_df[new_name] = hgnc_df[old_name]

    # 1. Ensembl ID -> Official Symbol
    print("1️⃣  Ensembl ID -> Official Symbol")
    ensembl_to_symbol = {}
    for _, row in hgnc_df.iterrows():
        if pd.notna(row.get('ensembl_gene_id')) and pd.notna(row.get('symbol')):
            ensembl_to_symbol[row['ensembl_gene_id']] = row['symbol']
    print(f"   Created {len(ensembl_to_symbol):,} mappings")
    mappers['ensembl_to_symbol'] = ensembl_to_symbol

    # 2. Symbol -> Ensembl ID
    print("2️⃣  Official Symbol -> Ensembl ID")
    symbol_to_ensembl = {}
    for _, row in hgnc_df.iterrows():
        if pd.notna(row.get('symbol')) and pd.notna(row.get('ensembl_gene_id')):
            symbol_to_ensembl[row['symbol']] = row['ensembl_gene_id']
    print(f"   Created {len(symbol_to_ensembl):,} mappings")
    mappers['symbol_to_ensembl'] = symbol_to_ensembl

    # 3. Previous symbols -> Official Symbol
    print("3️⃣  Previous Symbols -> Official Symbol")
    prev_to_symbol = {}
    for _, row in hgnc_df.iterrows():
        if pd.notna(row.get('prev_symbol')) and pd.notna(row.get('symbol')):
            prev_symbols = str(row['prev_symbol']).split(',')
            if len(prev_symbols) == 1:
                prev_symbols = str(row['prev_symbol']).split('|')
            for prev_sym in prev_symbols:
                prev_sym = prev_sym.strip()
                if prev_sym:
                    prev_to_symbol[prev_sym] = row['symbol']
    print(f"   Created {len(prev_to_symbol):,} mappings")
    mappers['prev_to_symbol'] = prev_to_symbol

    # 4. Alias symbols -> Official Symbol
    print("4️⃣  Alias Symbols -> Official Symbol")
    alias_to_symbol = {}
    for _, row in hgnc_df.iterrows():
        if pd.notna(row.get('alias_symbol')) and pd.notna(row.get('symbol')):
            alias_symbols = str(row['alias_symbol']).split(',')
            if len(alias_symbols) == 1:
                alias_symbols = str(row['alias_symbol']).split('|')
            for alias_sym in alias_symbols:
                alias_sym = alias_sym.strip()
                if alias_sym:
                    alias_to_symbol[alias_sym] = row['symbol']
    print(f"   Created {len(alias_to_symbol):,} mappings")
    mappers['alias_to_symbol'] = alias_to_symbol

    # 5. Combined mapper: any symbol -> official symbol
    print("5️⃣  Combined: Any Symbol -> Official Symbol")
    any_to_symbol = {}
    for sym in symbol_to_ensembl.keys():
        any_to_symbol[sym] = sym
    any_to_symbol.update(prev_to_symbol)
    any_to_symbol.update(alias_to_symbol)
    print(f"   Created {len(any_to_symbol):,} mappings")
    mappers['any_to_symbol'] = any_to_symbol

    return mappers

def save_mappers(mappers):
    """Save mappers to pickle file."""
    print(f"\n{'='*80}")
    print("SAVING MAPPERS")
    print(f"{'='*80}\n")

    with open('hgnc_mappers.pkl', 'wb') as f:
        pickle.dump(mappers, f)
    print("💾 Saved all mappers to: hgnc_mappers.pkl")

if __name__ == "__main__":
    print('='*80)
    print('STEP 1: CREATE HGNC MAPPER')
    print('='*80)

    hgnc_df = download_hgnc_data()

    if hgnc_df is not None:
        mappers = create_mappers(hgnc_df)
        save_mappers(mappers)
        print(f"\n✅ COMPLETE! Mappers ready for next step.")
    else:
        print("\n❌ Failed to download HGNC data.")
        print("   Please check your internet connection or download manually from:")
        print("   https://www.genenames.org/download/")

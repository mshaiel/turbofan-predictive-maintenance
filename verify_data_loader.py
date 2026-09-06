"""
Verification script for src/data_loader.py
Loads all 8 train/test splits and all 4 RUL files, displaying shapes and dtypes.
"""

from src.data_loader import load_dataset, load_rul, COLUMNS

def main():
    print("=" * 60)
    print("VERIFYING NASA C-MAPSS DATA INGESTION PIPELINE")
    print("=" * 60)

    subsets = ["FD001", "FD002", "FD003", "FD004"]
    for subset in subsets:
        print(f"\n--- Subset: {subset} ---")
        df_train = load_dataset(subset=subset, split="train")
        df_test = load_dataset(subset=subset, split="test")
        rul_series = load_rul(subset=subset)

        print(f"Train shape: {df_train.shape} | Engines: {df_train['engine_id'].nunique()}")
        print(f"Test shape : {df_test.shape}  | Engines: {df_test['engine_id'].nunique()}")
        print(f"RUL shape  : {rul_series.shape}     | Engines: {len(rul_series)}")

        assert df_train.shape[1] == len(COLUMNS), f"Mismatch in train columns: {df_train.shape[1]}"
        assert df_test.shape[1] == len(COLUMNS), f"Mismatch in test columns: {df_test.shape[1]}"
        assert list(df_train.columns) == COLUMNS, "Train column list mismatch"
        assert list(df_test.columns) == COLUMNS, "Test column list mismatch"
        assert not df_train.isnull().any().any(), "Null values detected in train"
        assert not df_test.isnull().any().any(), "Null values detected in test"
        assert not rul_series.isnull().any(), "Null values detected in RUL"
        assert (df_train["engine_id"].dtype == int or df_train["engine_id"].dtype == "int64"), "engine_id not int"
        assert (df_train["cycle"].dtype == int or df_train["cycle"].dtype == "int64"), "cycle not int"

    print("\n" + "=" * 60)
    print("DATA TYPES FOR FD001 TRAIN:")
    print("=" * 60)
    df_fd001 = load_dataset("FD001", "train")
    for col, dt in df_fd001.dtypes.items():
        print(f"  {col:<15}: {dt}")

    print("\n[SUCCESS] All 8 datasets and 4 RUL ground-truth files verified perfectly!")

if __name__ == "__main__":
    main()

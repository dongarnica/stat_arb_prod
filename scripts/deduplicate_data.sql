
-- Deduplication script for historical_bars_baseline
-- BACKUP YOUR DATA BEFORE RUNNING THIS!

-- Step 1: Create a temporary table with deduplicated data
CREATE TABLE historical_bars_baseline_dedup AS
SELECT DISTINCT ON (symbol, bar_date, bar_size)
    symbol, bar_date, bar_size, 
    open_price, high_price, low_price, close_price, volume
FROM historical_bars_baseline
ORDER BY symbol, bar_date, bar_size, ctid;

-- Step 2: Check the results
SELECT 
    'Original' as table_name,
    COUNT(*) as record_count
FROM historical_bars_baseline
UNION ALL
SELECT 
    'Deduplicated' as table_name,
    COUNT(*) as record_count  
FROM historical_bars_baseline_dedup;

-- Step 3: If results look good, replace the original table
-- BEGIN;
-- DROP TABLE historical_bars_baseline;
-- ALTER TABLE historical_bars_baseline_dedup RENAME TO historical_bars_baseline;
-- COMMIT;

-- Step 4: Add constraints to prevent future duplicates
-- ALTER TABLE historical_bars_baseline 
-- ADD CONSTRAINT unique_bar_key UNIQUE (symbol, bar_date, bar_size);

-- Step 5: Recreate indices
-- (Run the index creation script after deduplication)

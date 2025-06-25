#!/usr/bin/env python3
"""
Database optimization script for historical data loader.

This script provides tools to:
1. Create optimized database indices
2. Analyze query performance
3. Profile different query patterns
4. Generate optimization recommendations
"""

import sys
from pathlib import Path
import time
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statistics_service.data_ingestion.postgres_loader import (
    create_database_indices,
    get_engine,
    profile_query_performance,
    load_historical_data
)
from statistics_service.config.config_manager import config
from sqlalchemy import text


def analyze_table_stats():
    """Analyze the historical_bars_baseline table for optimization insights."""
    print("=== Table Statistics Analysis ===")
    
    engine = get_engine()
    
    queries = {
        "Total rows": "SELECT COUNT(*) FROM historical_bars_baseline",
        "Unique symbols": "SELECT COUNT(DISTINCT symbol) FROM historical_bars_baseline",
        "Unique timeframes": "SELECT COUNT(DISTINCT bar_size) FROM historical_bars_baseline",
        "Date range": """
            SELECT 
                MIN(bar_date) as earliest_date,
                MAX(bar_date) as latest_date,
                MAX(bar_date) - MIN(bar_date) as date_span
            FROM historical_bars_baseline
        """,
        "Table size": """
            SELECT 
                pg_size_pretty(pg_total_relation_size('historical_bars_baseline')) as table_size,
                pg_size_pretty(pg_relation_size('historical_bars_baseline')) as data_size,
                pg_size_pretty(pg_indexes_size('historical_bars_baseline')) as index_size
        """,
    }
    
    try:
        with engine.connect() as conn:
            for name, query in queries.items():
                start_time = time.time()
                result = conn.execute(text(query))
                query_time = time.time() - start_time
                
                rows = result.fetchall()
                print(f"\n{name}:")
                print(f"  Query time: {query_time:.3f}s")
                for row in rows:
                    print(f"  Result: {dict(row._mapping)}")
                    
    except Exception as e:
        print(f"Error analyzing table stats: {e}")


def check_existing_indices():
    """Check what indices currently exist on the table."""
    print("\n=== Existing Indices Analysis ===")
    
    engine = get_engine()
    
    query = """
        SELECT 
            indexname,
            indexdef,
            pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
        FROM pg_indexes 
        WHERE tablename = 'historical_bars_baseline'
        ORDER BY indexname
    """
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            indices = result.fetchall()
            
            if indices:
                print(f"Found {len(indices)} existing indices:")
                for idx in indices:
                    print(f"  - {idx.indexname} ({idx.index_size})")
                    print(f"    {idx.indexdef}")
            else:
                print("No custom indices found (only primary key/default)")
                
    except Exception as e:
        print(f"Error checking indices: {e}")


def benchmark_query_patterns():
    """Benchmark common query patterns to identify bottlenecks."""
    print("\n=== Query Performance Benchmarks ===")
    
    # Test different query patterns
    test_cases = [
        {
            "name": "Single symbol, no timeframe filter",
            "params": {"symbol": "SPY", "limit": 1000}
        },
        {
            "name": "Single symbol, with timeframe",
            "params": {"symbol": "SPY", "timeframe": "5min", "limit": 1000}
        },
        {
            "name": "Multiple symbols, no timeframe",
            "params": {"symbol": ["SPY", "QQQ", "IWM"], "limit": 1000}
        },
        {
            "name": "Date range query (last 30 days)",
            "params": {
                "symbol": "SPY",
                "start_date": datetime.now() - timedelta(days=30),
                "end_date": datetime.now()
            }
        },
        {
            "name": "Large result set (10k rows)",
            "params": {"symbol": "SPY", "limit": 10000}
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        try:
            start_time = time.time()
            df = load_historical_data(**test['params'], profile_query=True)
            total_time = time.time() - start_time
            
            result = {
                "test_name": test['name'],
                "total_time": total_time,
                "rows_returned": len(df),
                "rows_per_second": len(df) / total_time if total_time > 0 else 0,
                "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            results.append(result)
            
            print(f"  Time: {total_time:.3f}s")
            print(f"  Rows: {len(df)}")
            print(f"  Speed: {result['rows_per_second']:.0f} rows/sec")
            print(f"  Memory: {result['memory_mb']:.1f} MB")
            
        except Exception as e:
            print(f"  Error: {e}")
            
    return results


def explain_query_plans():
    """Analyze query execution plans for optimization opportunities."""
    print("\n=== Query Execution Plan Analysis ===")
    
    engine = get_engine()
    
    # Common query patterns to analyze
    queries = [
        {
            "name": "Symbol filter only",
            "sql": """
                EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                SELECT symbol, bar_date, close_price, bar_size 
                FROM historical_bars_baseline 
                WHERE symbol = 'SPY' 
                LIMIT 1000
            """
        },
        {
            "name": "Symbol + timeframe filter",
            "sql": """
                EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                SELECT symbol, bar_date, close_price, bar_size 
                FROM historical_bars_baseline 
                WHERE symbol = 'SPY' AND bar_size = '5min'
                LIMIT 1000
            """
        },
        {
            "name": "Date range filter",
            "sql": """
                EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                SELECT symbol, bar_date, close_price, bar_size 
                FROM historical_bars_baseline 
                WHERE bar_date >= CURRENT_DATE - INTERVAL '30 days'
                LIMIT 1000
            """
        }
    ]
    
    try:
        with engine.connect() as conn:
            for query in queries:
                print(f"\n{query['name']}:")
                result = conn.execute(text(query['sql']))
                plan = result.fetchone()[0]
                
                # Extract key metrics from execution plan
                if plan and len(plan) > 0:
                    exec_plan = plan[0]['Plan']
                    print(f"  Execution time: {plan[0].get('Execution Time', 'N/A')} ms")
                    print(f"  Planning time: {plan[0].get('Planning Time', 'N/A')} ms")
                    print(f"  Node type: {exec_plan.get('Node Type', 'N/A')}")
                    print(f"  Total cost: {exec_plan.get('Total Cost', 'N/A')}")
                    
                    if 'Index Name' in exec_plan:
                        print(f"  Using index: {exec_plan['Index Name']}")
                    else:
                        print("  No index used (sequential scan)")
                        
    except Exception as e:
        print(f"Error analyzing query plans: {e}")


def generate_optimization_recommendations():
    """Generate recommendations based on analysis results."""
    print("\n=== Optimization Recommendations ===")
    
    recommendations = [
        {
            "priority": "HIGH",
            "action": "Create composite index on (symbol, bar_date, bar_size)",
            "reason": "Most queries filter by symbol and often include date ranges",
            "sql": "CREATE INDEX CONCURRENTLY idx_symbol_date_barsize ON historical_bars_baseline (symbol, bar_date, bar_size);"
        },
        {
            "priority": "MEDIUM",
            "action": "Create separate index on bar_date for date-only queries",
            "reason": "Some queries filter only by date range",
            "sql": "CREATE INDEX CONCURRENTLY idx_bar_date ON historical_bars_baseline (bar_date);"
        },
        {
            "priority": "LOW",
            "action": "Consider partitioning by date",
            "reason": "For very large datasets, monthly partitions can improve query performance",
            "sql": "-- Requires table restructuring, consult DBA"
        },
        {
            "priority": "LOW",
            "action": "Increase shared_buffers in PostgreSQL config",
            "reason": "More buffer cache can reduce disk I/O for frequently accessed data",
            "sql": "-- Requires PostgreSQL configuration change"
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['priority']} PRIORITY:")
        print(f"  Action: {rec['action']}")
        print(f"  Reason: {rec['reason']}")
        print(f"  SQL: {rec['sql']}")


def main():
    """Run complete database optimization analysis."""
    print("Historical Data Loader - Database Optimization Analysis")
    print("=" * 60)
    
    try:
        # Step 1: Analyze current table statistics
        analyze_table_stats()
        
        # Step 2: Check existing indices
        check_existing_indices()
        
        # Step 3: Benchmark query performance
        benchmark_results = benchmark_query_patterns()
        
        # Step 4: Analyze query execution plans
        explain_query_plans()
        
        # Step 5: Generate recommendations
        generate_optimization_recommendations()
        
        # Step 6: Offer to create indices
        print("\n=== Index Creation ===")
        response = input("Create recommended database indices? (y/N): ")
        if response.lower() == 'y':
            print("Creating database indices...")
            create_database_indices()
            print("Indices created successfully!")
            
            # Re-run benchmarks to show improvement
            print("\nRe-running benchmarks to measure improvement...")
            new_results = benchmark_query_patterns()
            
            print("\n=== Performance Comparison ===")
            for old, new in zip(benchmark_results, new_results):
                if old and new:
                    improvement = (old['total_time'] - new['total_time']) / old['total_time'] * 100
                    print(f"{old['test_name']}: {improvement:+.1f}% time change")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

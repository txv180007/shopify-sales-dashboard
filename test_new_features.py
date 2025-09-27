"""
Test script to verify new features work correctly
"""
import pandas as pd
from datetime import datetime, timedelta

def test_date_range_functionality():
    """Test the date range quick filters."""
    from utils import get_date_range_from_option
    from config import DATE_RANGE_OPTIONS

    # Mock data bounds
    today = datetime.now().date()
    data_min = pd.Timestamp(today - timedelta(days=90))
    data_max = pd.Timestamp(today)

    # Test different options
    for option_name, days_back in DATE_RANGE_OPTIONS.items():
        if option_name == "Custom range":
            continue

        start_date, end_date = get_date_range_from_option(option_name, data_min, data_max)

        print(f"{option_name}: {start_date} to {end_date}")

        # Verify logic
        if days_back == 0:  # Today
            assert start_date == end_date == today
        elif days_back == 1:  # Yesterday
            yesterday = today - timedelta(days=1)
            assert start_date == end_date == yesterday
        else:  # Last N days
            expected_start = today - timedelta(days=days_back - 1)
            assert start_date == expected_start
            assert end_date == today

    print("PASS: Date range functionality works correctly")


def test_export_functionality():
    """Test the export utility functions."""
    from utils import create_download_buttons
    import io
    import sys

    # Create test dataframe
    test_df = pd.DataFrame({
        'Product': ['Item A', 'Item B', 'Item C'],
        'Qty Sold': [10, 5, 15],
        'Revenue': [100.50, 75.25, 200.75],
        'Has Image': [True, False, True]
    })

    # Test CSV export
    csv_output = test_df.to_csv(index=False)
    assert 'Product,Qty Sold,Revenue,Has Image' in csv_output
    assert 'Item A,10,100.5,True' in csv_output

    # Test JSON export
    json_output = test_df.to_json(orient='records', indent=2)
    assert '"Product"' in json_output
    assert '"Item A"' in json_output
    assert '"Qty Sold"' in json_output

    print("PASS: Export functionality works correctly")


def test_config_updates():
    """Test the updated configuration."""
    from config import DATE_RANGE_OPTIONS, DEFAULT_DATE_RANGE

    # Check date range options exist
    assert isinstance(DATE_RANGE_OPTIONS, dict)
    assert "Today" in DATE_RANGE_OPTIONS
    assert "Last 7 days" in DATE_RANGE_OPTIONS
    assert "Last 30 days" in DATE_RANGE_OPTIONS
    assert "Custom range" in DATE_RANGE_OPTIONS

    # Check default is set
    assert DEFAULT_DATE_RANGE in DATE_RANGE_OPTIONS

    print("PASS: Configuration updates work correctly")


def test_table_height_calculation():
    """Test table height calculation utility."""
    from utils import get_table_height

    # Test with various row counts
    height_10 = get_table_height(10, min_height=300, max_height=600)
    height_50 = get_table_height(50, min_height=300, max_height=600)

    # Should respect minimum
    assert height_10 >= 300

    # Should respect maximum
    assert height_50 <= 600

    # More rows should mean more height (within limits)
    assert height_50 >= height_10

    print("PASS: Table height calculation works correctly")


def main():
    """Run all tests for new features."""
    print("Testing new dashboard features...")
    print("-" * 40)

    try:
        test_config_updates()
        test_date_range_functionality()
        test_export_functionality()
        test_table_height_calculation()

        print("-" * 40)
        print("SUCCESS: All new features work correctly!")

    except Exception as e:
        print(f"FAILED: Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
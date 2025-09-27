"""
Test script to verify refactored modules work correctly
"""
import pandas as pd
import numpy as np

def test_data_processor():
    """Test data processing functions."""
    from data_processor import coerce_num, parse_search_terms, contains_any

    # Test coerce_num
    assert coerce_num("$123.45") == 123.45
    assert coerce_num("12.5%", is_percentage=True) == 0.125
    assert coerce_num("(100)") == -100.0
    assert pd.isna(coerce_num(""))

    # Test parse_search_terms
    assert parse_search_terms("apple, banana cherry") == ["apple", "banana", "cherry"]
    assert parse_search_terms("") == []

    # Test contains_any
    series = pd.Series(["Apple Pie", "Banana Bread", "Cherry Cake"])
    result = contains_any(series, ["apple", "bread"])
    assert result.iloc[0] == True  # Apple Pie matches "apple"
    assert result.iloc[1] == True  # Banana Bread matches "bread"
    assert result.iloc[2] == False # Cherry Cake matches neither

    print("PASS: Data processor tests passed")


def test_analytics():
    """Test analytics modules."""
    from analytics import StockAnalyzer

    # Create sample data
    sample_data = pd.DataFrame({
        'title': ['Product A', 'Product B', 'Product A'],
        'qtySold': [10, 5, 15],
        'netSales': [100.0, 50.0, 150.0],
        'inventoryNow': [20, 0, 25],
        'avgUnitNet': [10.0, 10.0, 10.0],
        'barcode': ['123', '456', '123'],
        'sku': ['SKU1', 'SKU2', 'SKU1'],
        'dateYMD': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02'])
    })

    analyzer = StockAnalyzer(sample_data)
    shopping_list = analyzer.get_shopping_list()

    assert not shopping_list.empty
    assert 'Product' in shopping_list.columns
    assert 'Qty Sold' in shopping_list.columns

    out_of_stock = analyzer.get_out_of_stock_items()
    # Product B should be out of stock (inventory = 0)
    assert len(out_of_stock[out_of_stock['Product'] == 'Product B']) > 0

    print("PASS: Analytics tests passed")


def test_config():
    """Test configuration module."""
    from config import NUMERIC_COLUMNS, STRING_COLUMNS, APP_TITLE

    assert isinstance(NUMERIC_COLUMNS, dict)
    assert isinstance(STRING_COLUMNS, list)
    assert APP_TITLE == "Daily Products Dashboard"

    print("PASS: Config tests passed")


def main():
    """Run all tests."""
    print("Running refactored code tests...")

    try:
        test_config()
        test_data_processor()
        test_analytics()

        print("\nSUCCESS: All tests passed! Refactored code is working correctly.")

    except Exception as e:
        print(f"\nFAILED: Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
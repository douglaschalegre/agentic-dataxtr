# Excel Extraction Example

This example demonstrates how to extract structured data from Excel spreadsheets.

## Quick Start

```bash
# Using default file (Malha Caso 15.xlsx)
uv run python examples/excel_extraction.py

# Using custom file
uv run python examples/excel_extraction.py path/to/your/file.xlsx
```

## Customizing the Schema

The example includes a basic schema. To customize it for your Excel file:

1. **Inspect your Excel file** first to understand the structure
2. **Edit `excel_extraction.py`** and modify the `schema` list
3. **Define fields** that match your spreadsheet data

### Example Schema Patterns

#### Financial Report
```python
schema = [
    FieldDefinition(
        name="period",
        description="Reporting period (month/year)",
        field_type=FieldType.DATE,
        required=True,
    ),
    FieldDefinition(
        name="revenue",
        description="Total revenue amount",
        field_type=FieldType.CURRENCY,
        required=True,
    ),
    FieldDefinition(
        name="expenses_breakdown",
        description="Table of expense categories with amounts",
        field_type=FieldType.TABLE,
        required=True,
    ),
]
```

#### Inventory List
```python
schema = [
    FieldDefinition(
        name="items",
        description="Table of items with SKU, description, quantity, and price",
        field_type=FieldType.TABLE,
        required=True,
    ),
    FieldDefinition(
        name="total_value",
        description="Total inventory value",
        field_type=FieldType.CURRENCY,
        required=True,
    ),
]
```

#### Contact List
```python
schema = [
    FieldDefinition(
        name="contacts",
        description="Table with names, emails, phones, and companies",
        field_type=FieldType.TABLE,
        required=True,
    ),
    FieldDefinition(
        name="total_contacts",
        description="Total number of contacts",
        field_type=FieldType.NUMBER,
        required=False,
    ),
]
```

## Field Types

| Type | Description | Example |
|------|-------------|---------|
| `TEXT` | Plain text | Company name, address |
| `NUMBER` | Numeric value | Count, quantity |
| `DATE` | Date value | 2024-01-15 |
| `CURRENCY` | Money amount | $1,234.56 |
| `BOOLEAN` | Yes/No | Active status |
| `TABLE` | Structured data | Multi-column table |
| `LIST` | Array of values | Tags, categories |

## Tips for Excel Extraction

1. **Use TABLE type for multi-row data** - Best for extracting entire ranges
2. **Be specific in descriptions** - Help the LLM understand what to look for
3. **Add examples** - Use the `examples` parameter for ambiguous fields
4. **Sheet names matter** - LLM will see sheet names in the context

## Troubleshooting

### No data extracted
- Check if Excel file has multiple sheets
- Verify field descriptions match actual data
- Try simpler field names first

### Table extraction returns None
- Excel file may have complex formatting
- Try extracting specific cells as TEXT first
- Use `search_document` tool to find data location

### Context length errors
- Break large schemas into smaller groups
- Extract one sheet at a time if multiple sheets
- Use targeted field descriptions

## Example Output

```json
{
  "empresa": {
    "value": "ACME Corp",
    "confidence": 0.95,
    "source": "Sheet1, Cell A1",
    "method": "text"
  },
  "valores_totais": {
    "value": {
      "headers": ["Category", "Amount"],
      "rows": [
        ["Revenue", "1000000"],
        ["Expenses", "750000"]
      ]
    },
    "confidence": 0.85,
    "method": "table"
  }
}
```

# Data Format Reference

Input: JSON (`.json`) or CSV (`.csv`/`.tsv`). All values are floats unless noted.

## Bar Chart (`--type bar`)

```json
{
  "labels": ["Group A", "Group B", "Group C"],
  "series": {
    "Series 1": [10.5, 20.3, 15.2],
    "Series 2": [12.0, 18.7, 16.5]
  },
  "errors": {
    "Series 1": [1.2, 0.8, 1.5],
    "Series 2": [0.9, 1.1, 0.7]
  },
  "significance": {
    "Series 1:0": "***",
    "Series 1:1": "*",
    "Series 1:2": "NS"
  }
}
```

- `labels` (array): X-axis group labels
- `series` (object): Each key = series name, value = array of bar heights
- `errors` (object, optional): Per-bar error values. Key must match series name
- `significance` (object, optional): `"series_prefix:group_index"` → label (`*`, `**`, `***`, `NS`)

### CSV equivalent
```
Group,Series1,Series2
A,10.5,12.0
B,20.3,18.7
C,15.2,16.5
```

Note: CSV mode does not support errors or significance — use JSON for those.

---

## Heatmap (`--type heatmap`)

```json
{
  "matrix": [[10, -5, 3], [8, 12, -2], [-1, 7, 15]],
  "row_labels": ["Row A", "Row B", "Row C"],
  "col_labels": ["Col 1", "Col 2", "Col 3"],
  "cbar_label": "Value (units)"
}
```

- `matrix` (2D array): Values for heatmap cells
- `row_labels` / `col_labels`: Axis labels
- `cbar_label` (optional): Colorbar label
- Override range: `--vmin -20 --vmax 45 --cmap RdBu_r`

---

## Scatter Plot (`--type scatter`)

```json
{
  "x": [1.2, 2.3, 3.1, 4.5],
  "y": [5.1, 6.2, 4.8, 7.3],
  "groups": ["A", "A", "B", "B"]
}
```

- `x`, `y`: Data arrays (equal length)
- `groups` (optional): Color grouping. Values become legend entries

---

## Line Chart (`--type line`)

```json
{
  "labels": ["Jan", "Feb", "Mar"],
  "series": {
    "Revenue": [42, 48, 55],
    "Profit": [12, 15, 18]
  },
  "errors": {
    "Revenue": [3, 2, 4],
    "Profit": [1, 1.5, 2]
  }
}
```

- `labels`: X-axis labels
- `series`: Named line series
- `errors` (optional): Error band width per series

### CSV equivalent
```
Month,Revenue,Profit
Jan,42,12
Feb,48,15
Mar,55,18
```

---

## Box Plot (`--type box`)

```json
{
  "labels": ["Treatment A", "Treatment B", "Control"],
  "series": {
    "Treatment A": [10, 12, 11, 13, 10, 14],
    "Treatment B": [8, 9, 10, 7, 11, 9],
    "Control": [5, 6, 4, 7, 5, 6]
  }
}
```

Each series value is an array of raw data points.

---

## Forest Plot (`--type forest`)

```json
{
  "labels": ["Study A", "Study B", "Study C"],
  "estimates": [1.2, 0.8, 1.5],
  "ci_low": [0.9, 0.5, 1.1],
  "ci_high": [1.6, 1.3, 2.0],
  "overall": {
    "estimate": 1.1,
    "ci_low": 0.85,
    "ci_high": 1.35
  },
  "ref_line": 1.0
}
```

- `labels`: Study names (Y-axis)
- `estimates`: Point estimates
- `ci_low` / `ci_high`: Confidence interval bounds
- `overall` (optional): Diamond summary estimate
- `ref_line` (optional): Reference/null line (default: 0)

---

## Violin Plot (`--type violin`)

Same data format as Box Plot — each series is an array of raw data points.

```json
{
  "labels": ["Treatment", "Control", "Placebo"],
  "series": {
    "Treatment": [8, 9, 10, 7, 11, 9, 8, 12],
    "Control": [5, 6, 4, 7, 5, 6, 4, 5],
    "Placebo": [6, 5, 7, 5, 6, 4, 6, 5]
  }
}
```

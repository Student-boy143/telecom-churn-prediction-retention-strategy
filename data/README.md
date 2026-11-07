# Data

The raw dataset is **not included** in this repository. It describes customers of an anonymised telecom operator, was provided under data-protection restrictions, and is not licensed for public redistribution.

## Expected files

Place the two files directly in this folder:

```
data/
├── Client.csv    # one row per customer: account, handset, household and credit information
└── Record.csv    # one row per customer: monthly usage, billing and call-quality metrics + churn label
```

| File | Rows | Columns | Key |
|---|---|---|---|
| `Client.csv` | 100,000 | 50 | `Customer_ID` |
| `Record.csv` | 100,000 | 51 (incl. `churn`) | `Customer_ID` |

The tables join one-to-one on `Customer_ID` (≈100 columns after the merge). The target `churn` marks customers who left **31–60 days after the observation date**. The sample is **balanced by design (49.6% churners)** and does not reflect real-world churn prevalence.

## How to obtain the data

Use only a copy you are authorised to access. The notebook will stop with a clear message if either file is missing. Nothing in `data/` (other than this README) is tracked by Git — see `.gitignore`.

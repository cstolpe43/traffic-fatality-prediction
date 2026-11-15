# Data

The raw data is not stored in this repository because of its size.

## Source

NHTSA Fatality Analysis Reporting System (FARS), 2015 and 2016 national CSV files:
https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/

For each year, download the national CSV zip (for example `FARS2015NationalCSV.zip`) and extract it.

## Expected layout

The notebooks read four files from `data/raw/`. Copy and rename them from the extracted folders:

| From the FARS download | Save as |
|------------------------|---------|
| 2015 `accident.csv` | `data/raw/accident_2015.csv` |
| 2015 `person.csv`   | `data/raw/person_2015.csv` |
| 2016 `accident.csv` | `data/raw/accident_2016.csv` |
| 2016 `person.csv`   | `data/raw/person_2016.csv` |

File name capitalization varies between FARS releases (`Person.CSV`, `person.csv`); either works on Windows.

## Generated files

`notebooks/02_data_preparation.ipynb` writes `data/fars_2015_prepared.csv` and `data/fars_2016_prepared.csv`, which notebooks 03–05 read.

## Coding notes

Codes follow the FARS Analytical User's Manual for 2015–2016. In particular:

- `ALC_RES` is in thousandths of g/dL (`80` = 0.080). Values 995–999 are status codes (not reported, test not given, result unknown, positive without value, unknown).
- Rural/urban comes from the accident file's `RUR_URB` (1 = rural, 2 = urban). `FUNC_SYS` is the road's functional class (1 = Interstate … 7 = Local).
- Codes such as 8/9, 98/99 and 96–99 mean "not reported", "unknown" or "not applicable" in many fields and are treated as missing.

Check these against the manual for the exact release you download, since FARS codes change between years.

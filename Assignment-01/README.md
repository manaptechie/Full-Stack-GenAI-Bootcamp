# Assignment-01 Run Guide

This assignment uses the existing Python 3.11 virtual environment at:

`env\Scripts\python.exe`

## 1. Open Terminal At Repo Root

Run all commands from:

`D:\GitHub\Full-Stack-GenAI-Bootcamp`

## 2. Install Dependencies (using uv)

```powershell
uv pip install --python .\env\Scripts\python.exe -r .\requirements.txt
```

## 3. Generate Real-world Review Dataset (CSV)

```powershell
uv run --python .\env\Scripts\python.exe .\Assignment-01\fetch_flipkart_reviews.py --output .\Assignment-01\flipkart_reviews.csv --target-count 110 --max-pages-per-sort 8
```

Expected output includes:

- `Saved 110 reviews to ...\Assignment-01\flipkart_reviews.csv`
- balanced sentiment counts (`positive` and `negative`)

## 4. Run The Notebook

Open:

`Assignment-01\EncodingImplementation.ipynb`

Then select kernel:

- `env (Python 3.11.x)`

Run all cells from top to bottom.

## 5. Output Files For Submission

- Notebook: `Assignment-01\EncodingImplementation.ipynb`
- Dataset: `Assignment-01\flipkart_reviews.csv`
- Report: `Assignment-01\AssignmentReport.md`

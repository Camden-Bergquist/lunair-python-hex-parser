# Implantable-Device Hex Log Parser

Python tool independently developed during my 2024 software engineering internship at Lunair Medical, a medical-device startup focused on sleep apnea. It converts hexadecimal logs from prototype implantable pulse generators (IPGs) into structured CSV files for inspection and analysis.

## Project Highlights

- Reverse-engineered the undocumented device-log format and implemented decoding logic in Python.
- Converted encoded packet and sample data into tabular output suitable for downstream analysis.
- Built an interactive file-selection workflow with progress indicators and repeat processing.
- Packaged the tool as a standalone Windows executable for users who did not need to work with the Python source.

## How It Works

1. Place a supported `.hex` or `.txt` log in `raw_data/`.
2. Launch the parser from the repository directory.
3. Select a file from the numbered list.
4. Collect the resulting CSV from `processed_data/`.

The parser creates the output directory if needed and prompts the user to process another file after each run. Processing time depends on log size.

## Repository Structure

| Location | Purpose |
|---|---|
| `main.py` | Log decoding, data transformation, and interactive processing |
| `raw_data/` | Example logs and input location |
| `processed_data/` | Generated CSV output location |

The `.hex` and `.txt` examples with matching names contain the same sample data in alternative file extensions.

## Running from Source

The source imports pandas, NumPy, and the `progressbar` module. With those dependencies installed, run from the repository root:

```bash
python main.py
```

The working directory matters because the script uses relative input and output paths. The internship tool was distributed as a Windows executable; the supplied source archive does not include that executable.

## Scope and Limitations

This is a historical tool for a specific prototype-device log format, rather than a general hexadecimal parser. Other devices or changed packet layouts require corresponding changes to the decoding logic. The CSV output supports engineering analysis; compatibility with unrelated device formats is not implied.

## Tools

Python, pandas, NumPy, and progress indicators through the `progressbar` module.

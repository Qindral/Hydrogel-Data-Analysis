from pathlib import Path
import re
import pandas as pd
from core.io import extract_particle_size_from_path
import matplotlib.pyplot as plt
root = Path(r"E:\PhD Data Analysis\SPT 2025 II")

results = []
print(len(list(root.rglob("*.rec"))))
for rec_file in root.rglob("*.rec"):
    # Extract particle size from filename using existing pattern
    match = re.search(r'(\d+)nm', rec_file.stem, re.IGNORECASE)
    particle_size_nm = extract_particle_size_from_path(rec_file) 
    
    # Read file and find power line
    power_value = None
    power_unit = None
    power_match = False
    try:
        with open(rec_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Search for lines containing power units
                if 'µW' in line or 'mW' in line or 'MW' in line:
                    # Extract unit
                    unit_match = re.search(r'([µu]W|mW|MW)', line, re.IGNORECASE)
                    if unit_match:
                        power_unit = unit_match.group(1)
                        
                        # Remove all letters and whitespace to extract number
                        cleaned = re.sub(r'[a-zA-Zµ\s]', '', line)
                        print(cleaned)
                        # Find first number (with optional decimal point)
                        number_match = float(cleaned)
                        if number_match:
                            power_value = float(number_match.group(0))
                            power_match = True
                            break
                if power_match:
                    power_value = power_value
                    power_unit = power_unit
                    break
    except Exception as e:
        print(f"Error reading {rec_file.name}: {e}")
        continue
    
    results.append({
        'file': rec_file.name,
        'particle_size_nm': particle_size_nm,
        'power_value': power_value,
        'power_unit': power_unit,
        'path': str(rec_file),
        "row": line.strip() if power_value is not None else None
    })

df = pd.DataFrame(results)
plt.figure(figsize=(10,6))
for particle_size in [20,50,100,200,500,1000]:
    subset = df[df['particle_size_nm'] == particle_size]
    for row in subset.itertuples():
        if row.power_value is not None:
            if row.power_unit == 'µW':
                power_mw = row.power_value * 1e-3
            elif row.power_unit == 'mW' or row.power_unit == 'MW'   :
                power_mw = row.power_value
            if row.file == "100 nm TriSpuffer_3.tif.rec":
                print(row.file, row.power_value, row.power_unit, row.row)
            plt.scatter(particle_size, power_mw)
plt.show()           
# print(df)
#df.to_csv(root / 'rec_file_analysis.csv', index=False)
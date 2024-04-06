# json converter. converts json to excel file.
# delimiter : |  

import json
import csv
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: json2csv.py <input json file> <output csv file>")
        sys.exit(1)

    json_file = sys.argv[1]
    csv_file = sys.argv[2]

    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(csv_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter='|')

        # write header
        csv_writer.writerow(data[0].keys())

        # write data
        for row in data:
            csv_writer.writerow(row.values())

if __name__ == "__main__":
    main()

    
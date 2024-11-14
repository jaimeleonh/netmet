import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

### give distinct names to json files 



def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(f"Loaded {len(data)} entries from {file_path}")
    return data


def compare_models(file1, file2):
    """Compare two JSON files and make histograms of the corresponding values."""
    # Load the JSON files
    data1 = load_json(file1)
    data2 = load_json(file2)

    # Get the names of the JSON files
    title1 = Path(file1).stem
    title2 = Path(file2).stem
    os.makedirs(f"plots/{title1}_vs_{title2}", exist_ok=True)
    path = f"plots/{title1}_vs_{title2}/"

    ##################################
    # L1 MET rate vs L1 NET MET rate #
    ##################################

    xrange = [0, 200]
    bins = xrange[1]

    #l1MET_bkg = data1['l1MET_bkg']  # same for both files
    l1NetMET_bkg_1 = data1['rate']['l1NetMET_bkg']
    l1NetMET_bkg_2 = data2['rate']['l1NetMET_bkg']

    #plt.hist(l1MET_bkg, bins=bins, range=xrange, histtype='step', label='L1 MET Rate', cumulative=-1, log=True)
    plt.hist(l1NetMET_bkg_1, bins=bins, range=xrange, histtype='step', label=f'L1 NetMET Rate {title1}', cumulative=-1, log=True)
    plt.hist(l1NetMET_bkg_2, bins=bins, range=xrange, histtype='step', label=f'L1 NetMET Rate {title2}', cumulative=-1, log=True)
    plt.legend()
    plt.xlabel('MET Rate')
    plt.ylabel('Events')
    plt.savefig(path + 'MET_rate.png')
    plt.close()

    ##################
    # MET Resolution #
    ##################
    l1MET_diff = data1['resolution']['l1MET_diff']  # L1MET difference is the same for both files
    l1NetMET_diff_1 = data1['resolution']['l1NetMET_diff']
    l1NetMET_diff_2 = data2['resolution']['l1NetMET_diff']

    bins = 80
    xrange = [-100, 100]
    plt.hist(l1MET_diff, bins=bins, range=xrange, histtype='step', label='L1 MET Diff')
    plt.hist(l1NetMET_diff_1, bins=bins, range=xrange, histtype='step', label=f'L1 NetMET Diff {title1}')
    plt.hist(l1NetMET_diff_2, bins=bins, range=xrange, histtype='step', label=f'L1 NetMET Diff {title2}')
    plt.legend()
    plt.xlabel('MET Difference')
    plt.ylabel('Events')
    plt.title('MET Difference')
    plt.savefig(path + 'MET_difference.png')
    plt.close()

    print(f"Plots saved in {path}")



if __name__ == "__main__":
    # Ensure exactly two JSON files are provided
    if len(sys.argv) != 3:
        print("Usage: python compare_models.py <json_file1> <json_file2>")
        sys.exit(1)

    # Get the JSON file paths from arguments
    file1, file2 = sys.argv[1], sys.argv[2]

    # Call the function to plot
    compare_models(file1, file2)


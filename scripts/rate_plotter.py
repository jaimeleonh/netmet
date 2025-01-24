from matplotlib import pyplot as plt
import numpy as np

folder = "/vols/cms/jleonhol/cmt/BasePlotting/base/test_250122{}/"

inputs = {
    "PUM": "",
    "No PUM": "_emu_121",
}

ax = plt.subplot()
for key, tag in inputs.items():
    with open(folder.format(tag) + "x_efficiency.npy" , 'rb') as f:
        x = np.load(f)
    with open(folder.format(tag) + "y_efficiency.npy" , 'rb') as f:
        y = np.load(f)
    with open(folder.format(tag) + "y_error_efficiency.npy" , 'rb') as f:
        y_errors = np.load(f)
    plt.errorbar(x, y, y_errors, label=key, marker='o', capsize=7, linestyle='none')
    print(key)
    print(y)
ax.set_xlabel('PUPPI MET No Mu [GeV]')
ax.set_ylabel('Efficiency')
plt.legend()
plt.savefig("eff_comparison.pdf")
plt.close('all')


ax = plt.subplot()
for key, tag in inputs.items():
    with open(folder.format(tag) + "x_rate.npy" , 'rb') as f:
        x = np.load(f)
    with open(folder.format(tag) + "y_rate.npy" , 'rb') as f:
        y = np.load(f)
    plt.stairs(y, x, label=key)
    
    if key == "PUM":
        target_rate = y[80]
    else:
        for i in range(len(y)):
            if y[i] < target_rate:
                print(i)
                break

    print(key)
    print(y)
ax.set_xlabel('PUPPI MET No Mu [GeV]')
ax.set_ylabel('Rate')
ax.set_yscale('log')
plt.legend()
plt.savefig("rate_comparison.pdf")
plt.close('all')

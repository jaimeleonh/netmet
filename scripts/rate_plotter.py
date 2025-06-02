from matplotlib import pyplot as plt
import numpy as np

folder = "/vols/cms/jleonhol/cmt/BasePlotting/base/test_2301_{}/"
netmet_folder = "/vols/cms/jleonhol/cmt/BDTValidation/base/FT_{}__N_10__MD_6__OBJ_reg_linear__RS_1__SIG_{}__RATIO_0/test_2301_90/"

inputs = {
    "L1 NETMET (current) > 116": ("hf", "signal_nopum_new"),
    "L1 NETMET (con) > 118": ("hf_emu", "signal_con"),
}

inputs_met = {
    "L1 MET (current) > 90": "old_90",
    # "L1 MET (pumOff) > 90": "pumoff_90",
    #"L1 MET (pumOff) > 127": "pumoff_127",
    "L1 MET (con) > 90": "con_90",
}

inputs_met_rate = {
    "L1 MET (current)": "old_90",
    "L1 MET (pumOff)": "pumoff_127",
    "L1 MET (con)": "con_90",
}

inputs_met_netmet = {
    "L1 MET (current)": "old_90",
    "L1 MET (con)": ("hf_emu", "signal_con"),
}

ax = plt.subplot()
#netmet
for key, tag in inputs.items():
    with open(netmet_folder.format(tag[0], tag[1]) + "x_efficiency.npy" , 'rb') as f:
        x = np.load(f)
    with open(netmet_folder.format(tag[0], tag[1]) + "y_efficiency.npy" , 'rb') as f:
        y = np.load(f)
    with open(netmet_folder.format(tag[0], tag[1]) + "y_error_efficiency.npy" , 'rb') as f:
        y_errors = np.load(f)

    for i, val in enumerate(y):
        if val > 0.95:
            print(key, 10 * i)
            break

    plt.errorbar(x, y, y_errors, label=key, marker='o', capsize=7, linestyle='none')
    # print(key)
    # print(y)
    
#met
for key, tag in inputs_met.items():
    with open(folder.format(tag) + "x_efficiency.npy" , 'rb') as f:
        x = np.load(f)
    with open(folder.format(tag) + "y_efficiency.npy" , 'rb') as f:
        y = np.load(f)
    with open(folder.format(tag) + "y_error_efficiency.npy" , 'rb') as f:
        y_errors = np.load(f)
    plt.errorbar(x, y, y_errors, label=key, marker='o', capsize=7, linestyle='none')
    
    for i, val in enumerate(y):
        if val > 0.95:
            print(key, 10 * i)
            break

    
    # print(key)
    # print(y)


ax.set_xlabel('PUPPI MET No Mu [GeV]')
ax.set_ylabel('Efficiency')
plt.legend()
plt.savefig("eff_comparison.pdf")
plt.close('all')


ax = plt.subplot()
# for key, tag in inputs_met_netmet.items():
for key, tag in inputs_met.items():
    # print(key, tag)
    # if "con" in key:
        # my_folder = netmet_folder.format(tag[0], tag[1])
    # else:
    my_folder = folder.format(tag)

    with open(my_folder + "x_rate.npy" , 'rb') as f:
        x = np.load(f)
    with open(my_folder + "y_rate.npy" , 'rb') as f:
        y = np.load(f)
    plt.stairs(y, x, label=key)
    
    if "current" in key:
        target_rate = y[90]
    else:
        for i in range(len(y)):
            if y[i] < target_rate:
                print(i)
                break

    # print(key)
    # print(y)
ax.set_xlabel('PUPPI MET No Mu [GeV]')
ax.set_ylabel('Rate')
ax.set_yscale('log')
plt.legend()
plt.savefig("rate_comparison.pdf")
plt.close('all')

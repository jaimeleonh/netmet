from matplotlib import pyplot as plt
import numpy as np

folder = "/vols/cms/jleonhol/cmt/BasePlotting/base/test_250122{}/"
netmet_folder = "/vols/cms/jleonhol/cmt/BDTValidation/base/FT_hf__N_10__MD_6__OBJ_reg_linear__RS_1__SIG_signal_nopum_new__RATIO_0/test_2301/"

inputs = {
    "L1 MET (PUM) > 80": "",
    "L1 MET (No PUM) > 121": "_emu_121",
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
    
#netmet
with open(netmet_folder + "x_efficiency.npy" , 'rb') as f:
    x = np.load(f)
with open(netmet_folder + "y_efficiency.npy" , 'rb') as f:
    y = np.load(f)
with open(netmet_folder + "y_error_efficiency.npy" , 'rb') as f:
    y_errors = np.load(f)
plt.errorbar(x, y, y_errors, label="L1 NetMET (PUM) > 103", marker='o', capsize=7, linestyle='none')

ax.set_xlabel('PUPPI MET No Mu [GeV]')
ax.set_ylabel('Efficiency')
plt.legend()
plt.savefig("eff_comparison.pdf")
plt.close('all')


# ax = plt.subplot()
# for key, tag in inputs.items():
    # with open(folder.format(tag) + "x_rate.npy" , 'rb') as f:
        # x = np.load(f)
    # with open(folder.format(tag) + "y_rate.npy" , 'rb') as f:
        # y = np.load(f)
    # plt.stairs(y, x, label=key)
    
    # if key == "PUM":
        # target_rate = y[80]
    # else:
        # for i in range(len(y)):
            # if y[i] < target_rate:
                # print(i)
                # break

    # print(key)
    # print(y)
# ax.set_xlabel('PUPPI MET No Mu [GeV]')
# ax.set_ylabel('Rate')
# ax.set_yscale('log')
# plt.legend()
# plt.savefig("rate_comparison.pdf")
# plt.close('all')

# Standard imports
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score,roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import os
from scipy.special import expit
import sys
import tasks.hls4ml_plotting as plotting
np.random.seed(0)

#os.environ['PATH'] = '/opt/local/Vivado/2024.1/bin:' + os.environ['PATH']
os.environ['PATH'] = '/opt/local/Vitis_HLS/2024.1/bin:' + os.environ['PATH']
os.environ["XILINX_AP_INCLUDE"] = os.path.expandvars("$CMT_BASE") + "../HLS_arbitrary_Precision_Types/include/"
os.environ["JSON_ROOT"] = os.path.expandvars("$CMT_BASE") + "../json/include"

import xgboost as xg
print("XGBoost version = ",xg.__version__)

import conifer
print("Conifer version = ",conifer.__version__)

import json

# Load XGboost model
modelDir = "hls4ml-tutorial/l1netmet/"
modelFile = "0811/max_depth_4/model.model"
modelConf = "0811/max_depth_4/model.json"
features = ["Jet_0_eta", "Jet_0_phi", "Jet_0_pt", "Jet_1_eta", "Jet_1_phi", "Jet_1_pt", "Jet_2_eta", "Jet_2_phi", "Jet_2_pt", "Jet_3_eta", "Jet_3_phi", "Jet_3_pt", "methf_0_pt", "ntt_0_pt"]

xgb_model = xg.Booster()
xgb_model.load_model('%s/%s' % (modelDir,modelFile))
with open('%s/%s' % (modelDir, modelConf), 'r') as file:
    config = json.load(file)

# print(json.dumps(config, indent=2))

# Conifer configuration (HLS backend)
cfg_hls = conifer.backends.xilinxhls.auto_config()
cfg_hls['OutputDir'] = '%s/conifer_prj_new' % (modelDir)
cfg_hls['XilinxPart'] = 'xcu250-figd2104-2L-e'
cfg_hls['Precision'] = 'ap_fixed<10,3>'

print('Modified Configuration\n' + '-' * 50)
plotting.print_dict(cfg_hls)
print('-' * 50)

# Convert the XGboost model to a Conifer model (0.2b0)
#cnf_model_hls = conifer.model(xgb_model, conifer.converters.xgboost, conifer.backends.xilinxhls, cfg_hls)
cnf_model_hls = conifer.converters.convert_from_xgboost(xgb_model, cfg_hls)
#os.system("rm -rf %s/conifer_prj_hls"%(modelDir))
cnf_model_hls.compile()

# Plot the model weights and thresholds
# cnf_model_hls.profile()



# # Load training/validation data to run the inference
# x_train = np.load("%s/x_train.npy"%(modelDir))
# y_train = np.load("%s/y_train.npy"%(modelDir))
# x_valid = np.load("%s/x_valid.npy"%(modelDir))
# y_valid = np.load("%s/y_valid.npy"%(modelDir))
# print (len(x_train), "events" )
# print (len(x_train[0]),"features")

# # Truth label (offline MET)
# print ("offline MET: ",y_train[:10])

# # Prediction with XGBoost
# train_data = xg.DMatrix( x_train, label=y_train, feature_names=features)
# y_xgb = xgb_model.predict(train_data)
# print("L1 BDT MET (xgboost): ",y_xgb[:10])

# # Prediction with Conifer (HLS Backend)
# y_cnf_hls_= cnf_model_hls.decision_function(x_train)
# y_cnf_hls = expit(y_cnf_hls_)
# y_cnf_hls_=y_cnf_hls_.reshape(-1)
# print("L1 BDT MET (conifer): ",y_cnf_hls_[:10])

# # Overlay the XGBoost and Conifer prediction
# plt.hist(y_xgb, bins=200, alpha=0.5, label="XGBoost")
# plt.hist(y_cnf_hls_, bins=200, alpha=0.5, label="Conifer")
# plt.legend()
# plt.show()
# plt.savefig("test.pdf")

cnf_model_hls.build()

report = cnf_model_hls.read_report()
plotting.print_dict(report)

#import hls4ml
#hls4ml.report.print_vivado_report(cfg_hls['OutputDir'])


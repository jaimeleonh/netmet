import os
import law
import luigi

from analysis_tools.utils import create_file_dir

from tasks.mlmet import MLTraining, BDTTraining, BDTTrainingWorkflow, BDTValidation


class BDTConiferCompilation(BDTTraining):
    precision = luigi.Parameter(default="ap_fixed<24,16>", description="precision to be used in "
        "conifer, default: ap_fixed<24,16>")

    def requires(self):
        return BDTTraining.vreq(self)

    def output(self):
        return {
            "model": self.local_target(f"model_{self.precision.replace(' ', '')}")
        }

    def run(self):
        import numpy as np
        np.random.seed(0)
        import xgboost as xg
        import conifer
        import tasks.hls4ml_plotting as plotting
        import json

        os.environ['PATH'] = '/opt/local/Vitis_HLS/2024.1/bin:' + os.environ['PATH']
        os.environ["XILINX_AP_INCLUDE"] = os.path.join(os.path.expandvars("$CMT_BASE"),
            "../HLS_arbitrary_Precision_Types/include/")
        os.environ["JSON_ROOT"] = os.path.join(os.path.expandvars("$CMT_BASE"), "../json/include")

        modelDir = self.output()["model"].path
        modelFile = self.input()["model"].path
        modelConf = self.input()["json"].path

        X, _ = MLTraining.get_data(self, nfiles=1)

        features = list(X.columns)
        xgb_model = xg.Booster()
        xgb_model.load_model(modelFile)
        with open(modelConf, 'r') as file:
            config = json.load(file)

        cfg_hls = conifer.backends.xilinxhls.auto_config()
        cfg_hls['OutputDir'] = '%s' % (modelDir)
        cfg_hls['ProjectName'] = 'nn_met_calib'
        create_file_dir(cfg_hls['OutputDir'])
        # cfg_hls['XilinxPart'] = 'xcu250-figd2104-2L-e'
        cfg_hls['XilinxPart'] = 'xc7vx690t'
        cfg_hls['Precision'] = self.precision

        cnf_model_hls = conifer.converters.convert_from_xgboost(xgb_model, cfg_hls)
        cnf_model_hls.compile()


class BDTConiferCompilationWorkflow(BDTTrainingWorkflow):
    def requires(self):
        return BDTConiferCompilation.vreq(self, **self.matching_branch_data(BDTConiferCompilation))


class BDTConiferSynthesis(BDTTraining):
    def requires(self):
        return BDTConiferCompilation.vreq(self)

    def output(self):
        return {
            "model": self.requires().output()["model"],
            "report": self.local_target(f"build_report_{self.requires().precision.replace(' ', '')}.txt")
        }

    def run(self):
        import numpy as np
        np.random.seed(0)
        import conifer
        import tasks.hls4ml_plotting as plotting
        import json

        os.environ['PATH'] = '/opt/local/Vitis_HLS/2024.1/bin:' + os.environ['PATH']
        #os.environ['PATH'] = '/opt/local/local.old/Viv2023/Vitis_HLS/2023.1/bin:' + os.environ['PATH']
        print(os.environ['PATH'])
        os.environ["XILINX_AP_INCLUDE"] = os.path.join(os.path.expandvars("$CMT_BASE"),
            "../HLS_arbitrary_Precision_Types/include/")
        os.environ["JSON_ROOT"] = os.path.join(os.path.expandvars("$CMT_BASE"), "../json/include")

        # modelDir = self.output()["model"].path
        # modelFile = self.input()["model"].path
        # modelConf = self.input()["json"].path

        # X, _ = MLTraining.get_data(self, nfiles=1)

        # features = list(X.columns)
        # xgb_model = xg.Booster()
        # xgb_model.load_model(modelFile)
        # with open(modelConf, 'r') as file:
            # config = json.load(file)

        # cfg_hls = conifer.backends.xilinxhls.auto_config()
        # cfg_hls['OutputDir'] = '%s/conifer_prj_new' % (modelDir)
        # create_file_dir(cfg_hls['OutputDir'])
        # cfg_hls['XilinxPart'] = 'xcu250-figd2104-2L-e'
        # cfg_hls['Precision'] = self.precision

        # cnf_model_hls = conifer.converters.convert_from_xgboost(xgb_model, cfg_hls)
        # cnf_model_hls.compile()

        # print('Modified Configuration\n' + '-' * 50)
        # plotting.print_dict(cfg_hls)
        # print('-' * 50)

        cnf_model_hls = conifer.model.load_model(os.path.join(self.input()["model"].path, "nn_met_calib.json"))

        cnf_model_hls.build()

        report = cnf_model_hls.read_report()
        print_report = plotting.print_dict(report)
        print(print_report)

        def write_dict(d, f, indent=0):
            for key, value in d.items():
                f.write(str(key))
                if isinstance(value, dict):
                    f.write("\n")
                    write_dict(value, indent + 1)
                else:
                    f.write(':' + ' ' * (20 - len(key) - 2 * indent) + str(value) + "\n")

        with open(create_file_dir(self.output()["report"].path), "w+") as f:
            write_dict(report, f)
            


class BDTConiferSynthesisWorkflow(BDTTrainingWorkflow):
    def requires(self):
        return BDTConiferSynthesis.vreq(self, **self.matching_branch_data(BDTConiferSynthesis))


class BDTConiferValidation(BDTValidation):
    def requires(self):
        return BDTConiferCompilation.vreq(self)

    def get_output_postfix(self):
        return "_" + self.requires().precision.replace(' ', '')

    def run(self):
        from sklearn.preprocessing import StandardScaler
        import xgboost
        import conifer
        from scipy.special import expit

        os.environ['PATH'] = '/opt/local/Vitis_HLS/2024.1/bin:' + os.environ['PATH']
        os.environ["XILINX_AP_INCLUDE"] = os.path.join(os.path.expandvars("$CMT_BASE"),
            "../HLS_arbitrary_Precision_Types/include/")
        os.environ["JSON_ROOT"] = os.path.join(os.path.expandvars("$CMT_BASE"), "../json/include")

        # os.environ['PATH'] = '/opt/local/Vitis_HLS/2024.1/bin:' + os.environ['PATH']
        # os.environ["XILINX_AP_INCLUDE"] = os.path.join(os.path.expandvars("$CMT_BASE"),
            # "../HLS_arbitrary_Precision_Types/include/")
        # os.environ["JSON_ROOT"] = os.path.join(os.path.expandvars("$CMT_BASE"), "../json/include")

        feature_params = self.config.training_feature_groups()[self.feature_tag]
        scaleData = feature_params.get("scaleData", False)
        trainFrac = feature_params.get("trainFrac", 0.5)

        X, Y = MLTraining.get_data(self, nfiles=-1)
        # X, Y = MLTraining.get_data(self, nfiles=2)

        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))
        X_train = X.sample(frac=trainFrac, random_state=3).dropna()
        Y_train = Y.loc[X_train.index]

        # predict values for efficiency
        modelFile = self.input()["model"].path
        cnf_model_hls = conifer.model.load_model(os.path.join(self.input()["model"].path, "nn_met_calib.json"))
        cnf_model_hls.compile()
        
        Xp = X.drop(X_train.index)
        Yp = cnf_model_hls.decision_function(Xp.to_numpy())
        #Yp = expit(Yp)
        Yp = Yp.reshape(-1)

        Xp_bkg, _ = MLTraining.get_data(self, self.background_dataset, output_y=False, nfiles=100)
        # Xp_bkg, _ = MLTraining.get_data(self, self.background_dataset, output_y=False, nfiles=1)
        if scaleData:
            Xp_bkg[Xp_bkg.columns] = pd.DataFrame(scaler.fit_transform(Xp_bkg))

        Yp_bkg = cnf_model_hls.decision_function(Xp_bkg.to_numpy())
        #Yp_bkg = expit(Yp_bkg)
        Yp_bkg = Yp_bkg.reshape(-1)

        X_bkg_ref = ""
        if self.reference_background_dataset_name:
            reference_background_dataset = self.config.datasets.get(self.reference_background_dataset_name)
            X_bkg_ref, _ = MLTraining.get_data(self, reference_background_dataset, output_y=False, nfiles=100)

        self.get_performance_plots(X, Y, X_train, Y_train, Xp, Yp, Xp_bkg, Yp_bkg, X_bkg_ref)


class BDTConiferValidationWorkflow(BDTTrainingWorkflow):
    def requires(self):
        return BDTConiferValidation.vreq(self, **self.matching_branch_data(BDTConiferValidation))


class BDTConiferComparison(BDTConiferValidation):
    def requires(self):
        return {
            "hls": BDTConiferCompilation.vreq(self),
            "training": BDTTraining.vreq(self)
        }

    def get_output_postfix(self):
        return "_" + self.requires()["hls"].precision.replace(' ', '')

    def output(self):
        postfix = self.get_output_postfix()
        return {
            "diff": self.local_target(f"diff{postfix}.pdf"),
            "diff_log": self.local_target(f"diff{postfix}_log.pdf")
        }

    def run(self):
        from sklearn.preprocessing import StandardScaler
        import xgboost
        import conifer
        from scipy.special import expit
        from matplotlib import pyplot as plt

        self.precision = self.requires()["hls"].precision

        os.environ['PATH'] = '/opt/local/Vitis_HLS/2024.1/bin:' + os.environ['PATH']
        os.environ["XILINX_AP_INCLUDE"] = os.path.join(os.path.expandvars("$CMT_BASE"),
            "../HLS_arbitrary_Precision_Types/include/")
        os.environ["JSON_ROOT"] = os.path.join(os.path.expandvars("$CMT_BASE"), "../json/include")

        # os.environ['PATH'] = '/opt/local/Vitis_HLS/2024.1/bin:' + os.environ['PATH']
        # os.environ["XILINX_AP_INCLUDE"] = os.path.join(os.path.expandvars("$CMT_BASE"),
            # "../HLS_arbitrary_Precision_Types/include/")
        # os.environ["JSON_ROOT"] = os.path.join(os.path.expandvars("$CMT_BASE"), "../json/include")

        feature_params = self.config.training_feature_groups()[self.feature_tag]
        scaleData = feature_params.get("scaleData", False)
        trainFrac = feature_params.get("trainFrac", 0.5)

        X, Y = MLTraining.get_data(self, nfiles=-1)
        # X, Y = MLTraining.get_data(self, nfiles=2)

        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))
        X_train = X.sample(frac=trainFrac, random_state=3).dropna()
        Y_train = Y.loc[X_train.index]

        # predict values for efficiency - Conifer model
        modelFile = self.input()["hls"]["model"].path
        cnf_model_hls = conifer.model.load_model(os.path.join(self.input()["hls"]["model"].path, "nn_met_calib.json"))
        cnf_model_hls.compile()
        
        Xp = X.drop(X_train.index)
        Yp = cnf_model_hls.decision_function(Xp.to_numpy())
        #Yp = expit(Yp)
        Yp_hls = Yp.reshape(-1)

        # predict values for efficiency - Offline model
        modelFile = self.input()["training"]["model"].path

        model = xgboost.XGBRegressor()
        model.load_model(modelFile)
        Yp = model.predict(Xp)

        l1MET_diff = Yp - Yp_hls
        plt.hist(l1MET_diff, bins=100, range=[-20, 20], label=f"L1 NetMET (Exact - {self.precision})",
            histtype='step')
        plt.xlabel(f"L1 NetMET (Exact - {self.precision}) [GeV]")
        plt.savefig(create_file_dir(self.output()["diff"].path))
        plt.close('all')

        plt.hist(l1MET_diff, bins=40, range=[-20, 20], label=f"L1 NetMET (Exact - {self.precision})",
            histtype='step', log=True)
        plt.xlabel(f"L1 NetMET (Exact - {self.precision}) [GeV]")
        plt.savefig(create_file_dir(self.output()["diff_log"].path))
        plt.close('all')


class BDTConiferComparisonWorkflow(BDTTrainingWorkflow):
    def requires(self):
        return BDTConiferComparison.vreq(self, **self.matching_branch_data(BDTConiferComparison))


class BDTConiferEmulatorComparison(BDTConiferComparison):
    dataset_name = luigi.Parameter(default="signal_nopum_new", description="name of the signal dataset, "
        "default: signal_nopum_new")
    def requires(self):
        return {
            "hls": BDTConiferCompilation.vreq(self),
        }

    def run(self):
        from sklearn.preprocessing import StandardScaler
        import xgboost
        import conifer
        from scipy.special import expit
        from matplotlib import pyplot as plt

        self.precision = self.requires()["hls"].precision

        os.environ['PATH'] = '/opt/local/Vitis_HLS/2024.1/bin:' + os.environ['PATH']
        os.environ["XILINX_AP_INCLUDE"] = os.path.join(os.path.expandvars("$CMT_BASE"),
            "../HLS_arbitrary_Precision_Types/include/")
        os.environ["JSON_ROOT"] = os.path.join(os.path.expandvars("$CMT_BASE"), "../json/include")

        # os.environ['PATH'] = '/opt/local/Vitis_HLS/2024.1/bin:' + os.environ['PATH']
        # os.environ["XILINX_AP_INCLUDE"] = os.path.join(os.path.expandvars("$CMT_BASE"),
            # "../HLS_arbitrary_Precision_Types/include/")
        # os.environ["JSON_ROOT"] = os.path.join(os.path.expandvars("$CMT_BASE"), "../json/include")
        
        #self.feature_tag = "hf_emu"
        feature_params = self.config.training_feature_groups()[self.feature_tag]
        scaleData = feature_params.get("scaleData", False)
        trainFrac = feature_params.get("trainFrac", 0.5)

        X, Y = MLTraining.get_data(self, nfiles=-1, dataset=self.config.datasets.get(self.dataset_name))
        # X, Y = MLTraining.get_data(self, nfiles=2)

        # predict values for efficiency - Conifer model
        modelFile = self.input()["hls"]["model"].path
        cnf_model_hls = conifer.model.load_model(os.path.join(self.input()["hls"]["model"].path, "nn_met_calib.json"))
        cnf_model_hls.compile()

        Yp = cnf_model_hls.decision_function(X.to_numpy())
        Yp_hls = Yp.reshape(-1)


        Yp_hls_int = Yp_hls.astype(int)

        # extract values from the ntuple
        self.feature_tag = "metnohf"
        Xp_metnohf, _ = MLTraining.get_data(self, nfiles=-1, output_y=False,
            dataset=self.config.datasets.get(self.dataset_name))

        # print(Xp_metnohf.shape, Yp_hls.shape)
        #print(Xp_metnohf.iloc[132], Yp_hls)
        
        # print(Xp_metnohf["met_0_hwPt"][0], Yp_hls[0])

        # print(type(X), type(Yp_hls))

        l1MET_diff = Xp_metnohf["met_0_hwPt"] - Yp_hls_int
        
        # print(Xp_metnohf["met_0_hwPt"])
        # print(Yp_hls)
        # print(l1MET_diff)
        #print(l1MET_diff[l1MET_diff != 0])
        
        # for elem in X.iterrows():
            # for val in elem:
                # print(val, end=" ")

        plt.hist(l1MET_diff, bins=41, range=[-20.5, 20.5], label=f"L1 NetMET (Emulator - python)",
            histtype='step')
        plt.xlabel(f"L1 NetMET (Emulator - python) [GeV]")
        plt.savefig(create_file_dir(self.output()["diff"].path))
        plt.close('all')

        plt.hist(l1MET_diff, bins=41, range=[-20.5, 20.5], label=f"L1 NetMET (Emulator - python)",
            histtype='step', log=True)
        plt.xlabel(f"L1 NetMET (Emulator - python) [GeV]")
        plt.savefig(create_file_dir(self.output()["diff_log"].path))
        plt.close('all')


class BDTConiferEmulatorComparisonWorkflow(BDTConiferComparisonWorkflow):
    
    def requires(self):
        return BDTConiferEmulatorComparison.vreq(self, **self.matching_branch_data(BDTConiferEmulatorComparison))

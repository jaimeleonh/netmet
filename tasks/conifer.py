import os
import law
import luigi

from analysis_tools.utils import create_file_dir

from tasks.mlmet import MLTraining, BDTTraining, BDTTrainingWorkflow

class BDTSynthesis(BDTTraining):
    precision = luigi.Parameter(default="ap_fixed<10,3>", description="precision to be used in "
        "conifer, default: ap_fixed<10,3>")

    def __init__(self, *args, **kwargs):
        super(BDTSynthesis, self).__init__(*args, **kwargs)

    def requires(self):
        return BDTTraining.vreq(self)

    def output(self):
        return {
            "model_dir": self.local_target("model_dir"),
            "report": self.local_target("build_report.txt")
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

        modelDir = self.output()["model_dir"].path
        modelFile = self.input()["model"].path
        modelConf = self.input()["json"].path

        X, _ = MLTraining.get_data(self, nfiles=1)

        features = list(X.columns)
        xgb_model = xg.Booster()
        xgb_model.load_model(modelFile)
        with open(modelConf, 'r') as file:
            config = json.load(file)

        cfg_hls = conifer.backends.xilinxhls.auto_config()
        cfg_hls['OutputDir'] = '%s/conifer_prj_new' % (modelDir)
        create_file_dir(cfg_hls['OutputDir'])
        cfg_hls['XilinxPart'] = 'xcu250-figd2104-2L-e'
        cfg_hls['Precision'] = self.precision

        cnf_model_hls = conifer.converters.convert_from_xgboost(xgb_model, cfg_hls)
        cnf_model_hls.compile()

        print('Modified Configuration\n' + '-' * 50)
        plotting.print_dict(cfg_hls)
        print('-' * 50)

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
            


class BDTSynthesisWorkflow(BDTTrainingWorkflow):
    def requires(self):
        return BDTSynthesis.vreq(self, **self.matching_branch_data(BDTSynthesis))

from analysis_tools import ObjectCollection, Category, Process, Dataset, Feature, Systematic
from analysis_tools.utils import DotDict
from analysis_tools.utils import join_root_selection as jrs
from plotting_tools import Label
from collections import OrderedDict

from cmt.config.base_config import Config as cmt_config
from cmt.base_tasks.base import Task


class Config(cmt_config):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def add_categories(self, **kwargs):
        categories = []
        return ObjectCollection(categories)

    def add_processes(self):
        processes = [
            Process("background", Label("Background"), color=(87, 144, 252)),
            Process("signal", Label("Signal"), color=(248, 156, 32), isSignal=True),
        ]

        process_group_names = {
            "default": [
                "background",
                "signal",
            ],
        }

        process_training_names = {
            "default": DotDict(
                processes=[
                    "background"
                    "signal"
                ],
                process_group_ids=(
                    (1.0, (0,)),
                    (1.0, (1,)),
                )
            )
        }

        return ObjectCollection(processes), process_group_names, process_training_names


    def create_signal_dataset(self, name, dataset, xs, tags):
        process_name = name
        for key in ["_rew", "_ext"]:
            process_name = process_name.replace(key, "")

        return Dataset(name=name,
            dataset=dataset,
            process=self.processes.get(process_name),
            check_empty=False,
            prefix="gfe02.grid.hep.ph.ic.ac.uk/pnfs/hep.ph.ic.ac.uk/data/cms",
            xs=xs,
            tags=tags,
        )

    def add_datasets(self):
        datasets = [
            Dataset("background",
                folder="/eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/"\
                    "bundocka/ZeroBias/zb24E_NetMET/240628_140446/0000/",
                process=self.processes.get("background"),
                check_empty=False,
                prefix="eoscms.cern.ch/",
            ),
            Dataset("signal",
                folder="/eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/"\
                    "bundocka/Muon0/zmu24E_NetMET/240628_140553/0000/",
                process=self.processes.get("signal"),
                check_empty=False,
                prefix="eoscms.cern.ch/",
                skipFiles=[
                    "/store/group/dpg_trigger/comm_trigger/L1Trigger/bundocka/Muon0/zmu24E_NetMET/240628_140553/0000/nano_130.root",
                    "/store/group/dpg_trigger/comm_trigger/L1Trigger/bundocka/Muon0/zmu24E_NetMET/240628_140553/0000/nano_290.root"
                ]
            ),
            Dataset("background_nopum",
                folder="/eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/"\
                    "bundocka/ZeroBias/zb24I_PUMOff/241224_140730/0000/",
                process=self.processes.get("background"),
                check_empty=False,
                prefix="eoscms.cern.ch/",
            ),
            Dataset("signal_nopum",
                folder="/eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/"\
                    "bundocka/Muon0/zmu24I_PUMOff/241224_140801/0000/",
                process=self.processes.get("signal"),
                check_empty=False,
                prefix="eoscms.cern.ch/",
            ),
             Dataset("background_nopum_new",
                folder="/eos/user/b/bundocka/jec/zb24I_noJEC_noPUS_noPUM/",
                process=self.processes.get("background"),
                check_empty=False,
                prefix="eosuser.cern.ch/",
            ),
            Dataset("signal_nopum_new",
                folder="/eos/user/b/bundocka/jec/zmu24I_noJEC_noPUS_noPUM/",
                process=self.processes.get("signal"),
                check_empty=False,
                prefix="eosuser.cern.ch/",
            ),
        ]
        return ObjectCollection(datasets)

    def add_features(self):
        features = [
            # Feature("jet_pt", "Jet_pt", binning=(30, 0, 150),
                # x_title=Label("jet p_{T}"),
                # units="GeV",),
        ]
        return ObjectCollection(features)

    def add_weights(self):
        weights = DotDict()
        weights.default = "1"

        return weights

    def add_systematics(self):
        systematics = []
        return ObjectCollection(systematics)

    def add_default_module_files(self):
        defaults = {}
        defaults["PreprocessRDF"] = "modules"
        defaults["PreCounter"] = "weights"
        return defaults

    def training_feature_groups(self):
        return {
            "default": {
                "inputs": ["Jet"],
                "inputSums": ["methf", "ntt"],
                "nObj": 4
            },
            "default_emu": {
                "inputs": ["Jet"],
                "inputSums": ["methf", "ntt"],
                "nObj": 4,
                "useEmu": True,
            },
            "default_saturated": {
                "inputs": ["Jet"],
                "inputSums": ["methf", "ntt"],
                "nObj": 4,
                "remove_saturated": True,
            },
            "eg": {
                "inputs": ["Jet", "EG"],
                "inputSums": ["methf", "ntt"],
                "nObj": 4
            }
        }

# config = Config("base", year=2018, ecm=13, lumi_pb=59741)
config = Config("base", year=2024, ecm=13.6, lumi_pb=1)

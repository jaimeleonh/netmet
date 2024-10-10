# coding: utf-8

"""
Preprocessing tasks.
"""

__all__ = []


import abc
import contextlib
import itertools
from collections import OrderedDict, defaultdict
import os
import sys
from subprocess import call
import json
import numpy as np
import pandas as pd

import law
import luigi

from analysis_tools.utils import join_root_selection as jrs
from analysis_tools.utils import import_root, create_file_dir

from cmt.base_tasks.base import (
    Task, ConfigTask, InputData, HTCondorWorkflow, SGEWorkflow, SlurmWorkflow
)

from cmt.util import parse_workflow_file

from tasks.mlmet import MLTraining


class BasePlotting(ConfigTask, law.LocalWorkflow, HTCondorWorkflow):
    signal_dataset_name = luigi.Parameter(default="signal", description="name of the signal dataset, "
        "default: signal")
    background_dataset_name = luigi.Parameter(default="background", description="name of the "
        "background dataset, default: background")
    l1_met_threshold = luigi.IntParameter(default=80, description="value of the MET threshold, "
        "default: 80")
    puppi_met_cut = luigi.IntParameter(default=0, description="minimum cut on the PuppiMET value, "
        "default: 0")

    def __init__(self, *args, **kwargs):
        super(BasePlotting, self).__init__(*args, **kwargs)
        self.signal_dataset = self.config.datasets.get(self.signal_dataset_name)
        self.background_dataset = self.config.datasets.get(self.background_dataset_name)

    def create_branch_map(self):
        return 1

    def requires(self):
        return InputData.req(self, dataset_name=self.signal_dataset_name, file_index=0)
        # I don't really need it, it's just a placeholder
    
    def workflow_requires(self):
        return {"data": InputData.req(self, dataset_name=self.signal_dataset_name)}
        # I don't really need it, it's just a placeholder

    def output(self):
        return {
            key: {
                "rate": self.local_target(f"rate.{key}"),
                "resolution": self.local_target(f"resolution.{key}"),
                "dist": self.local_target(f"distributions.{key}"),
                "met_puppi": self.local_target(
                    f"l1met_vs_puppimet__puppimet_{self.puppi_met_cut}.{key}"),
                "met_puppi_log": self.local_target(
                    f"l1met_vs_puppimet__puppimet_{self.puppi_met_cut}_log.{key}"),
                "efficiency": self.local_target(f"efficiency.{key}"),
            } for key in ["pdf", "png"]
        }
    
    def get_performance_plots(self, X, Y, X_train, Y_train, Xp, Xp_bkg):
        import utils.plotting as plotting
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm, NoNorm

        ###############
        # L1 MET rate #
        ###############

        l1MET_bkg = Xp_bkg['methf_0_pt']

        ax = plt.subplot()
        # rate plots must be in bins of GeV
        xrange = [0, 200]
        bins = xrange[1]

        rateHist = plt.hist(l1MET_bkg, bins=bins, range=xrange, histtype='step', label='L1 MET Rate', cumulative=-1, log=True)
        plt.legend()
        ax.set_xlabel('MET $p_{T}$ [GeV]')
        ax.set_ylabel('Events')
        plt.savefig(create_file_dir(self.output()["pdf"]["rate"].path))
        plt.savefig(create_file_dir(self.output()["png"]["rate"].path))
        plt.close('all')

        # get rate at threshold
        l1MET_fixed_rate = rateHist[0][int(self.l1_met_threshold) * int((xrange[1] / bins))]

        ##################
        # MET Resolution #
        ##################

        l1MET = X['methf_0_pt']
        l1MET_test = l1MET.drop(X_train.index)
        puppiMETNoMu_df_test = Y.drop(X_train.index)
        plt.hist((l1MET_test - puppiMETNoMu_df_test['PuppiMET_pt']), bins=80, range=[-120, 120], label="L1 MET Diff", histtype='step')
        ax.set_xlabel('$\Delta p_{T}$ [GeV]')
        ax.set_ylabel('Events / 3 GeV')
        plt.legend()
        plt.savefig(create_file_dir(self.output()["pdf"]["resolution"].path))
        plt.savefig(create_file_dir(self.output()["png"]["resolution"].path))
        plt.close('all')

        #################
        # Distributions #
        #################

        ax = plt.subplot()
        plt.hist(puppiMETNoMu_df_test['PuppiMET_pt'], bins=100, range=[0, 200], histtype='step', log=True, label="PUPPI MET NoMu")
        plt.hist(l1MET_test, bins=100, range=[0, 200], histtype='step', label="L1MET")
        ax.set_xlabel('MET $p_{T}$ [GeV]')
        ax.set_ylabel('Events / 2 GeV')
        plt.legend(fontsize=16)
        plt.savefig(create_file_dir(self.output()["pdf"]["dist"].path))
        plt.savefig(create_file_dir(self.output()["png"]["dist"].path))
        plt.close('all')

        import awkward as ak

        fig = plt.figure()
        ax = plt.subplot()

        img = plt.hist2d(ak.to_numpy(l1MET_test), ak.to_numpy(puppiMETNoMu_df_test['PuppiMET_pt']),
            bins=[50, 50],
            range=[[self.puppi_met_cut, 200 + self.puppi_met_cut],
                [self.puppi_met_cut, 200 + self.puppi_met_cut]])
        cbar = fig.colorbar(img[3])
        ax.set_xlabel('L1 MET [GeV]')
        ax.set_ylabel('PUPPI MET No Mu [GeV]')
        plt.savefig(create_file_dir(self.output()["pdf"]["met_puppi"].path))
        plt.savefig(create_file_dir(self.output()["png"]["met_puppi"].path))
        plt.close('all')

        fig = plt.figure()
        ax = plt.subplot()
        plt.hist2d(ak.to_numpy(l1MET_test), ak.to_numpy(puppiMETNoMu_df_test['PuppiMET_pt']),
            bins=[50, 50],
            range=[[self.puppi_met_cut, 200 + self.puppi_met_cut],
                [self.puppi_met_cut, 200 + self.puppi_met_cut]],
            norm=LogNorm()
        )
        cbar = fig.colorbar(img[3])
        ax.set_xlabel('L1 MET [GeV]')
        ax.set_ylabel('PUPPI MET No Mu [GeV]')
        plt.savefig(create_file_dir(self.output()["pdf"]["met_puppi_log"].path))
        plt.savefig(create_file_dir(self.output()["png"]["met_puppi_log"].path))
        plt.close('all')

        ##################
        # MET Efficiency #
        ##################

        fig = plt.figure()
        ax = plt.subplot()
        eff_data, xvals, eff_errors = plotting.efficiency(l1MET, Y['PuppiMET_pt'],
            self.l1_met_threshold, 10, 400)
        plt.axhline(0.95, linestyle='--', color='black')
        plt.errorbar(xvals, eff_data, eff_errors, label="L1 MET > " + str(self.l1_met_threshold),
            marker='o', capsize=7, linestyle='none')
        ax.set_xlabel('PUPPI MET No Mu [GeV]')
        ax.set_ylabel('Efficiency')
        plt.legend()
        plt.savefig(create_file_dir(self.output()["pdf"]["efficiency"].path))
        plt.savefig(create_file_dir(self.output()["png"]["efficiency"].path))
        plt.close('all')

    def run(self):
        from sklearn.preprocessing import StandardScaler
        import xgboost

        self.feature_tag = "default"
        feature_params = self.config.training_feature_groups()[self.feature_tag]
        scaleData = feature_params.get("scaleData", False)
        trainFrac = feature_params.get("trainFrac", 0.5)

        X, Y = MLTraining.get_data(self, nfiles=200)

        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))
        X_train = X.sample(frac=trainFrac, random_state=3).dropna()
        Y_train = Y.loc[X_train.index]

        Xp = X.drop(X_train.index)

        Xp_bkg, _ = MLTraining.get_data(self, self.background_dataset, output_y=False, nfiles=50)
        if scaleData:
            Xp_bkg[Xp_bkg.columns] = pd.DataFrame(scaler.fit_transform(Xp_bkg))

        self.get_performance_plots(X, Y, X_train, Y_train, Xp, Xp_bkg)

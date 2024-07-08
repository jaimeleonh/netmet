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

import law
import luigi

from analysis_tools.utils import join_root_selection as jrs
from analysis_tools.utils import import_root, create_file_dir

from cmt.base_tasks.base import (
    Task, ConfigTask, InputData, HTCondorWorkflow, SGEWorkflow, SlurmWorkflow
)

from cmt.util import parse_workflow_file

class MLTrainingTask(ConfigTask):
    # training_config_name = luigi.Parameter(default="default", description="the name of the "
        # "training configuration, default: default")
    # training_category_name = luigi.Parameter(default="baseline_os_odd", description="the name of "
        # "a category whose selection rules are applied to data to train on, default: "
        # "baseline_os_odd")
    feature_tag = luigi.Parameter(default="default", description="the tag of features to use for "
        "the training, default: default")
    architecture = luigi.Parameter(default="dense:256_64_16:relu", description="a string "
        "describing the network architecture, default: dense:256_64_16:relu")
    loss_name = luigi.Parameter(default="wsgce", description="the name of the loss function, "
        "default: mean_absolute_percentage_error")
    # l2_norm = luigi.FloatParameter(default=1e-3, description="the l2 regularization factor, "
        # "default: 1e-3")
    epochs = luigi.IntParameter(default=10, description="number of epochs for training, "
        "default: 10")
    learning_rate = luigi.FloatParameter(default=0.01, description="the learning rate, default: "
        "0.01")
    dropout_rate = luigi.FloatParameter(default=0.1, description="the dropout rate, default: 0.")
    batch_norm = luigi.BoolParameter(default=True, description="whether batch normalization should "
        "be applied, default: True")
    batch_size = luigi.IntParameter(default=500, description="the training batch size, default: "
        "500")
    random_seed = luigi.IntParameter(default=1, description="random seed for weight "
        "initialization, 0 means non-deterministic, default: 1")
    min_feature_score = luigi.FloatParameter(default=law.NO_FLOAT, description="minimum score "
        "for filtering used features, requires the FeatureRanking when not empty, default: empty")
    signal_dataset_name = luigi.Parameter(default="signal", description="name of the signal dataset, "
        "default: signal")
    background_dataset_name = luigi.Parameter(default="background", description="name of the "
        "background dataset, default: background")

    training_id = luigi.IntParameter(default=law.NO_INT, description="when given, overwrite "
        "training parameters from the training with this branch in the training_workflow_file, "
        "default: empty")
    training_workflow_file = luigi.Parameter(description="filename with training parameters",
        default="hyperopt")

    training_hash_params = [
        "feature_tag", "architecture",
        "loss_name", "epochs", "learning_rate", "dropout_rate", "batch_norm",
        "batch_size", "random_seed", "min_feature_score",
        "signal_dataset_name", "background_dataset_name"
        # "training_category_name", "training_config_name", "l2_norm", "event_weights",
    ]

    @classmethod
    def modify_param_values(cls, params):
        if "training_workflow_file" not in params or "training_id" not in params:
            return params
        if params["training_id"] == law.NO_INT:
            return params

        branch_map = parse_workflow_file(
            self.retrieve_file(f"config/{self.training_workflow_file}.yaml"))[1]

        param_names = cls.get_param_names()
        for name, value in branch_map[params["training_id"]].items():
            if name in param_names:
                params[name] = value

        params["training_id"] = law.NO_INT

        return params

    @classmethod
    def create_training_hash(cls, **kwargs):
        def fmt(key, prefix, fn):
            if key not in kwargs:
                return None
            value = fn(kwargs[key])
            if value in ("", None):
                return None
            return prefix + "_" + str(value)
        def num(n, tmpl="{}", skip_empty=False):
            if skip_empty and n in (law.NO_INT, law.NO_FLOAT):
                return None
            else:
                return tmpl.format(n).replace(".", "p")
        parts = [
            #fmt("training_category_name", "TC", str),
            fmt("feature_tag", "FT", lambda v: v.replace("*", "X").replace("?", "Y")),
            fmt("architecture", "AR", lambda v: v.replace(":", "_")),
            fmt("loss_name", "LN", str),
            # fmt("l2_norm", "L2", lambda v: num(v, "{:.2e}")),
            fmt("epochs", "EP", int),
            fmt("learning_rate", "LR", lambda v: num(v, "{:.2e}")),
            fmt("dropout_rate", "DO", num),
            fmt("batch_norm", "BN", int),
            fmt("batch_size", "BS", num),
            fmt("random_seed", "RS", str),
            fmt("min_feature_score", "MF", lambda v: num(v, skip_empty=True)),
            fmt("signal_dataset_name", "SIG", str),
            fmt("background_dataset_name", "BKG", str),
        ]
        return "__".join(part for part in parts if part)

    def __init__(self, *args, **kwargs):
        super(MLTrainingTask, self).__init__(*args, **kwargs)

        # store the training config
        # self.training_config = self.config.training[self.training_config_name]

        # store the category and check for compositeness
        # self.training_category = self.config.categories.get(self.training_category_name)
        # if self.training_category.x("composite", False) and \
                # not self.allow_composite_training_category:
            # raise Exception("training category '{}' is composite, prohibited by task {}".format(
                # self.training_category.name, self))

        # save training features, without minimum feature score filtering applied
        self.training_features = [
            feature for feature in self.config.features
            if feature.has_tag(self.feature_tag)
        ]

        # compute the storage hash
        # print(self.get_training_hash_data())
        self.training_hash = self.create_training_hash(**self.get_training_hash_data())

    def get_training_hash_data(self):
        return {p: getattr(self, p) for p in self.training_hash_params}

    def store_parts(self):
        parts = super(MLTrainingTask, self).store_parts()
        parts["training_hash"] = self.training_hash
        return parts

    # def expand_training_category(self):
        # if self.training_category.x("composite", False):
            # return list(self.training_category.get_leaf_categories())
        # else:
            # return [self.training_category]


class MLTraining(MLTrainingTask):
    def __init__(self, *args, **kwargs):
        super(MLTraining, self).__init__(*args, **kwargs)
        self.signal_dataset = self.config.datasets.get(self.signal_dataset_name)

    def requires(self):
        return InputData.req(self, dataset_name=self.signal_dataset_name, file_index=0)
        # I don't really need it, it's just a placeholder

    def output(self):
        return self.local_target("model.h5")

    def generate_model(self, nInput):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras.layers import Normalization
        model = Sequential()
        if self.batch_norm:
            normalizer = tf.keras.layers.Normalization(input_shape=[nInput,], axis=-1)
            normalizer.adapt(X_train)
            model.add(normalizer)
        arch, act = self.architecture.split(":")
        for i, nneur in enumerate(arch.split("_")):
            if i == 0:
                model.add(Dense(nneur, input_shape=(nInput,), activation=act))
            else:
                model.add(Dense(nneur, activation=act))
            if self.dropout_rate > 0.:
                model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        model.compile(
            loss=self.loss_name,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

        return model

    def run(self):
        import utils.tools as tools
        from sklearn.preprocessing import StandardScaler
        
        feature_params = self.config.training_feature_groups()[self.feature_tag]
        inputs = feature_params.get("inputs", [])
        inputSums = feature_params.get("inputSums", [])
        nObj = feature_params.get("nObj", 4)
        useEmu = feature_params.get("useEmu", False)
        useMP = feature_params.get("useMP", False)
        scaleData = feature_params.get("scaleData", False)

        branches = tools.getBranches(inputs, useEmu, useMP)

        dataset_files = self.signal_dataset.get_files(
            os.path.expandvars("$CMT_TMP_DIR/%s/" % self.config_name), check_empty=True)[:2]
        sig_data = tools.getArrays(dataset_files, branches, len(dataset_files), None)

        # get puppiMETs
        puppiMET, puppiMET_noMu = tools.getPUPPIMET(sig_data)
        puppiMETNoMu_df = tools.arrayToDataframe(puppiMET_noMu, 'puppiMET_noMu', None)

        # define data
        X = sig_df.copy()
        Y = puppiMETNoMu_df.copy()
        scaler = StandardScaler()
        if scaleData:
            X[X.columns] = pd.DataFrame(scaler.fit_transform(X))
        X_train = X.sample(frac=trainFrac, random_state=3).dropna()
        Y_train = Y.loc[X_train.index]

        with tf.device('CPU: 0'):
            model = self.generate_model(X_train.shape[1])
            model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
            model.save(create_file_dir(self.output.path))


class MLTrainingWorkflowBase(Task, law.LocalWorkflow, HTCondorWorkflow, SGEWorkflow, SlurmWorkflow):
    training_workflow_file = MLTrainingTask.training_workflow_file
    output_collection_cls = law.FileCollection

    def __init__(self, *args, **kwargs):
        super(MLTrainingWorkflowBase, self).__init__(*args, **kwargs)

        self.workflow_data, self._file_branch_map = parse_workflow_file(
            self.retrieve_file(f"config/{self.training_workflow_file}.yaml"))
        
    def create_branch_map(self):
        return self._file_branch_map

    def workflow_requires(self):
        return {
            b: self.trace_branch_requires(self.as_branch(b).requires())
            for b in self.workflow_data["require_branches"]
        }

    def trace_branch_requires(self, branch_req):
        return branch_req.requires()

    def output(self):
        return self.requires().output()

    def matching_branch_data(self, task_cls):
        assert(self.is_branch())
        param_names = task_cls.get_param_names()
        return {
            key: value for key, value in self.branch_data.items()
            if key in param_names
        }


class MLTrainingWorkflow(MLTrainingWorkflowBase):
    def requires(self):
        return MLTraining.vreq(self, **self.matching_branch_data(MLTraining))

    def output(self):
        return self.requires().output()

    def run(self):
        pass



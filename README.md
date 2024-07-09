# L1 NetMET
## Installation

```
git clone https://github.com/jaimeleonh/netmet.git
cd netmet
git clone https://gitlab.cern.ch/cms-phys-ciemat/nanoaod_base_analysis.git --branch py3 nanoaod_base_analysis/
git clone https://github.com/jaimeleonh/L1NetMET.git
source setup.sh
law index --verbose #to do only after installation or including a new task
```

## Usage
### Training:
```
law run MLTraining --config-name base --version test
```

Several parameters can be included, see `law run MLTraining --help`.

Launching to htcondor is possible via `law run MLTrainingWorkflow`. Trainings will consider the parameters stored under [config/hyperopt.yaml](https://github.com/jaimeleonh/netmet/blob/main/config/hyperopt.yaml). In case only selected trainings want to be trained, one can add to the previous command `--branches 0,1,3-5`, where the branch numbers are set in [config/hyperopt.yaml, L30](https://github.com/jaimeleonh/netmet/blob/main/config/hyperopt.yaml#L30).

### Validation plots
```
law run MLValidation --config-name base --version test
```

Several parameters can be included, see `law run MLValidation --help`. These parameters will be also used when training.

Similarly, launching to htcondor is possible via `law run MLValidationWorkflow`.

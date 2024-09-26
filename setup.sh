#!/usr/bin/env bash

action() {
    #
    # global variables
    #

    # determine the directory of this file
    cd nanoaod_base_analysis
    #local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    #local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
    export CMT_BASE="DUMMY"
    #export CMT_BASE="/home/jleonhol/netmet/netmet/nanoaod_base_analysis"
    #export CMT_BASE="/home/hep/jleonhol/netmet/nanoaod_base_analysis"
    if [[ "$CMT_BASE" == "DUMMY" ]]; then
        echo "Need to change the path stored in CMT_BASE to the present folder (ending in nanoaod_base_analysis)"
        return "1"
    fi 

    export CMT_ON_LOCAL="1"

    # check if this setup script is sourced by a remote job
    if [ "$CMT_ON_HTCONDOR" = "1" ]; then
        export CMT_REMOTE_JOB="1"
    else
        export CMT_REMOTE_JOB="0"
    fi

    # check if we're on lxplus
    if [[ "$( hostname )" = lxplus*.cern.ch ]]; then
        export CMT_ON_LXPLUS="1"
    else
        export CMT_ON_LXPLUS="0"
    fi

    # check if we're at ic
    if [[ "$( hostname -f )" = *.hep.ph.ic.ac.uk ]]; then
        export CMT_ON_IC="1"
    else
        export CMT_ON_IC="0"
    fi

    # default cern name
    if [ -z "$CMT_CERN_USER" ]; then
        if [ "$CMT_ON_LXPLUS" = "1" ]; then
            export CMT_CERN_USER="$( whoami )"
        elif [ "$CMT_ON_IC" = "0" ]; then
            export CMT_ON_LOCAL="1"
	    export CMT_LOCAL_USER="$( whoami )"
	    #2>&1 echo "please set CMT_CERN_USER to your CERN user name and try again"
            #return "1"
        fi
    fi

    # default ciemat name
    if [ -z "$CMT_IC_USER" ]; then
        if [ "$CMT_ON_IC" = "1" ]; then
            export CMT_IC_USER="$( whoami )"
        # elif [ "$CMT_ON_LXPLUS" = "0" ]; then
            # 2>&1 echo "please set CMT_IC_USER to your CIEMAT user name and try again"
            # return "1"
        fi
    fi

    # default data directory
    if [ -z "$CMT_DATA" ]; then
        if [ "$CMT_ON_LXPLUS" = "1" ]; then
            export CMT_DATA="$CMT_BASE/data"
        else
            # TODO: better default when not on lxplus
            export CMT_DATA="$CMT_BASE/data"
        fi
    fi

    # other defaults
    [ -z "$CMT_SOFTWARE" ] && export CMT_SOFTWARE="$CMT_DATA/software"
    [ -z "$CMT_STORE_LOCAL" ] && export CMT_STORE_LOCAL="$CMT_DATA/store"
    if [ -n "$CMT_IC_USER" ]; then
      [ -z "$CMT_STORE_EOS" ] && export CMT_STORE_EOS="/vols/cms/$CMT_IC_USER/cmt"
    elif [ -n "$CMT_CERN_USER" ]; then
      [ -z "$CMT_STORE_EOS" ] && export CMT_STORE_EOS="/eos/user/${CMT_CERN_USER:0:1}/$CMT_CERN_USER/cmt"
    elif [ -n "$CMT_LOCAL_USER" ]; then
      [ -z "$CMT_STORE_EOS" ] && export CMT_STORE_EOS="/home/${CMT_LOCAL_USER}/netmet/cmt"
    fi
    [ -z "$CMT_STORE" ] && export CMT_STORE="$CMT_STORE_EOS"
    [ -z "$CMT_JOB_DIR" ] && export CMT_JOB_DIR="$CMT_DATA/jobs"
    [ -z "$CMT_TMP_DIR" ] && export CMT_TMP_DIR="$CMT_DATA/tmp"
    [ -z "$CMT_PYTHON_VERSION" ] && export CMT_PYTHON_VERSION="3"

    # specific eos dirs
    [ -z "$CMT_STORE_EOS_PREPROCESSING" ] && export CMT_STORE_EOS_PREPROCESSING="$CMT_STORE_EOS"
    [ -z "$CMT_STORE_EOS_CATEGORIZATION" ] && export CMT_STORE_EOS_CATEGORIZATION="$CMT_STORE_EOS"
    [ -z "$CMT_STORE_EOS_MERGECATEGORIZATION" ] && export CMT_STORE_EOS_MERGECATEGORIZATION="$CMT_STORE_EOS"
    [ -z "$CMT_STORE_EOS_SHARDS" ] && export CMT_STORE_EOS_SHARDS="$CMT_STORE_EOS"
    [ -z "$CMT_STORE_EOS_EVALUATION" ] && export CMT_STORE_EOS_EVALUATION="$CMT_STORE_EOS"
    
    export CMT_REMOTE_PREPROCESSING="1"
    
    if [ -n "$CMT_IC_USER" ]; then
       if [ -n "$CMT_TMPDIR" ]; then
         export TMPDIR="$CMT_TMPDIR"
       else
         export TMPDIR="${CMT_STORE_EOS}/tmp"
       fi
       mkdir -p "$TMPDIR"
    fi

    #if [[ $CMT_IC_USER == jleonhol ]]; then
    #    echo "running export CMT_STORE_EOS_CATEGORIZATION=/vols/cms/khl216/cmt..."
    #    export CMT_STORE_EOS_CATEGORIZATION=/vols/cms/khl216/cmt
    #fi

    # create some dirs already
    mkdir -p "$CMT_TMP_DIR"


    #
    # helper functions
    #

    cmt_pip_install() {
        if [ "$CMT_PYTHON_VERSION" = "2" ]; then
            env pip install --ignore-installed --no-cache-dir --upgrade --prefix "$CMT_SOFTWARE" "$@"
        else
            env pip3 install --ignore-installed --no-cache-dir --upgrade --prefix "$CMT_SOFTWARE" "$@"
        fi
    }
    export -f cmt_pip_install

    cmt_add_py() {
        export PYTHONPATH="$1:$PYTHONPATH"
    }
    export -f cmt_add_py

    cmt_add_bin() {
        export PATH="$1:$PATH"
    }
    export -f cmt_add_bin

    cmt_add_lib() {
        export LD_LIBRARY_PATH="$1:$LD_LIBRARY_PATH"
    }
    export -f cmt_add_lib

    cmt_add_root_inc() {
        export ROOT_INCLUDE_PATH="$ROOT_INCLUDE_PATH:$1"
    }
    export -f cmt_add_root_inc



    #
    # minimal software stack
    #

    # add this repo to PATH and PYTHONPATH
    cmt_add_bin "$CMT_BASE/bin"
    cmt_add_py "$CMT_BASE"
    cmt_add_py "$CMT_BASE/../"
    cmt_add_py "$CMT_BASE/../L1NetMET/netMET/"

    # variables for external software
    export GLOBUS_THREAD_MODEL="none"

    # certificate proxy handling
    [ "$CMT_REMOTE_JOB" != "1" ] && export X509_USER_PROXY="$CMT_BASE/x509up"

    # software that is used in this project
    cmt_setup_software() {
        local origin="$( pwd )"
        local mode="$1"

        if [ "$mode" = "force" ] || [ "$mode" = "force_py" ]; then
            echo "remove software stack in $CMT_SOFTWARE"
            rm -rf "$CMT_SOFTWARE"
        fi

        if [ "$mode" = "force" ] || [ "$mode" = "force_gfal" ]; then
            echo "remove gfal installation in $CMT_GFAL_DIR"
            rm -rf "$CMT_GFAL_DIR"
        fi

        cd "$origin"

        # get the python version
        if [ "$CMT_PYTHON_VERSION" = "2" ]; then
            local pyv="$( python -c "import sys; print('{0.major}.{0.minor}'.format(sys.version_info))" )"
        else
            local pyv="$( python3 -c "import sys; print('{0.major}.{0.minor}'.format(sys.version_info))" )"
        fi

        # ammend software paths
        cmt_add_bin "$CMT_SOFTWARE/bin"
        cmt_add_py "$CMT_SOFTWARE/lib/python$pyv/site-packages:$CMT_SOFTWARE/lib64/python$pyv/site-packages"

        # setup custom software
        if [ ! -d "$CMT_SOFTWARE" ]; then
            echo "installing software stack at $CMT_SOFTWARE"
            mkdir -p "$CMT_SOFTWARE"
            cd "$CMT_SOFTWARE"
            python3 -m venv NetMET
            source NetMET/bin/activate
            pip install pip --upgrade
            pip install tabulate --no-cache-dir
            pip install luigi --no-cache-dir
            pip install yaml --no-cache-dir
            pip install git+https://gitlab.cern.ch/cms-phys-ciemat/analysis_tools.git --no-cache-dir
            pip install git+https://gitlab.cern.ch/cms-phys-ciemat/plotting_tools.git --no-cache-dir
            pip install --no-deps git+https://github.com/riga/law --no-cache-dir
            pip install --no-deps git+https://github.com/riga/plotlib --no-cache-dir
            pip install matplotlib==3.5.1 --no-cache-dir
            pip install tensorflow==2.10.0 --no-cache-dir
            pip install uproot==5.2.1 --no-cache-dir
            pip install pandas==1.5.0 --no-cache-dir
            pip install scikeras==0.12.0 --no-cache-dir
            pip install mplhep==0.3.32 --no-cache-dir
            pip install pyarrow==15.0.0 --no-cache-dir
            pip install tables==3.8.0 --no-cache-dir
            pip install xrootd --no-cache-dir
            pip install fsspec-xrootd --no-cache-dir
            pip install cppyy --no-cache-dir
            pip install envyaml --no-cache-dir
            cd "$origin"
        else
            cd "$CMT_SOFTWARE"
            source NetMET/bin/activate
            cd "$origin"
        fi
    }
    export -f cmt_setup_software

    # setup the software initially when no explicitly skipped
    if [ "$CMT_SKIP_SOFTWARE" != "1" ]; then
        if [ "$CMT_FORCE_SOFTWARE" = "1" ]; then
            cmt_setup_software force
        else
            if [ "$CMT_FORCE_CMSSW" = "1" ]; then
                cmt_setup_software force_cmssw
            else
                cmt_setup_software silent
            fi
        fi
    fi


    #
    # law setup
    #

    export LAW_HOME="$CMT_DATA/law"
    export LAW_CONFIG_FILE="$CMT_BASE/../law.cfg"
    [ -z "$CMT_SCHEDULER_PORT" ] && export CMT_SCHEDULER_PORT="80"
    if [ -z "$CMT_LOCAL_SCHEDULER" ]; then
        if [ -z "$CMT_SCHEDULER_HOST" ]; then
            export CMT_LOCAL_SCHEDULER="True"
        else
            export CMT_LOCAL_SCHEDULER="False"
        fi
    fi
    if [ -z "$CMT_LUIGI_WORKER_KEEP_ALIVE" ]; then
        if [ "$CMT_REMOTE_JOB" = "0" ]; then
            export CMT_LUIGI_WORKER_KEEP_ALIVE="False"
        else
            export CMT_LUIGI_WORKER_KEEP_ALIVE="False"
        fi
    fi

    # try to source the law completion script when available
    which law &> /dev/null && source "$( law completion )" ""
}
action "$@"
cd ..
#voms-proxy-init --voms cms -valid 192:0

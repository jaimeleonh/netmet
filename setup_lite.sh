add_py() {
    export PYTHONPATH="$1:$PYTHONPATH"
}
export -f add_py

if [ ! -d "venv" ]; then
    echo "installing software stack at venv"
    python3 -m virtualenv venv
    source venv/bin/activate
    pip install pip --upgrade
    pip install yaml --no-cache-dir
    pip install git+https://gitlab.cern.ch/cms-phys-ciemat/analysis_tools.git --no-cache-dir
    pip install matplotlib==3.5.1 --no-cache-dir
    pip install uproot==5.2.1 --no-cache-dir
    pip install pandas==1.5.0 --no-cache-dir
    pip install scikeras==0.12.0 --no-cache-dir
    pip install mplhep==0.3.32 --no-cache-dir
    pip install pyarrow==15.0.0 --no-cache-dir
    pip install tables==3.8.0 --no-cache-dir
    pip install xgboost==1.7.5 --no-cache-dir
    pip install jupyter --no-cache-dir
    #pip install conifer==1.5 --no-cache-dir
else
    source venv/bin/activate
fi

add_py "$(pwd)/L1NetMET/netMET/"

# 3dCeusProcessing

## Requirements

* Any version of [Python3.9](https://www.python.org/downloads/)

## Building

### Mac

```shell
git clone https://github.com/TUL-Dev/3dCeusProcessing
cd QusTools
pip install --upgrade pip
python3.9 -m pip install virtualenv
virtualenv --python="python3.9" venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

### Windows

```shell
git clone https://github.com/davidspector67/QusTools.git
cd QusTools
pip install --upgrade pip
python3.9 -m pip install virtualenv
virtualenv --python="python3.9" venv
call venv\scripts\activate.bat
pip install -r requirements.txt
deactivate
```

## Running

### Mac/Linux

```shell
source venv/bin/activate
python paramap.py [ABS_PATH_TO_4D_NIFTI_DATA_FILE] \
[ABS_PATH_TO_SEG_NIFTI_FILE] [X_DIM] [Y_DIM] \ 
[Z_DIM] [ABS_PATH_TO_PARAMAP_DEST_FILE]
deactivate
```

### Windows

```shell
call venv\scripts\activate.bat
python paramap.py [ABS_PATH_TO_4D_NIFTI_DATA_FILE] \
[ABS_PATH_TO_SEG_NIFTI_FILE] [X_DIM] [Y_DIM] \ 
[Z_DIM] [ABS_PATH_TO_PARAMAP_DEST_FILE]
deactivate
```

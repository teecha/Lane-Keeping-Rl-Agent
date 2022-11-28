# Lane changing Rl car

A microscopic traffic simulation platform with vehicle-to-infrastructure network. We use [Flow](https://github.com/flow-project/flow) and SUMO to simulate micro-scopic traffic simulations.

We are in early-devlopement phase, so expect bugs.

# Installation

If you don't have our git repo, download it using the following command.
```bash
git clone https://github.com/teecha/Lane-Keeping-Rl-Agent.git
cd Lane-Keeping-Rl-Agent
```


We assume, you already have conda installed. If not, please install Miniconda or Anaconda. Once, conda is installed, we can install all the dependencies.

```bash
conda env create -f environment.yml
conda activate flow
python setup.py develop
```

Once all dependecies are satisfied, begin install the library.
```bash
pip install -e .
```

Next, we will install sumo binaries. We support Ubuntu 18.04 as of now.

```
scripts/setup_sumo_ubuntu1804.sh
source ~/.bashrc
```

Test the new installation.
```bash
conda acitvate flow
python -m unittest v2i/tests/envs/inMerge/test_twoInMerge.py
```

## Output Video

https://youtu.be/T-yGhOJTagA

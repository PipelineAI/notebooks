# Based on https://github.com/datascienceinc/Skater.git

## Install

### Option 1: without rule lists and without deepinterpreter
pip install -U skater

### Option 2: without rule lists and with deepinterpreter:
1. Ubuntu: pip3 install --upgrade tensorflow (follow instructions at https://www.tensorflow.org/install/ for details and          best practices)
2. sudo pip install keras
3. pip install -U skater==1.1.1b2

### Option 3: For everything included
1. conda install gxx_linux-64
2. Ubuntu: pip3 install --upgrade tensorflow (follow instructions https://www.tensorflow.org/install/ for
   details and best practices)
3. sudo pip install keras
4. sudo pip install -U --no-deps --force-reinstall --install-option="--rl=True" skater==1.1.1b2

## Test It Out
```
python -c "from skater.tests.all_tests import run_tests; run_tests()"
```

#!/bin/bash

python imitate-task.py --config config/square/nn-seg-way.json
python imitate-task.py --config config/square/nn-seg.json
python imitate-task.py --config config/square/nn-way.json
python imitate-task.py --config config/square/nn.json

python imitate-task.py --config config/square/snds-seg-way.json
python imitate-task.py --config config/square/snds-seg.json
python imitate-task.py --config config/square/snds-way.json
python imitate-task.py --config config/square/snds.json
# ========================================================================
# Copyright 2021, The CFL Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================


import os
import pickle

from typing import Dict, List
from datetime import datetime


class ExperimentLog:
    """ Store and preserve important experimental data.

    TODO: still not implemented properly!
    """

    def __init__(self, exp_name: str):
        self.__name = exp_name
        self.__date = datetime.today

        # data
        self.__opt_time: float = 0.0
        self.__n_iterations: float = 0
        self.__mse_per_iter: List[float] = []
        self.__final_mse: float = 0.0
        self.__opt_success: bool = True
        self.__opt_message: str = ""

    @property
    def data(self):
        data_dict: Dict = {}
        data_dict["opt_time"] = self.__opt_time
        data_dict["n_iterations"] = len(self.__mse_per_iter)
        data_dict["mse_per_iter"] = self.__mse_per_iter
        data_dict["final_mse"] = self.__mse_per_iter[-1]
        data_dict["opt_success"] = self.__opt_success
        data_dict["opt_message"] = self.__opt_message

    def store_opt_status(self, succ: bool, time: float = None, msg: str = None):
        self.__opt_time, self.__opt_success, self.__opt_message = time, succ, msg

    def store_mse(self, mse: float):
        self.__mse_per_iter.append(mse)

    def save(self, save: bool = False, save_dir: str = "expdata"):
        os.makedirs(save_dir, exist_ok=True)
        pickle.dump(self, open(os.path.join(save_dir, f'{self.__name}.pickle'), 'wb'))
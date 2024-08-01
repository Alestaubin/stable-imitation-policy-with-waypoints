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

import logging
import sys

# Set the defult log level here
logging_level = logging.INFO

# Change the logging format here
logging.basicConfig(format='%(levelname)s - %(filename)s - %(lineno)d - %(asctime)s - %(message)s',
                    level=logging_level,
                    stream=sys.stdout)

# Create the logger for other classes to use
logger = logging.getLogger('simulation')

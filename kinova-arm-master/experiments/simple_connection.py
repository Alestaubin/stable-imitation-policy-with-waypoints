# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.

import sys
import os

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, Base_pb2



################  IMPORTANT NOTE: remember to pass arguments such as ip, username, and password! #########################
################  hardcoded for ease of use, but this is not the best practice. We use arguments later on. ###############
class CONFIGS:
    IP = "192.168.1.1"
    PORT = 1024
    USERNAME = "admin"
    PASSWORD = "1234"

def example_api_creation():
    '''
    This function creates all required objects and connections to use the arm's services.
    It is easier to use the DeviceConnection utility class to create the router and then
    create the services you need (as done in the other examples).
    However, if you want to create all objects yourself, this function tells you how to do it.
    '''

    # create a error_callback for handling exceptions cleanly
    error_callback = lambda kException: print("_________ callback error _________ {}".format(kException))

    # create a TCP transport object and a router client
    transport = TCPTransport()
    router = RouterClient(transport, error_callback)
    transport.connect(CONFIGS.IP, CONFIGS.PORT)

    # define a session with the given arguments: username and password
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = CONFIGS.USERNAME
    session_info.password = CONFIGS.PASSWORD
    session_info.session_inactivity_timeout = 60000   # (milliseconds)
    session_info.connection_inactivity_timeout = 2000 # (milliseconds)
    session_manager = SessionManager(router)
    session_manager.CreateSession(session_info)
    print("Session created successfully!")

    # create required services
    device_config = DeviceConfigClient(router)
    base = BaseClient(router)

    print(f'Device type: {device_config.GetDeviceType()}')
    print(f'Arm states: {base.GetArmState()}')

    # close the session and disconnect
    session_manager.CloseSession()
    transport.disconnect()

if __name__ == "__main__":
    example_api_creation()
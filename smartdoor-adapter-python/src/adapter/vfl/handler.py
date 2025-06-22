import logging
import time

from datetime import datetime

from generic.api import label_pb2
from generic.api.configuration import ConfigurationItem, Configuration
from generic.api.label import Label, Sort
from generic.api.parameter import Type, Parameter
from generic.handler import Handler as AbstractHandler
from vfl.vfl_connection import VflConnection

def _response(name, channel='vfl', parameters=None):
    """ Helper method to create a response Label. """
    return Label(Sort.RESPONSE, name, channel, parameters=parameters)

def _stimulus(name, channel='vfl', parameters=None):
    """ Helper method to create a stimulus Label. """
    return Label(Sort.STIMULUS, name, channel, parameters=parameters)

class Handler(AbstractHandler):
    """
    This class handles the interaction between AMP and the SmartDoor SUT.
    """

    def __init__(self):
        super().__init__()
        self.sut = None

    def send_message_to_amp(self, raw_message: str):
        """
        Send a message back to AMP. The message from the SUT needs to be converted to a Label.

        Args:
            raw_message (str): The message to send to AMP.
        """
        logging.debug('response received: {label}'.format(label=raw_message))

        if raw_message == 'RESET_PERFORMED':
            # After 'RESET_PERFORMED', the SUT is ready for a new test case.
            self.adapter_core.send_ready()
        else:
            label = self._message2label(raw_message)
            self.adapter_core.send_response(label)

    def start(self):
        """
        Start a test.
        """
        self.sut = VflConnection(self)
        self.sut.connect()


    def reset(self):
        """
        Prepare the SUT for the next test case.
        """
        logging.info('Resetting the SUT for a new test case')
        self.sut.send('RESET')

    def stop(self):
        """
        Stop the SUT from testing.
        """
        logging.info('Stopping the plugin handler')
        self.sut.stop()
        self.sut = None

        logging.debug('Finished stopping the plugin handler')

    def stimulate(self, pb_label: label_pb2.Label):
        """
        Processes a stimulus of a given label at the SUT.

        Args:
            pb_label (label_pb2.Label): stimulus that the Axini Modeling Platform has sent
        """

        label = Label.decode(pb_label)
        sut_msg = self._label2message(label)

        # send confirmation of stimulus back to AMP
        pb_label.timestamp = time.time_ns()
        pb_label.physical_label = bytes(sut_msg, 'UTF-8')
        self.adapter_core.send_stimulus_confirmation(pb_label)

        # leading spaces are needed to justify the stimuli and responses
        logging.info('      Injecting stimulus @SUT: ?{name}'.format(name=label.name))
        self.sut.send(sut_msg)

    def supported_labels(self):
        """
        The labels supported by the adapter.

        Returns:
             [Label]: List of all supported labels of this adapter
        """
        return [
        _stimulus("start", parameters=[Parameter("rounds", Type.INTEGER)]),
        _response("training_started"),
        _response("round_done", parameters=[
            Parameter("round", Type.INTEGER),
            Parameter("accuracy", Type.STRING),
        ]),
        _response("training_done", parameters=[Parameter("accuracy", Type.STRING)]),
        _stimulus("reset"),
        _response("reset_performed"),
        _stimulus("stop"),
        _response("stopped"),
        _response("error", parameters=[Parameter("reason", Type.STRING)]),
        ]


    def default_configuration(self) -> Configuration:
        """
        The default configuration of this adapter.

        Returns:
            Configuration: the default configuration required by this adapter.
        """
        return Configuration([ConfigurationItem(\
            name='endpoint',
            tipe=Type.STRING,
            description='Base websocket URL of the SmartDoor API',
            value='ws://localhost:3001'),
        ])

    def _label2message(self, label: Label) -> str:
        if label.name == "start":
            return f"START:{label.parameters[0].value}"
        return label.name.upper()


    def _message2label(self, msg: str) -> Label:
        parts = msg.split(":")
        name  = parts[0].lower()
        params = []
        if name in {"round_done"}:
            params = [Parameter("round", Type.INTEGER, int(parts[1])),
                    Parameter("accuracy", Type.STRING, parts[2])]
        elif name in {"training_done"}:
            params = [Parameter("accuracy", Type.STRING, parts[1])]
        elif name in {"error"}:
            params = [Parameter("reason", Type.STRING, ":".join(parts[1:]))]
        return Label(Sort.RESPONSE, name, "vfl", parameters=params)


# https://freedium.cfd/https://medium.com/geobeyond/create-ogc-processes-in-pygeoapi-11c0f7d3be61
# https://docs.pygeoapi.io/en/latest/plugins.html
    
import math

from pygeoapi.process.base import BaseProcessor, ProcessorExecuteError

PROCESS_METADATA = {
    # reduced for brevity (see examples of PROCESS_METADATA in pygeoapi/process/hello_world.py)
}

class MyCoolSqrtProcessor(BaseProcessor)
    """My cool sqrt process plugin"""

    def __init__(self, processor_def):
        """
        Initialize object

        :param processor_def: provider definition

        :returns: pygeoapi.process.mycoolsqrtprocess.MyCoolSqrtProcessor
        """

        super().__init__(processor_def, PROCESS_METADATA)

    def execute(self, data):

        mimetype = 'application/json'
        number = data.get('number')

        if number is None:
            raise ProcessorExecuteError('Cannot process without a number')

        try:
            number = float(data.get('number'))
        except TypeError:
            raise ProcessorExecuteError('Number required')

        value = math.sqrt(number)

        outputs = {
            'id': 'sqrt',
            'value': value
        }

        return mimetype, outputs

    def __repr__(self):
        return f'<MyCoolSqrtProcessor> {self.name}'
from datetime import datetime
import logging

# TODO: this is reinforce specific, so make it more generic
# set up simple logging
logging.basicConfig(filename=f'log/reinforce_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

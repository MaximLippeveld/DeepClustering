from torch.utils.tensorboard import SummaryWriter
import logging

def target(output, queue):
    writer = SummaryWriter(output / "tb/rep")

    while True:
        item = queue.get()
        if item == None:
            break

        func, args = item
        getattr(writer, func)(*args)

        for arg in args:
            del arg
    
    writer.close()
    

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler("debug.log", mode="w")
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

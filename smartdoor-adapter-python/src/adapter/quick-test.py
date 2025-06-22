from vfl.handler import Handler
from generic.api.label import Label, Sort      # << new import

h = Handler()
h.start()

stimulus = Label(                         # Sort.STIMULUS == outgoing label
    Sort.STIMULUS,
    name="start",
    channel="vfl",
    parameters={"rounds": 2},             # accepts plain dict here
)
h.stimulate(stimulus.encode())            # .encode() → protobuf

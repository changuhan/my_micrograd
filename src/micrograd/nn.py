import random
from micrograd.engine import Value

class Module:
  def __init__(self):
    self.training = True 
    self._parameters = {}
    self._submodules = {}
  
  @staticmethod
  def _validate_name(name:str) -> None:
    if not isinstance(name, str):
      raise TypeError(f"name must be a str, got {type(name).__name__}")
    if name.strip() == "":
      raise ValueError("name must be a non-empty string")
    if "." in name:
      raise ValueError('name must not contain "." (used for dotted paths)')
  
  def add_parameter(self, name: str, p:Value) -> Value:
    self._validate_name(name)
    if not isinstance(p, Value):
      raise TypeError(f"parameter '{name}' must be a Value/Parameter, got {type(p).__name__}")
    if name in self._parameters:
      raise KeyError(f"parameter '{name}' already exists in this module")
    self._parameters[name] = p
  
    return p
  
  def add_module(self, name: str, m: "Module") -> "Module":
    self._validate_name(name)
    if not isinstance(m, Module): 
      raise TypeError(f"module '{name}' must be a Module, got {type(m).__name__}")
    if name in self._submodules:
      raise KeyError(f"submodule '{name}' already exists in this module")
    self._submodules[name] = m

    return m
  
  def parameters(self):
    params = []
    seen = set()
    
    for p in self._parameters.values():
        # if id(p) not in seen: add and mark seen
        if id(p) not in seen:
          params.append(p)
          seen.add(id(p))

    for child in self._submodules.values():
        for p in child.parameters():
            pid = id(p)
            if pid not in seen:
              seen.add(pid)
              params.append(p)
    
    return params




class Neuron:
  def __init__(self, nin):
    self.w = [(Value(random.uniform(-1,1))) for _ in range(nin)]
    self.b = (Value(random.uniform(-1,1)))

  def __call__(self, x):
    act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]

class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
import tinygrad
from tinygrad.device import Device
from tinygrad.helpers import Context
from tinygrad.nn.datasets import mnist
from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.nn import optim, state
from tinygrad.engine.jit import TinyJit

import time

from model import BasicModel



@TinyJit
def train_step(model, X_train):
  Tensor.training = True  # makes dropout work
  samples = Tensor.randint(batch_size, high=X_train.shape[0])
  X, Y = X_train[samples], Y_train[samples]
  optimiser.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  optimiser.step()
  return loss

if __name__ == "__main__":
  print(f"Setting device={Device.DEFAULT}")
  batch_size = 128
  X_train, Y_train, X_test, Y_test = mnist()

  model = BasicModel()
  optimiser = optim.Adam(state.get_parameters(model))

  acc = (model(X_test).argmax(axis=1) == Y_test).mean()
  print(acc)

  with Context(BEAM=2):
    train_step(model, X_train)

  start = time.time()
  for iteration in range(7000):
    loss = train_step(model, X_train)
    if iteration % 100 == 0:
      Tensor.training = False
      acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
      print(f"step {iteration:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
  end = time.time()

  print()
  print(f"Time taken: {end - start}")

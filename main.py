from tinygrad.nn.datasets import mnist
from model import BasicModel
from tinygrad.tensor import Tensor
from tinygrad import nn

X_train, Y_train, X_test, Y_test = mnist()
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar


model = BasicModel()
acc = (model(X_test).argmax(axis=1) == Y_test).mean()
# NOTE: tinygrad is lazy, and hasn't actually run anything by this point
# print(acc.item())  # ~10% accuracy, as expected from a random model
# print()



# training
optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128
def step():
  Tensor.training = True  # makes dropout work
  samples = Tensor.randint(batch_size, high=X_train.shape[0])
  X, Y = X_train[samples], Y_train[samples]
  optim.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  optim.step()
  return loss


# time things
import timeit
print(timeit.repeat(step, repeat=5, number=1))
print()

# from tinygrad import GlobalCounters, Context
# GlobalCounters.reset()
# with Context(DEBUG=2): 
#   step()


# make go brrrr
from tinygrad.engine.jit import TinyJit
print("\nSetting JIT")
jit_step = TinyJit(step)
print(timeit.repeat(jit_step, repeat=5, number=1))


if __name__ == "__main__":
  for step in range(7000):
    loss = jit_step()
    if step % 100 == 0:
      Tensor.training = False
      acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
      print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")

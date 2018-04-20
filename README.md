Pytorch implementation of CVPR'17 - Local Binary Convolutional Neural Networks.
* paper: http://xujuefei.com/lbcnn.html
* original Torch (Lua) repository: https://github.com/juefeix/lbcnn.torch

Training even MNIST with parameters, stated in the original repository, is incredibly slow. Here is an example of training a toy model -- "2 x {BatchNorm2d(8) -> ConvLBP(8, 16, 3) -> Conv(16, 8, 1)} -> FC(200) -> FC(50) -> FC(10)" -- on MNIST:

```
Epoch 0/5: 100%|██████████| 235/235 [00:06<00:00, 37.74it/s]
Epoch 0 train accuracy: 0.948
Epoch 1/5: 100%|██████████| 235/235 [00:05<00:00, 41.98it/s]
Epoch 1 train accuracy: 0.962
Epoch 2/5: 100%|██████████| 235/235 [00:05<00:00, 42.01it/s]
Epoch 2 train accuracy: 0.969
Epoch 3/5: 100%|██████████| 235/235 [00:05<00:00, 42.04it/s]
Epoch 3 train accuracy: 0.971
Epoch 4/5: 100%|██████████| 235/235 [00:05<00:00, 41.84it/s]
Epoch 4 train accuracy: 0.971
Finished Training. Total training time: 41 sec
Full forward pass: 100%|██████████| 40/40 [00:00<00:00, 100.42it/s]
MNIST test accuracy: 0.974
```

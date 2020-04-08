# ACAL-pytorch
pytorch 1.0 implementation for "augmented cyclic adversarial learning for low resource domain adaptation"

# about   
This repository is unofficial partial implementation of "augmented cyclic adversarial learning for low resource domain adaptation"(ICLR 2019, Ehsan Hosseini-As et al.)   
We implemented the case of supervised setting, especially few shot setting.   

# requirements
- pytorch 1.0   
- CUDA 10.0   
- wandb   

# usage   
run   
```
mkdir dataset
mkdir result
python train.py ./config/digit/config.yaml   
```

# result
We treat few shot adaptation setting from SVHN to MNIST.   
By exploiting wandb sweeping tool, we recoreded 80.5% accuracy on target domain while the paper reported to be about 84%.   

# Unified SAM
This is the official repository for the code used to run the experiments in the paper that proposed the Unified SAM optimizer. There are two types of experiments: numpy experiments and pytorch experiments. 
The pytorch optimizer is adapted from: [https://github.com/weizeming/SAM_AT](https://github.com/weizeming/SAM_AT)


**Sharpness-Aware Minimization: General Analysis and Improved Rates**\
*D. Oikonomou, N. Loizou*\
Paper: [https://openreview.net/forum?id=8rvqpiTTFv](https://openreview.net/forum?id=8rvqpiTTFv)



## Usage

Import the optimizers (with their default hyper-parameters) in Python with

``` python
from unifiedsam import unifiedSAM
opt = unifiedSAM(params, base_opt, rho, lambd, lr, momentum, weight_decay)
```

The user needs to specify the following arguements:
 * `params`: A pytorch model
 * `base_opt`: A pytorch optimizer. In all our experiments we used `torch.optim.SGD`
 * `rho`: The $\rho$ hyper-parameter of Unified SAM
 * `lambd`: The $\lambda$ hyper-parameter of Unified SAM. Currently our code supports any $\lambda\in[0,1]$ as well as the values `lambd = '1/t'` and `lambd = '1-1/t'` which correspond to $\lambda_t=\frac{1}{t}$ and $\lambda_t=1-\frac{1}{t}$
 * `lr`: The learning rate for the base optimizer, i.e. SGD
 * `momentum`: The momentum for the base optimizer, i.e. SGD. In all experiments `momentum=0.9`
 * `weight_decay`: The weight decay for the base optimizer, i.e. SGD. In all experiments `weight_decay=5e-4`

## Citation

If you find our work useful, please consider citing our paper.

```
@inproceedings{
  oikonomou2025sharpness,
  title={Sharpness-Aware Minimization: General Analysis and Improved Rates},
  author={Dimitris Oikonomou and Nicolas Loizou},
  booktitle={{ICLR}},
  year={2025},
}
```

## CRNN
This repository researched the influence of cnn and rnn on crnn.

### data
The training and validation dateset is 364w Synthetic Chinese String Dataset and the dict contains 5990 chinese characters.

#### hardware
GPU: GTX 2080, 11G
CPU: AMD Ryzen 7 3700X 8-Core Processor, 16G

# cnn
| cnn_name 	    | epoch         | accuracy  	| model_size 	|
|--------------	|------------	|------------	|------------   |
| DefaultCNN_512| 10            | 99.4%         |  45.5M        |
| ResNet18_512  | 10            | 99.3%         |  68.0M        |
| DenseNet_128  | 24            | 99.01%        |  21.0M        |
| DenseNet18_256| 4             | 98.18%        |  27.0M        |

## models
[the models trained in baidudisk](https://pan.baidu.com/s/1DCgfjmABsBRqhKMXr-MlTg) 
 passwd: ub4j
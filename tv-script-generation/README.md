# 电视剧剧本生成
经过调参优化后，dlnd_tv_script_generation-zh.ipynb 生成的剧本更加符合语法、更加有意义。
三个模型生成的电视剧剧本保存在了 rnn_tv_scripts.txt 文件中。

## 20180831改动信息
### 修改代码
更新 get_init_cell 的实现方式，确保每个 LSTM 单独创建，保证了 LSTM 不会被 TF 认为是同一个 LSTM 单元。
### 调节参数
原始参数：
\# Number of Epochs
num_epochs = 300
\# Batch Size
batch_size = 128
\# RNN Size
rnn_size = 1024
\# Embedding Dimension Size
embed_dim = 256
\# Sequence Length
seq_length = 16
\# Learning Rate
learning_rate = 0.001
\# Show stats for every n number of batches
show_every_n_batches = 10

改动参数：
\# Number of Epochs
num_epochs = 350
\# Batch Size
batch_size = 256
\# 增大seq_length以便学习句子间的关系
seq_length = 24
\# 增大初始LR以便加快计算速度
learning_rate = 0.01
\# 引入Learning Rate Decay以便获得更好的性能
learning_rate_decay = 0.98
\# 引入min_learning_rate防止LR过小
min_learning_rate = 0.001
\# Show stats for every n number of batches
show_every_n_batches = 100

## 20180831增加内容
data/simpsons/ 增加了 simpsons_script_all_lines.txt，该数据包含 simpsons 全部剧本；

### dlnd_tv_script_generation-zh-all-data.ipynb/py/html
使用 simpsons_script_all_lines.txt 数据进行训练，由于计算量过大，不得不改动参数以加快训练。相对于最新的 dlnd_tv_script_generation-zh.ipynb 改动参数如下：
num_epochs = 10
rnn_size = 256
\# 增大了seq_length以便学习句子间的关系
seq_length = 32
learning_rate = 0.02
learning_rate_decay = 0.999
min_learning_rate = 0.01
show_every_n_batches = 297

## 后续改进
- 后续使用性能更强大的设备训练完整数据集能够获得更高的性能；
- 使用 Adam 之类的深度学习优化算法进一步优化性能；
- 加大训练量有可能会过拟合，采用 Dropout / Early Stop / 限制权值来抑制过拟合；
- 如果缺乏高性能设备，考虑使用 GRU 代替 LSTM 以减轻计算量（具体性能有可能会下降，需要对比实验）。

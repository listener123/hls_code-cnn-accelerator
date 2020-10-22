分块迭代加速器
其中包括BN、POOL、CONV3x3、CONV1x1的分块处理代码
分块处理：设置基础计算模块尺寸定义在top.h中，这样做为了把基础计算模块尽可能并行处理。
每个块处理间保持流水结构：
大致结构如下，
【load_weight】
【load_feature】【compute】【save_result】
                             【load_weight】
                             【load_feature】【compute】【save_result】
                                                      
注：load部分互不影响所以可以并行。

非流水结构，
【load_weight】【load_feature】【compute】【save_result】【load_weight】【load_feature】【  compute】【save_result】

非流水时间=3*(T(【load_weight】)+T(【  load_feature  】)+T(【      compute      】)+T(【save_result】))
流水时间=T(【  load_feature  】)+3*T(【      compute      】)+T(【save_result】)

测试程序：
测试程序为mnist数据集合和lenet网络，测试的是POOL和CONV3x3

版本问题缺陷：
基础计算模块定义没有调参
模型中有一些缓存机制存在冗余
目前模型还是float型计算，如果能精确位宽（比如每个值8位）会一部分速度提升。

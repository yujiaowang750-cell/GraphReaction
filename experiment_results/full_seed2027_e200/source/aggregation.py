"""按索引求均值：先累加，再除以真实条数；没有成员的位置返回零。"""

import torch


def mean_by_index(values, index, count):
    """values [M,D]、index [M] -> mean [count,D]、degree [count]。

    多条记录可以指向同一个目标。分母是记录数量，不是 rate 之和。
    """
    degree = torch.bincount(index, minlength=count).to(values.dtype)
    summed = values.new_zeros((count, values.shape[1])).index_add(0, index, values)
    return summed / degree.clamp_min(1).unsqueeze(1), degree

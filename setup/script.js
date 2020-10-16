import * as tf from '@tensorflow/tfjs'

// 传统for循环
const input = [1, 2, 3, 4] //输入
// 第一层 神经元 是一个二维数组   第一层存储四个神经元 每一个神经元有存储了 权重 四个权重
const w = [
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7]
]
const output = [0, 0, 0, 0]
for (let i = 0; i < w.length; i++) {
    // 遍历每一个神经元             输入怎么计算呢 累加起来
    for (let j = 0; j < input.length; j++) {
        output[i] += input[j] * w[i][j] //算出来每一个神经元的值

    }

}
console.log(output)

// 权重点乘我们的输入
// 向量化的话  gpu加速 就贼快
tf.tensor(w).dot(tf.tensor(input)).print()

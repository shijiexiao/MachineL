import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'


window.onload=()=>{
    const xs=[1,2,3,4]
    const ys=[1,3,5,7]

    // 散点图
    tfvis.render.scatterplot(
        {name:'线性回归训练集'},
        {values:xs.map((x,i)=>({x,y:ys[i]})     )},
        {xAxisDomain:[0,5] , yAxisDomain:[0,8]} ,//x轴变大了 y轴也变宽了
    )
    // 让机器自己学习

    // 定义模型结构  单层单个神经元组成的神经网络
    const model=tf.sequential();//这层的输入一定是上一层的输出 连续的 绝大多数都是sequential

    // 接下来给模型添加层
    // units是神经元的个数
    // 一维数组 1代表特征的数量 也就是 x
    model.add(tf.layers.dense({units:1,inputShape: [1] }))//全链接层
    // 初始化一个神经网络模型

    // 为神经网络模型添加层
    // 设计层的神经元个数 和inputshape

}
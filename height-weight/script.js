import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'


window.onload = async () => {

    // 1.准备身高体重训练数据并归一化
    const heights = [150, 160, 170]
    const weights = [40, 50, 60]
    // 2.训练模型并预测，将结果反归一化为正常数据
    tfvis.render.scatterplot({
        name: '身高体重训练数据'
    }, {
        values: heights.map((x, i) => ({
            x,
            y: weights[i]
        }))
    },{
        xAxisDomain:[140,180],
        yAxisDomain:[30,70]
    }
    )
    // 归一化
    // 压缩数据  
    const inputs=tf.tensor(heights).sub(150).div(20) //sub 减法减去身高最小值操作 除法 除去宽度
    // 每一项都减去150
    const labels=tf.tensor(weights).sub(40).div(20)   // 减去体重最小值
    inputs.print()
    labels.print()


    // 其实就是所有数据先减去它的最小值  然后再除以     最大值减去最小值
    const model = tf.sequential(); //这层的输入一定是上一层的输出 连续的 绝大多数都是sequential

    // 接下来给模型添加层
    // units是神经元的个数
    // 一维数组 1代表特征的数量 也就是 x
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    })) //全链接层
    // 初始化一个神经网络模型

    // 为神经网络模型添加层
    // 设计层的神经元个数 和inputshape

    // 损失函数 和 均方误差 mse 告诉神经网络 你错的有多离谱
    // 对一个模型设置损失函数 //只有一行
    // sigmoidCrossEntropy交叉商  meansquarederoor 均方误差
    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.sgd(0.1), // 学习率
    })
    // 优化器 随即梯度下降  
    console.log('model')

    // 神经网络通过均方误差知道自己错了 就用 优化器 ：随机梯度下降       sgd     神经网络就是通过优化器不断优化自己
    // 改进的方向 
    // 梯度就是方向加大小
    // 偏导数  =》梯度下降法


    // 训练模型并可视化过程 
    // 将训练数据转为tensor
    // 训练模型
    // tfvis 可视化训练过程
   
    //  labels 正确的值
    // 随即梯度下降法
    await model.fit(inputs, labels, {
        batchSize: 3, // 每次要学的数据样本有多大 //3ge 
        epochs: 200, // 超参数
        callbacks: tfvis.show.fitCallbacks({
                name: '训练过程'
            },
            ['loss'],
        )
    })
    // 调参数 第一个学习效率要调  批量大小  epoch迭代次数


    // 运用上述模型进行预测
    
    // 进行预测
    // 将带预测的数据转为tensor
    // 使用训练好的模型进行预测
    // 将输出的tensor 转为普通数据并显示
    const output = model.predict(tf.tensor([180])).sub(150).div(20)
    // 返归一化
    // 预测加过要返归一化后  展示数据
    console.log('预测体重',output.mul(20).add(40).dataSync())


}
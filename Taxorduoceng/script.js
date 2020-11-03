import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {
    getData
} from './data.js';

window.onload = async () => {
    const data = getData(400);

    tfvis.render.scatterplot({
        name: 'XOR 训练数据'
    }, {
        values: [
            data.filter(p => p.label === 1),
            data.filter(p => p.label === 0),
        ]
    });

    // sequential模型
    const model = tf.sequential();
    // 添加第一层 全联接层

    //隐藏层
    model.add(tf.layers.dense({
        units: 4, // 神经元个数为4 
        inputShape: [2], // 长度为2 的一位数组  x，y
        activation: 'relu' //激活函数     所有的线形叠加出来还是线形的
    }));

    //先dense全联接层    神经元个数为1
    model.add(tf.layers.dense({
        units: 1, // 输出一个概率
        activation: 'sigmoid' // 激活函数 
    }));
    //上面都是模型结构


    // 损失函数和优化器
    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1) //优化器 初始化学习效率为0.1
    });

    // 把所有的数据都转化为tensor

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));

    //开始训练 
    await model.fit(inputs, labels, {
        epochs: 10,//
        callbacks: tfvis.show.fitCallbacks({ // 可视化训练过程
                name: '训练效果' //训练标题
            },
            ['loss'] //看他的损失
        )
    });

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([
            [form.x.value * 1, form.y.value * 1]
        ]));
        alert(`预测结果：${pred.dataSync()[0]}`);
    };
};
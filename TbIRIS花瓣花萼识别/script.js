import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getIrisData, IRIS_CLASSES } from './data';

window.onload = async () => {
    // 15%用于作为验证集
    // xTain 训练集的输入特征   yTain 输出的结果       
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);


    const model = tf.sequential();
    // 带来非线性变化的激活函数
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [xTrain.shape[1]],//特征数量
        activation: 'sigmoid'//输出一个概率
    }));
    // 设计神经元个数 必须是输出类别的个数   神经元个数为3
    // inputshape第一层才有
    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax' // 激活函数 
    }));

    model.compile({
        loss: 'categoricalCrossentropy',//交叉墒损失函数 ：就是对数损失函数的多分类版本
        optimizer: tf.train.adam(0.1),//学习效率
        metrics: ['accuracy']  //准确度
    });

        //  准确度的度量
    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest],
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    window.predict = (form) => {
        const input = tf.tensor([[
            form.a.value * 1,
            form.b.value * 1,
            form.c.value * 1,
            form.d.value * 1,
        ]]);
        const pred = model.predict(input);
        alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`);
    };
};
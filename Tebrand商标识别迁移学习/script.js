import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {
    getInputs
} from './data';
import {
    img2x,
    file2img
} from './utils';

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json';
const NUM_CLASSES = 3;
const BRAND_CLASSES = ['android', 'apple', 'windows'];

window.onload = async () => {
    const {
        inputs,
        labels
    } = await getInputs();
    // 抽屉组件 visor  
    const surface = tfvis.visor().surface({
        name: '输入示例',
        styles: {
            height: 250
        }
    });
    inputs.forEach(img => {
        surface.drawArea.appendChild(img);
    });

    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    // 模型结构 卷机层 非常多 
    mobilenet.summary();
    // conv_pw_13_relu
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    // 截断truncated 
    const truncatedMobilenet = tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output //从这里拐出来
    });

    //开始迁移学习模型
    const model = tf.sequential();
    // 拍平 输入形状  高位特征 摊平
    model.add(tf.layers.flatten({
        inputShape: layer.outputShape.slice(1) //输出层的输出形状
        //【null，7，7，256】  切割       取出后三个就行
    }));

    // 双层神经网络
    // 添加 dense层
    model.add(tf.layers.dense({
        units: 10, //超参数
        activation: 'relu'
    }));
    // 分类的一个层
    model.add(tf.layers.dense({
        units: NUM_CLASSES, //分类的个数为3
        activation: 'softmax' //为了多分类
    }));
    // 
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam()
    });

    // 
    const {
        xs,
        ys
    } = tf.tidy(() => {
        // 喂给截断模型
        //合并大tensor
        const xs = tf.concat(inputs.map(imgEl => truncatedMobilenet.predict(img2x(imgEl))));
        // 一维数组转化为tensor
        const ys = tf.tensor(labels);
        return {
            xs,
            ys
        };
    });

    //模型训练
    await model.fit(xs, ys, {
        epochs: 20, //
        callbacks: tfvis.show.fitCallbacks({
                name: '训练效果'
            },
            ['loss'], { // 
                callbacks: ['onEpochEnd'] //
            }
        )
    });

    window.predict = async (file) => {
        const img = await file2img(file);
        //文件转img
        document.body.appendChild(img);
        const pred = tf.tidy(() => {
            const x = img2x(img);
            // 结果截断模型后喂给我们的双层模型
            const input = truncatedMobilenet.predict(x);
            // 再用新模型来预测
            return model.predict(input);
        });


        const index = pred.argMax(1).dataSync()[0];
        setTimeout(() => {
            alert(`预测结果：${BRAND_CLASSES[index]}`);
        }, 0);
    };

    window.download = async () => {
        await model.save('downloads://model');
    };
};
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {
    MnistData
} from './data';

window.onload = async () => {

    const data = new MnistData()
    await data.load()
    const examples = data.nextTestBatch(20) //加载一些验证集 测试集
    //clg 20个数据 0～9 10个数字
    // xs 图片的tensor 20  784  28*28*1  像素 黑白照片  20个图片  每个图片有784个像素值组成


    // 加载数据并且可视化数据
    const surface = tfvis.visor().surface({
        name: 'shuru 例子'
    })
    for (let i = 0; i < 20; i += 1) {
        const imageTensor = tf.tidy(() => {
            return examples.xs
                .slice([i, 0], [1, 784])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);
    }
    const model = tf.sequential() //连续模型 大部分都是这个模型
    //添加层 卷机层
    model.add(tf.layers.conv2d({ // conv  2维
        inputShape: [28, 28, 1], // 接受图片信息的尺寸。图片的宽高 ，个数  灰度图所以是1 如果是彩色的就是3
        kernelSize: 5, //卷机核大小  奇数有中心点 
        filters: 8, // 超参数可以调节的
        strides: 1, //移动一步
        activation: 'relu', //激活函数可以移除掉一些不常用的特征 从0 ，x提取最大值
        kernelInitializer: 'varianceScaling' //加快收敛速度 适合卷机神经网络的初始化方法
    }))
    //提取一轮特征
    model.add(tf.layers.maxPool2d({ //二weidu
        poolSize: [2, 2], //池化的  尺寸 
        strides: [2, 2] //移动的步数   超参数
    }));
    // 特征的组合
    // 再重复另一个卷机 核 maxpool的操作
    // 第一轮横竖 第二轮直角
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16, //提取更加复杂的特征
        strides: 1, //步长为一
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    //重复一下池化层
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    // 结果两轮的提取   
    // 如何把二维变为一维？ 摊平到全联接层
    model.add(tf.layers.flatten());

    //全连接层
    model.add(tf.layers.dense({
        units: 10, // 10个分类
        activation: 'softmax', //多分类激活函数
        kernelInitializer: 'varianceScaling'
    }));

    // 上面就是神经网络 构建
    //     建模成功  开始训练模型

    // 设置损失函数 与优化器

    // 准备训练集  验证集

    // 训练模型并且可视化
    model.compile({
        loss: 'categoricalCrossentropy', //交叉商损失函数
        optimizer: tf.train.adam(), //优化器
        metrics: ['accuracy'] //度量单位 就可以看到准确度
    });

    //中间的tensor就会被清楚掉  就会提高性能
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(1000);
        return [
            d.xs.reshape([1000, 28, 28, 1]), // 训练机  1000张图片   yiweidu转为三weidu
            d.labels
        ];
    });
    //测试集
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(200); //200个图片用来验证训练得好不好
        return [
            d.xs.reshape([200, 28, 28, 1]),
            d.labels
        ];
    });

    // 第三部  训练 调用fit方法来训练
    await model.fit(trainXs, trainYs, {
        validationData: [testXs, testYs],
        batchSize: 500,
        epochs: 20, //超参数
        callbacks: tfvis.show.fitCallbacks({
                name: '识别数字训练效果'
            },
            //损失 验证集的损失  精确度 验证集的精确度  
            ['loss', 'val_loss', 'acc', 'val_acc'], {
                callbacks: ['onEpochEnd'] // 否则会出现2个图表
            }
        )
    });
    const canvas = document.querySelector('canvas');

    canvas.addEventListener('mousemove', (e) => {
        if (e.buttons === 1) { // 按着鼠标的左键
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgb(255,255,255)'; // 纯白图片
            ctx.fillRect(e.offsetX, e.offsetY, 25, 25); //juxin   
        }
    });

    window.clear = () => {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.fillRect(0, 0, 300, 300);
    };

    clear();

    window.predict = () => {
        const input = tf.tidy(() => {
            // canvas转化为tensor 
            return tf.image.resizeBilinear( // 
                    tf.browser.fromPixels(canvas), //fromPixels接受cnavas 
                    [28, 28],
                    true // 四个边角
                ).slice([0, 0, 0], [28, 28, 1]) //切蛋糕  转化为黑白图片
                .toFloat()
                .div(255)
                .reshape([1, 28, 28, 1]); //1个图片跟训练数据保持一致
            //做个过归一化
        });
        const pred = model.predict(input).argMax(1);
        alert(`预测结果为 ${pred.dataSync()[0]}`);
    };

};
import * as speechCommands from '@tensorflow-models/speech-commands';

const MODEL_PATH = 'http://127.0.0.1:8080';
let transferRecognizer;
let curIndex = 0;
// 迁移学习器 使用二进制的数据
window.onload = async () => {
    // 模型静态服务地址
    // 识别器
    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        null,
        MODEL_PATH + '/speech/model.json',
        MODEL_PATH + '/speech/metadata.json',
    );

    // 识别器的模型  确保加载完
    await recognizer.ensureModelLoaded();
    // 创建迁移学习器
    transferRecognizer = recognizer.createTransfer('轮播图');
    // 不需要区采集数据了，我们用加载 二进制文件  也就是我的训练数据
    const res = await fetch(MODEL_PATH + '/slider/data.bin');
    // 转化为arraybuffer

    const arrayBuffer = await res.arrayBuffer();
    // 加载 arraybuffer
    transferRecognizer.loadExamples(arrayBuffer);
    // label也保存下来了

    // 训练一下 
    await transferRecognizer.train({
        epochs: 30
    });

    console.log('done');
};

// 监听开关
window.toggle = async (checked) => {
    if (checked) {
        // 迁移学习器的listen 方法
        await transferRecognizer.listen(result => {
            const {
                scores
            } = result;
            const labels = transferRecognizer.wordLabels();//获取所有的labels
            // 再拿到最大值的index

            const index = scores.indexOf(Math.max(...scores));
            // 获得到得分最高的label
            // 传给play方法
            window.play(labels[index]);
        }, {
            overlapFactor: 0,
            probabilityThreshold: 0.5 // yuzhi
        });
    } else {
        transferRecognizer.stopListening();
    }
};

window.play = (label) => {
    const div = document.querySelector('.slider>div');
    if (label === '上一张') {
        if (curIndex === 0) {
            return;
        }
        curIndex -= 1;
    } else {
        if (curIndex === document.querySelectorAll('img').length - 1) {
            return;
        }
        curIndex += 1;
    }
    div.style.transition = "transform 1s"
    div.style.transform = `translateX(-${100 * curIndex}%)`;
};
import * as speechCommands from '@tensorflow-models/speech-commands';
import * as tfvis from '@tensorflow/tfjs-vis';

const MODEL_PATH = 'http://127.0.0.1:8080';
let transferRecognizer; //迁移学习器

//在浏览器中收集中文语音训练数据
window.onload = async () => {
    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        null,
        MODEL_PATH + '/speech/model.json', // 自定义模型的url
        MODEL_PATH + '/speech/metadata.json'
    );
    //创造迁移学习器
    await recognizer.ensureModelLoaded();

    // 创造迁移学习器   name：‘都可以写’
    transferRecognizer = recognizer.createTransfer('轮播图');
};
//收集语音
window.collect = async (btn) => {
    //禁用
    btn.disabled = true;
    // inneText  获取btn的label文字
    const label = btn.innerText;
    // 迁移学习期的语音训练素材     
    await transferRecognizer.collectExample(
        label === '背景噪音' ? '_background_noise_' : label
    );
    // 收集完后 回复btn  采样率和声卡有关
    btn.disabled = false;
    document.querySelector('#count').innerHTML = JSON.stringify(transferRecognizer.countExamples(), null, 2);
};

//训练按钮
window.train = async () => {
    //和fit方法非常像
    await transferRecognizer.train({
        epochs: 30,
        callback: tfvis.show.fitCallbacks({
                name: '训练效果' //图标的标题
            },
            ['loss', 'acc'], { // ‘显示他的损失 和他的准确度
                callbacks: ['onEpochEnd'] // 
            }
        )
    });
};

window.toggle = async (checked) => {
    // 开的话
    if (checked) {
        // 迁移学习器
        await transferRecognizer.listen(result => {
            const {
                scores
            } = result;
            // 
            const labels = transferRecognizer.wordLabels();
            // 拿到最大的分数       所在的index
            const index = scores.indexOf(Math.max(...scores));
            拿到labels
            console.log(labels[index]);
        }, {
            overlapFactor: 0, // 用于控制识别的频率  0.99就会识别得狠频繁
            probabilityThreshold: 0.75 //
        });
    } else {
        transferRecognizer.stopListening();
    }
};

window.save = () => {
    // 转为二进制数据 保存到data.bin 文件里面
    const arrayBuffer = transferRecognizer.serializeExamples();
    const blob = new Blob([arrayBuffer]);
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = 'data.bin'; //下载名称
    // 模拟点击  web标准  来做
    link.click();
};
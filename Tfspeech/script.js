import * as speechCommands from '@tensorflow-models/speech-commands';

const MODEL_PATH = 'http://127.0.0.1:8080/speech';

window.onload = async () => {
    // 1. 建一个识别器
    const recognizer = speechCommands.create(
        'BROWSER_FFT', //fft 傅立叶变换
        null, // 识别默认的单词
        MODEL_PATH + '/model.json',
        MODEL_PATH + '/metadata.json' // 自定义源信息data
    );

    await recognizer.ensureModelLoaded(); //调用识别器 加载

    //recognizer 。listen 调出麦克风来监听
    // 
    const labels = recognizer.wordLabels().slice(2);
    // 

    const resultEl = document.querySelector('#result');
    resultEl.innerHTML = labels.map(l => `
        <div>${l}</div>
    `).join('');

    // 监听
    recognizer.listen(result => {
        const { scores } = result;
        const maxValue = Math.max(...scores);// 然后提取其中最大值
        const index = scores.indexOf(maxValue) - 2; //取出最大值所在的index
        // 放到labels就可以输出
        resultEl.innerHTML = labels.map((l, i) => `
        <div style="background: ${i === index && 'green'}">${l}</div>
        `).join('');
    }, {
        overlapFactor: 0.3,//设置的越高，识别的次数越多 波形里面 99%覆盖 
        probabilityThreshold: 0.85 // 可能性yuzhi 我说的单词必须和他训练的单词有75的相似度才打印
    });
};
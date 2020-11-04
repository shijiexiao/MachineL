const IMAGE_SIZE = 224;

const loadImg = (src) => {
    return new Promise(resolve => {
        const img = new Image();
        img.crossOrigin = "anonymous"; // 图片连接8080端口 跨域 来允许
        img.src = src;
        img.width = IMAGE_SIZE;
        img.height = IMAGE_SIZE;
        img.onload = () => resolve(img); //加载图片
    });
};
export const getInputs = async () => {
    const loadImgs = [];
    const labels = [];// 
    for (let i = 0; i < 30; i += 1) {
        ['android', 'apple', 'windows'].forEach(label => {
            const src = `http://127.0.0.1:8080/brand/train/${label}-${i}.jpg`;
            const img = loadImg(src);
            loadImgs.push(img);
            labels.push([
                label === 'android' ? 1 : 0,
                label === 'apple' ? 1 : 0,
                label === 'windows' ? 1 : 0,
            ]);
        });
    }
    const inputs = await Promise.all(loadImgs); //同时加载这些图片
    return {
        inputs,
        labels,
    };
}
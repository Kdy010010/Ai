const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

async function loadDataFromFile() {
  const data = fs.readFileSync('data.txt', 'utf-8').split('\n');
  return data;
}

async function preprocessTextData(textData) {
  // 텍스트 데이터 전처리 작업 수행 (예: 토큰화, 벡터화 등)
  // 본 예시에서는 단어 수를 세어 사용합니다.
  const wordCounts = {};
  textData.forEach(sentence => {
    const words = sentence.split(' ');
    words.forEach(word => {
      wordCounts[word] = (wordCounts[word] || 0) + 1;
    });
  });
  return wordCounts;
}

async function trainModel(wordCounts) {
  // 텍스트 생성 모델 구성 및 학습
  // 본 예시에서는 단어 수를 기반으로 가장 많이 나온 단어를 예측하는 모델을 사용합니다.
  const words = Object.keys(wordCounts);
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: words.length, inputShape: [words.length] }));
  model.compile({ loss: 'categoricalCrossentropy', optimizer: 'adam' });

  const xs = tf.eye(words.length);
  const ys = tf.oneHot(words.indexOf('target_word'), words.length);
  await model.fit(xs, ys, { epochs: 100 });

  return model;
}

async function generateTextUsingModel(model, startingWord, wordCounts) {
  // 모델을 사용하여 텍스트 생성
  const words = Object.keys(wordCounts);
  let inputIndex = words.indexOf(startingWord);
  const input = tf.oneHot(inputIndex, words.length);
  const prediction = model.predict(input);
  const predictedIndex = tf.argMax(prediction).dataSync()[0];
  const predictedWord = words[predictedIndex];

  return predictedWord;
}

async function main() {
  const textData = await loadDataFromFile();
  const wordCounts = await preprocessTextData(textData);
  const model = await trainModel(wordCounts);

  const startingWord = 'hello'; // 시작 단어
  const generatedWord = await generateTextUsingModel(model, startingWord, wordCounts);

  console.log(`시작 단어 '${startingWord}' 다음 예측 단어: ${generatedWord}`);
}

main().catch(error => console.error(error));

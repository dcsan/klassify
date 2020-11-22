// require('@tensorflow/tfjs-node')
import * as tf from "@tensorflow/tfjs-node";
import * as sentenceEncoder from "@tensorflow-models/universal-sentence-encoder";
const debug = require('debug-levels')('TfModel')

import * as path from 'path'

// total different categories
const N_CLASSES = 3;
const tags = [
  'BOOK', 'RUN', 'PLAY'
]

class TfModel {
  modelPath: string
  modelUrl: string
  encoder: any
  model: any
  loaded: boolean = false

  constructor(modelName = 'tfModel') {
    this.modelPath = path.join(__dirname, 'data', 'modelCache', modelName)
    this.modelUrl = `file://${this.modelPath}`
  }

  async load() {
    if (this.loaded) return
    this.encoder = await sentenceEncoder.load()
    this.loaded = true
  }

  async encodeData(tasks: any) {
    const sentences = tasks.map(t => t.text.toLowerCase());
    const embeddings = await this.encoder.embed(sentences);
    return embeddings;
  }

  // force = ignore cached model
  async trainModel(utterances: any, useCache: boolean = false) {
    await this.load()

    if (useCache) {
      try {
        const modelFile = `${this.modelUrl}/model.json` // annoying TF glitch
        const loadedModel = await tf.loadLayersModel(
          modelFile
        );
        debug.log("Using existing model");
        this.model = loadedModel
        return loadedModel;
      } catch (err) {
        debug.log("err loading model", err);
        debug.log("Training new model");
      }
    }
    const xTrain = await this.encodeData(utterances);

    // TODO - make this more dynamic. just testing for now with 3 labels
    const labels = (
      utterances.map(t => {
        // const yt = [t.icon === "BOOK" ? 1 : 0, t.icon === "RUN" ? 1 : 0]
        let yt
        switch (t.icon) {
          case 'PLAY':
            yt = [1, 0, 0]
            break
          case 'RUN':
            yt = [0, 1, 0]
            break
          case 'BOOK':
            yt = [0, 0, 1]
            break
        }
        debug.log(t, yt)
        return yt
      })
    );
    const yTrain = tf.tensor2d(labels)
    const inputShape = [xTrain.shape[1]]
    const model = tf.sequential();

    debug.log('labels', labels)
    debug.log('yTrain', {
      xTrain, yTrain, inputShape
    })
    // debug.log('xTrain', xTrain)

    model.add(
      tf.layers.dense({
        inputShape: inputShape,
        activation: "softmax",
        units: N_CLASSES
      })
    );

    // model.add(
    //   tf.layers.dense({
    //     inputShape: [xTrain.shape[1]],
    //     activation: "softmax",
    //     units: N_CLASSES
    //   })
    // );
    // model.add(
    //   tf.layers.dense({
    //     inputShape: [xTrain.shape[1]],
    //     activation: "softmax",
    //     units: N_CLASSES
    //   })
    // );

    model.compile({
      loss: "categoricalCrossentropy",
      optimizer: tf.train.adam(0.001),
      metrics: ["accuracy"]
    });

    // const lossContainer = document.getElementById("loss-cont");

    await model.fit(xTrain, yTrain, {
      batchSize: 32,
      validationSplit: 0.1,
      shuffle: true,
      epochs: 150,
      // callbacks: tfvis.show.fitCallbacks(
      //   lossContainer,
      //   ["loss", "val_loss", "acc", "val_acc"],
      //   {
      //     callbacks: ["onEpochEnd"]
      //   }
      // )
    });

    this.model = model

    debug.log('writing model', this.modelUrl)
    await model.save(this.modelUrl);

    return model;
  }

  async predict(input: string, _threshold: number = 0.5) {
    if (!input.trim().includes(" ")) {
      return null;
    }

    const xPredict = await this.encodeData([{ text: input }])

    const tensor = await this.model.predict(xPredict);
    const pdata: number[] = await tensor.data()

    const pcts = pdata.map(p => Math.round(p * 100))

    const confidence = Math.max(...pdata)
    const tagId = pdata.indexOf(confidence)
    const tagName = tags[tagId]
    console.table({
      input, tagId, tagName, confidence, ...pcts
    })
    return tagName

    // if (predictions[0] > threshold) {
    //   return "BOOK";
    // } else if (predictions[1] > threshold) {
    //   return "RUN";
    // } else {
    //   return null;
    // }
  };

}

export { TfModel };

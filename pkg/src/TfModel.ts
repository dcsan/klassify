// require('@tensorflow/tfjs-node')
import * as tf from "@tensorflow/tfjs-node";
import * as sentenceEncoder from "@tensorflow-models/universal-sentence-encoder";
const debug = require('debug-levels')('TfModel')
import * as _ from 'lodash'

import * as path from 'path'

// total different categories

class TfModel {
  modelPath: string
  modelUrl: string
  encoder: any
  model: any
  loaded: boolean = false
  uniqueTags: string[] = []

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
    // TODO - make this more dynamic based on position in tags
    // just testing for now with 3 labels
    const allTags = utterances.map(t => t.tag)
    this.uniqueTags = _.uniq(allTags)

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

    // returns an array like [0,0,1,0] for each entry
    const labels = (
      utterances.map(utt => {
        const pos = this.uniqueTags.indexOf(utt.tag)
        const mat = new Array(this.uniqueTags.length).fill(0)
        mat[pos] = 1
        return mat
      })
    );
    const yTrain = tf.tensor2d(labels)
    const inputShape = [xTrain.shape[1]]
    const model = tf.sequential();

    // debug.log('labels', labels)
    // debug.log('yTrain', {
    //   tags: this.uniqueTags,
    //   labels,
    //   xTrain, yTrain, inputShape
    // })
    // debug.log('xTrain', xTrain)

    model.add(
      tf.layers.dense({
        inputShape: inputShape,
        activation: "softmax",
        units: this.uniqueTags.length // number of classes for classifier
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
    input = input.trim()
    if (!input) {
      debug.warn('empty input to predictor')
      return
    }
    const xPredict = await this.encodeData([{ text: input }])
    const tensor = await this.model.predict(xPredict);
    const pdata: number[] = await tensor.data()

    // const pcts = pdata.map(p => Math.round(p * 100))
    // find most confident item
    const confidence = Math.max(...pdata)
    const tagIndex = pdata.indexOf(confidence)
    const tag = this.uniqueTags[tagIndex]

    // debug.log('prediction', {
    //   pdata,
    //   confidence,
    //   tagIndex,
    //   tag
    // })

    // const vals = pcts.join(',')
    const others: any = this.uniqueTags.map((tagName, index) => {
      const confidence = pdata[index]
      return [tagName, confidence]
    })

    const result = {
      input,
      tag,
      confidence,
      others
    }

    // debug.log('result', result)
    return result

  };

}

export { TfModel };

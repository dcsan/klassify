// import { strict as assert } from 'assert';
import { Klassify } from './Klassify'
import { TfModel } from "./TfModel";

// const kls = new Klassify()
// const result = kls.hello('alpha')
// // console.log('result:', result)
// console.assert(result === 'hello alpha', 'kls.hello')
// assert.equal(result, 'hello axpal')

// let opts: any = {}

const useModelCache = false
// const useModelCache = true

const TestRunner = {

  async run() {
    const testModel = new TfModel('test2')
    await testModel.load()

    const bookData = require("./data/inputs/book.json");
    const runData = require("./data/inputs/run.json");
    const playData = require("./data/inputs/play.json");

    const utterances = [].concat(bookData, runData, playData);
    await testModel.trainModel(utterances, useModelCache)
    await TestRunner.predict(testModel)
  },

  async predict(testModel) {
    const inputs = [
      'go to library',
      'read about animals',
      'get new passport',
      'do more exercise',
      'go to the gym',
      'go to the club',
      'dance your ass off',
      'play games',
      'play with friends',
      'play games',
    ]
    inputs.map(async input => {
      const result = await testModel.predict(input, 0.5)
      console.log(input, result)
    })
  }

}

TestRunner.run()

export { TestRunner }

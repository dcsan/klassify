// import { strict as assert } from 'assert';
import { Klassify } from './Klassify'
import { TfModel } from "./TfModel";

const bookData = require("./data/book.json");
const runData = require("./data/run.json");

// const kls = new Klassify()
// const result = kls.hello('alpha')
// // console.log('result:', result)
// console.assert(result === 'hello alpha', 'kls.hello')
// assert.equal(result, 'hello axpal')

let opts: any = {}

const TestRunner = {

  async run() {
    const testModel = new TfModel('test2')
    await testModel.load()
    const utterances = bookData.concat(runData);
    await testModel.trainModel(utterances)
    await TestRunner.predict(testModel)
  },

  async predict(testModel) {
    const input = 'go to library'
    const result = await testModel.predict(input, 0.5)
    console.log('pr:', input, result)
  }

}

TestRunner.run()

export { TestRunner }

import { TfClassifier } from "./TfClassifier";
import { readCsvFile } from './FileUtils'

const debug = require('debug-levels')('TestRunner')

// import { Klassify } from './Klassify'

// const useModelCache = false
const useModelCache = true

const TestRunner = {

  async run() {
    const testModel = new TfClassifier('test2')
    await testModel.load()

    const utterances = await readCsvFile('./data/inputs/train.csv')
    // debug.log('utterances', utterances)
    // @ts-ignore
    utterances.map((utt: any) => {
      console.log('utt', utt.tag, utt.text)
    })

    await testModel.trainModel(utterances, useModelCache)
    await TestRunner.predict(testModel)
  },

  async predict(testModel) {
    const testLines = await readCsvFile('./data/inputs/test.csv')
    console.log('passed\tactual\texpect\tconfidence\t\ttext')
    testLines.map(async line => {
      const prediction = await testModel.predict(line.text.trim())
      const passed = prediction.tag === line.tag ? '√' : 'x'
      console.log(`${passed} \t${line.tag} \t${prediction.tag} \t${prediction.confidence}\t${line.text.trim()}`)
      // console.log(prediction.others)
    })
  }

}

TestRunner.run()

export { TestRunner }

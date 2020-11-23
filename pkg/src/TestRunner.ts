import { TfClassifier } from "./TfClassifier";
import { readCsvFile } from './FileUtils'
import chalk from 'chalk'

// const debug = require('debug-levels')('TestRunner')
const useCache = false

const TestRunner = {

  async prepare() {
    const testModel = new TfClassifier('test2')
    await testModel.loadEncoder()
    await testModel.loadCsvInputs('./data/inputs/train.csv')
    await testModel.trainModel({ useCache: useCache })
    return testModel
  },

  async predict(testModel: TfClassifier) {
    const testLines = await readCsvFile('./data/inputs/test.csv')
    console.log('passed\tactual\texpect\tconfidence\t\ttext')
    testLines.map(async line => {
      const prediction = await testModel.classify(line.text)
      const passed = prediction?.tag === line.tag ? chalk.green('√ PASS') : chalk.red('X FAIL')
      const output = (`${passed} \t${line.tag} \t${prediction?.tag} \t${prediction?.confidence}\t${line.text.trim()}`)
      console.log(output)
      console.assert(prediction?.tag === line.tag, chalk.red(`FAILED classify \n${line.text}\n`), prediction?.found?.text)
      // console.log(prediction.others)
    })
  },

  async runSuite() {
    const testModel = await TestRunner.prepare()
    await TestRunner.predict(testModel)
  }

}

TestRunner.runSuite()

export { TestRunner }

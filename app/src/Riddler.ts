import { TfClassifier } from '@dcsan/klassify'
import { readCsvFile } from '@dcsan/klassify'
const debug = require('debug-levels')('Riddler')
import * as _ from 'lodash'

const useModelCache = true

const Riddler = {
  async init() {
    const testModel = new TfClassifier('riddles')
    await testModel.load()  // load the sentenceEncoder

    let utterances = await readCsvFile('./data/riddles/riddle-logs.csv', __dirname)
    // debug.log('utterances', utterances)
    // @ts-ignore
    // utterances.map((utt: any) => {
    //   // console.log('utt', utt.tag, '\t', utt.text)
    // })
    utterances = utterances.filter(utt => utt.text && utt.tag) // remove empty items
    console.table(utterances, ['tag', 'text'])

    await testModel.trainModel(utterances, useModelCache)
    const tests = [
      'are the bikes real?',
      'Did anyone go to the airport?'
    ]
    tests.map(async (item) => {
      let result: any = await testModel.predict(item, { maxHits: 5 })
      const matched = utterances.find((utt: any) => {
        return (utt.tag === result.tag)
      })
      debug.log('result', result, matched)
    })
  }
}

Riddler.init()


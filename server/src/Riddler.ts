import { TfClassifier, ITaggedInput, IMatch } from '@dcsan/klassify'
const debug = require('debug-levels')('Riddler')
import * as _ from 'lodash'

// const useModelCache = true
const useModelCache = false   // force new model compile every time
let testModel: TfClassifier

const Riddler = {
  async init() {
    testModel = new TfClassifier('riddles')
    await testModel.loadEncoder()  // load the sentenceEncoder

    await testModel.loadCsvInputs('./data/riddles/riddle-logs.csv', __dirname)
    await testModel.trainModel({ useCache: useModelCache })
  },

  async testRun() {
    await Riddler.init()
    const tests = [
      'are the bikes real?',
      'Did anyone go to the airport?',
      'Do criminals fly there?',
      'Did he donate something?'
    ]
    tests.map(async (item) => {
      let matches: IMatch[] | undefined = await testModel.classify(item, { maxHits: 5 })
      debug.log('test: ', item)
      // debug.log('result', matches)
      // const sources = matches![0].sources
      matches?.map((match, index) => {
        match.sources?.map(source => {
          debug.log('   - ', index, match.pct, '\t', source.text)
        })
      })
    })
  }

}

export { Riddler }

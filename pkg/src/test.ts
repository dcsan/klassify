// import { strict as assert } from 'assert';

import { Klassify } from './Klassify'

const kls = new Klassify()
const result = kls.hello('alpha')
// console.log('result:', result)
console.assert(result === 'hello alpha', 'kls.hello')
// assert.equal(result, 'hello axpal')


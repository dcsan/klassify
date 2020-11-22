import { Klassify } from '@dcsan/klassify'

const kls = new Klassify()
const result = kls.hello('alpha')
// console.log('result:', result)
console.assert(result === 'hello alpha', 'kls.hello')

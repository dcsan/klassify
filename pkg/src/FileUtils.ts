const csv = require('csv-parser');
const fs = require('fs');
import * as path from 'path'

const readCsvFile = (fp: string, basePath: string = __dirname): Promise<any[]> => {
  const fullPath = path.join(basePath, fp)

  return new Promise((resolve, reject) => {
    let lines: string[] = []
    fs.createReadStream(fullPath)
      .pipe(csv())
      .on('data', (row) => {
        lines.push(row)
      })
      .on('end', () => {
        resolve(lines)
      });
  })

}

export { readCsvFile }


test:
	DEBUG=* DEBUG_LEVEL=info ts-node src/TestRun.ts

uninstall:
	npm uninstall @dcsan/klassify

# install library from local copy
installLocal: uninstall buildPackage
	npm i -S ../pkg

# install from npm
installRemote: uninstall
	npm install @dcsan/klassify@latest

buildPackage:
	cd ../pkg && npm run build

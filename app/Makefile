
test:
	ts-node src/test.ts

uninstall:
	npm uninstall @dcsan/klassify

# install library from local copy
installLocal: uninstall
	npm i -S ../pkg

# install from npm
installRemote: uninstall
	npm install @dcsan/klassify@latest

module.exports = {
  // Transform all files using babel-jest
  transform: {
    "^.+\\.(js|jsx)$": "babel-jest"
  },
  // Transform specific node_modules that use ESM syntax
  transformIgnorePatterns: [
    "node_modules/(?!(axios)/)"
  ],
  // Setup test environment
  testEnvironment: "jsdom",
  // Setup global test environment (polyfills, etc.)
  setupFilesAfterEnv: ["<rootDir>/src/setupTests.js"],
  // Provide module name mapper for non-JS assets
  moduleNameMapper: {
    "\\.(css|less|scss|sass)$": "identity-obj-proxy",
    "\\.(jpg|jpeg|png|gif|svg|eot|otf|webp|ttf|woff|woff2|mp4|webm|wav|mp3|m4a|aac|oga)$": "<rootDir>/src/tests/__mocks__/fileMock.js"
  },
  // Allow ES modules
  moduleFileExtensions: ["js", "jsx", "json", "node"]
};
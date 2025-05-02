// Mock implementation of axios
const mockAxios = jest.fn();

// Add mock methods to the function object
mockAxios.get = jest.fn();
mockAxios.post = jest.fn();
mockAxios.put = jest.fn();
mockAxios.delete = jest.fn();

// Mock response helpers
mockAxios.mockResolvedValueOnce = jest.fn();
mockAxios.mockRejectedValueOnce = jest.fn();
mockAxios.mockImplementationOnce = jest.fn();
mockAxios.mockImplementation = jest.fn();
mockAxios.mockReset = jest.fn();
mockAxios.mockClear = jest.fn();

module.exports = mockAxios;
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export const processData = async (data, config) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/process-data`, { data, config });
    return response.data;
  } catch (error) {
    throw error;
  }
};

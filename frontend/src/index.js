/**
 * @file index.js
 * @description Entry point for the React application. Renders App with BrowserRouter.
 */
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import './styles/main.css';
import App from './App';
import { GraphDataProvider } from './context/GraphDataContext';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <GraphDataProvider>
        <App />
      </GraphDataProvider>
    </BrowserRouter>
  </React.StrictMode>
);

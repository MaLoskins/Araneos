// src/components/Sidebar.js
import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import './Sidebar.css'; // Import Sidebar-specific styles
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Button
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { FiDownload, FiBarChart2, FiDatabase, FiActivity, FiCpu } from 'react-icons/fi';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Title, Tooltip as ChartTooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';

import InfoButton from './InfoButton';
import sectionsInfo from '../sectionsInfo';

// Register Chart.js components so chartjs-2 can use them
ChartJS.register(BarElement, CategoryScale, LinearScale, Title, ChartTooltip, Legend);

function Sidebar({ graphData, csvData, requestNavigation, isSyncing, syncError, status }) {
  const [degreeData, setDegreeData] = useState({ labels: [], values: [] });

  // Basic stats about the CSV (rows & columns)
  const csvRows = csvData?.length || 0;
  const csvCols = csvData?.[0] ? Object.keys(csvData[0]).length : 0;

  // Basic stats about the graph
  const nodeCount = graphData?.nodes?.length || 0;
  const edgeCount = graphData?.links?.length || 0;

  /**
   * Compute histogram of node degrees
   * - We'll read graphData.links, count how many edges each node has, and produce a [degree -> count] map
   */
  useEffect(() => {
    if (!graphData?.nodes?.length || !graphData?.links?.length) {
      setDegreeData({ labels: [], values: [] });
      return;
    }

    const degrees = {};
    for (const node of graphData.nodes) {
      degrees[node.id] = 0;
    }
    for (const link of graphData.links) {
      degrees[link.source] = (degrees[link.source] || 0) + 1;
      degrees[link.target] = (degrees[link.target] || 0) + 1;
    }

    // Now we have { nodeId: degree, ... }
    // We want a histogram: degree -> frequency
    const degreeFreq = {};
    Object.values(degrees).forEach((deg) => {
      degreeFreq[deg] = (degreeFreq[deg] || 0) + 1;
    });

    // Sort by degree
    const sortedDegrees = Object.keys(degreeFreq).map(d => parseInt(d)).sort((a, b) => a - b);
    const labels = sortedDegrees.map(String);
    const values = sortedDegrees.map(d => degreeFreq[d]);

    setDegreeData({ labels, values });
  }, [graphData]);

  /**
   * Example function to download graph as JSON
   * (with nodes & links). You can adapt for CSV, etc.
   */
  const handleDownloadJSON = () => {
    if (!graphData) return;
    const jsonString = JSON.stringify(graphData, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = 'graph_data.json';
    link.click();

    URL.revokeObjectURL(url);
  };

  /**
   * Example: Download CSV version (very rough example).
   */
  const handleDownloadCSV = () => {
    if (!graphData) return;

    // We'll do minimal CSV: nodeId, nodeType, features
    // If you want edges or something else, do similarly
    let csvContent = 'id,type,features\n';
    graphData.nodes.forEach((node) => {
      // features might be an object. Turn into a JSON string for CSV
      const featuresStr = node.features ? JSON.stringify(node.features).replaceAll(',', ';') : '';
      csvContent += `${node.id},${node.type || ''},${featuresStr}\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = 'graph_data.csv';
    link.click();

    URL.revokeObjectURL(url);
  };

  return (
    <div className="sidebar">
      <h2>
        Sidebar
        <InfoButton
          title={sectionsInfo.sidebar.title}
          description={sectionsInfo.sidebar.description}
        />
      </h2>

      {/* Navigation Links */}
      <div className="sidebar-nav">
        <button
          className="sidebar-nav-link"
          role="tab"
          aria-label="GraphNet"
          data-testid="nav-graphnet"
          onClick={() => requestNavigation ? requestNavigation("/") : null}
          disabled={isSyncing}
          style={{ background: "none", border: "none", padding: 0, width: "100%", textAlign: "left", cursor: isSyncing ? "not-allowed" : "pointer" }}
        >
          <FiActivity className="sidebar-icon" />
          <span>GraphNet</span>
        </button>
        <button
          className="sidebar-nav-link"
          role="tab"
          aria-label="Model Training"
          data-testid="nav-model-training"
          onClick={() => requestNavigation ? requestNavigation("/train") : null}
          disabled={isSyncing}
          style={{ background: "none", border: "none", padding: 0, width: "100%", textAlign: "left", cursor: isSyncing ? "not-allowed" : "pointer" }}
        >
          <FiCpu className="sidebar-icon" />
          <span>Model Training</span>
        </button>
        {isSyncing && (
          <div
            className="sidebar-sync-status"
            data-testid="sidebar-sync-status"
            style={{ color: "#6c63ff", marginTop: 8 }}
          >
            <span>Synchronizing graph data...</span>
          </div>
        )}
        {syncError && (
          <div
            className="sidebar-sync-error"
            data-testid="sidebar-sync-error"
            style={{ color: "#d32f2f", marginTop: 8 }}
          >
            <span>
              Error syncing: {typeof syncError === "string" ? syncError : (syncError?.message || "Unknown error")}
            </span>
          </div>
        )}
      </div>

      {/* 1) Basic Graph Info */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <FiDatabase className="sidebar-icon" />
          <strong>Graph & CSV Summary</strong>
        </AccordionSummary>
        <AccordionDetails>
          <p><strong>Node Count:</strong> {nodeCount}</p>
          <p><strong>Edge Count:</strong> {edgeCount}</p>
          <p><strong>CSV Rows:</strong> {csvRows}</p>
          <p><strong>CSV Columns:</strong> {csvCols}</p>
        </AccordionDetails>
      </Accordion>

      {/* 2) Download Options */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <FiDownload className="sidebar-icon" />
          <strong>Download Graph</strong>
        </AccordionSummary>
        <AccordionDetails>
          <div className="sidebar-download-options">
            <Button
              variant="contained"
              color="primary"
              onClick={handleDownloadJSON}
              startIcon={<FiDownload />}
            >
              JSON
            </Button>
            <Button
              variant="contained"
              color="primary"
              onClick={handleDownloadCSV}
              startIcon={<FiDownload />}
            >
              CSV
            </Button>
          </div>
          <p className="sidebar-download-description">
            Downloads the current graph data (nodes, links, features) in the selected format.
          </p>
        </AccordionDetails>
      </Accordion>

      {/* 3) Charts & Visualization */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <FiBarChart2 className="sidebar-icon" />
          <strong>Node Degree Distribution</strong>
        </AccordionSummary>
        <AccordionDetails>
          {degreeData.labels.length > 0 ? (
            <div className="chart-container">
              <Bar
                data={{
                  labels: degreeData.labels,
                  datasets: [
                    {
                      label: 'Count of Nodes',
                      data: degreeData.values,
                      backgroundColor: 'rgba(153, 102, 255, 0.6)',
                      borderColor: 'rgba(153, 102, 255, 1)',
                      borderWidth: 1
                    }
                  ]
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    x: { title: { display: true, text: 'Degree' } },
                    y: { title: { display: true, text: 'Number of Nodes' } }
                  }
                }}
              />
            </div>
          ) : (
            <p>No edges or insufficient data for a degree distribution.</p>
          )}
        </AccordionDetails>
      </Accordion>
    </div>
  );
}

export default Sidebar;

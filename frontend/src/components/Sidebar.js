// src/components/Sidebar.js
import React, { useEffect, useState } from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Button,
  Tooltip,
  IconButton
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { FiDownload, FiBarChart2, FiDatabase } from 'react-icons/fi';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Title, Tooltip as ChartTooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';

import InfoButton from './InfoButton';
import sectionsInfo from '../sectionsInfo';

// Register Chart.js components so chartjs-2 can use them
ChartJS.register(BarElement, CategoryScale, LinearScale, Title, ChartTooltip, Legend);

function Sidebar({ graphData, csvData }) {
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
   * If you just want node embeddings or structure separately,
   * you’d build that CSV logic here.
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

      {/* 1) Basic Graph Info */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <FiDatabase style={{ marginRight: 8 }} />
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
          <FiDownload style={{ marginRight: 8 }} />
          <strong>Download Graph</strong>
        </AccordionSummary>
        <AccordionDetails>
          <div style={{ display: 'flex', gap: '10px' }}>
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
          <p style={{ marginTop: '10px', fontSize: '0.9em' }}>
            Downloads the current graph data (nodes, links, features) in the selected format.
          </p>
        </AccordionDetails>
      </Accordion>

      {/* 3) Charts & Visualization */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <FiBarChart2 style={{ marginRight: 8 }} />
          <strong>Node Degree Distribution</strong>
        </AccordionSummary>
        <AccordionDetails>
          {degreeData.labels.length > 0 ? (
            <div style={{ width: '100%', height: '250px' }}>
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

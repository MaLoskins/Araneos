import React, { useRef, useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import InfoButton from '../InfoButton';
import sectionsInfo from '../../sectionsInfo';

const GraphVisualizer = ({ graphData, onNodeClick }) => {
  const fgRef = useRef();
  const containerRef = useRef(null);

  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [processedGraphData, setProcessedGraphData] = useState(null);
  const [hoverNode, setHoverNode] = useState(null);

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge').strength(-200);
    }
  }, []);

  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height });
      }
    });
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    return () => {
      if (containerRef.current) {
        resizeObserver.unobserve(containerRef.current);
      }
    };
  }, [containerRef]);

  useEffect(() => {
    if (graphData?.nodes && (graphData?.links || graphData?.edges)) {
      // Always use 'links' for D3, mapping from 'edges' if necessary
      const links = graphData.links
        ? JSON.parse(JSON.stringify(graphData.links))
        : JSON.parse(JSON.stringify(graphData.edges));

      const nodes = JSON.parse(JSON.stringify(graphData.nodes));
      const nodeMap = new Map();
      nodes.forEach(node => {
        nodeMap.set(node.id, node);
      });

      // Filter out links whose source/target are not present in nodes
      const validLinks = links.filter(link => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;
        return nodeMap.has(sourceId) && nodeMap.has(targetId);
      });

      // Calculate node degrees for sizing
      const degrees = {};
      validLinks.forEach(link => {
        const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
        const targetId = typeof link.target === 'object' ? link.target.id : link.target;

        degrees[sourceId] = (degrees[sourceId] || 0) + 1;
        degrees[targetId] = (degrees[targetId] || 0) + 1;

        // Replace references directly in the links array
        link.source = sourceId;
        link.target = targetId;
      });

      // Update node properties
      nodes.forEach(node => {
        node.val = degrees[node.id] || 1;
        node.type = node.type || 'default';
        if (node.label === undefined) {
          node.label = node.id;
        }
      });

      setProcessedGraphData({
        nodes,
        links: validLinks,
        directed: graphData.directed
      });
    }
  }, [graphData]);

  // Configure the background color from CSS variables
  const backgroundColor = getComputedStyle(document.documentElement)
    .getPropertyValue('--background-color')
    .trim() || '#121212';

  // Determine if directed graph
  const isDirected = graphData && graphData.directed === true;

  return (
    <div className="graph-section">
      <h2>
        Graph Visualization
        <InfoButton
          title={sectionsInfo.graphVisualization.title}
          description={sectionsInfo.graphVisualization.description}
        />
      </h2>
      <div className="graph-container" ref={containerRef}>
        {processedGraphData && (
          <ForceGraph2D
            ref={fgRef}
            graphData={processedGraphData}
            nodeAutoColorBy="type"
            linkAutoColorBy="type"
            nodeLabel={(node) => `${node.id}${node.label && node.label !== node.id ? ` (${node.label})` : ''}`}
            linkLabel="type"
            nodeVal={(node) => node.val || 1}
            onNodeClick={(node) => {
              if (onNodeClick) onNodeClick(node);
            }}
            onNodeHover={(node) => {
              setHoverNode(node);
            }}
            onBackgroundClick={() => setHoverNode(null)}
            linkDirectionalArrowLength={isDirected ? 6 : 0}
            linkDirectionalArrowRelPos={0.5}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor={backgroundColor}
          />
        )}
        {hoverNode && (
          <div className="node-tooltip">
            <strong>ID:</strong> {hoverNode.id}<br />
            {hoverNode.label && hoverNode.label !== hoverNode.id && (
              <><strong>Label:</strong> {hoverNode.label}<br /></>
            )}
            <strong>Type:</strong> {hoverNode.type || 'default'}<br />
            {hoverNode.features && Object.keys(hoverNode.features).length > 0 && (
              <div>
                <strong>Features:</strong>
                {Object.entries(hoverNode.features).map(([key, value]) => (
                  <div key={key} style={{ marginLeft: '10px' }}>
                    <strong>{key}:</strong> {Array.isArray(value) 
                      ? value.slice(0, 3).map(f => JSON.stringify(f)).join(', ') + (value.length > 3 ? '...' : '')
                      : JSON.stringify(value)}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default GraphVisualizer;
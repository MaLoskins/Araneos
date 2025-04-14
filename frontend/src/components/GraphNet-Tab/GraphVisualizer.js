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
    if (graphData?.nodes && graphData?.links) {
      const degrees = {};
      graphData.links.forEach((link) => {
        degrees[link.source] = (degrees[link.source] || 0) + 1;
        degrees[link.target] = (degrees[link.target] || 0) + 1;
      });
      const newNodes = graphData.nodes.map((node) => ({
        ...node,
        val: degrees[node.id] || 1,
      }));
      setProcessedGraphData({ ...graphData, nodes: newNodes });
    }
  }, [graphData]);

  const isDirected = graphData && graphData.directed === true;
  const backgroundColor = getComputedStyle(document.documentElement)
    .getPropertyValue('--background-color')
    .trim() || '#121212';

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
            nodeLabel="id"
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
            <strong>Type:</strong> {hoverNode.type}<br />
            {hoverNode.features && Object.keys(hoverNode.features).map((key) => (
              <div key={key}>
                <strong>{key}:</strong> {Array.isArray(hoverNode.features[key]) ? hoverNode.features[key].slice(0, 3).map(f => JSON.stringify(f)).join(', ') + '...' : JSON.stringify(hoverNode.features[key])}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default GraphVisualizer;

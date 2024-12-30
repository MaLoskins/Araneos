// src/sectionsInfo.js

const sectionsInfo = {
  sidebar: {
    title: "Sidebar Information",
    description: `
      <p>This sidebar can be used to navigate or display additional controls.</p>
      <p>Currently it's just a placeholder, but you can customize it to your liking!</p>
    `
  },
  configurationPanel: {
    title: "Configuration Panel",
    description: `
      <p>Use this panel to select node columns and optionally define feature embeddings.</p>
      <p>Features can include text embeddings (BERT, GloVe, Word2Vec) or numeric transformations.</p>
      <p>When you finish, click "Process Graph" to build your network.</p>
      <p>
        Select which columns become <strong>nodes</strong> in the graph
        (e.g. “tweet”, “replied_to_tweet”), and optionally create edges
        in React Flow for relationships like “replied_to_tweet”.
      </p>
    `
  },
  featureColumns: {
    title: "Feature Columns",
    description: `
      <p>Here you define how embeddings are generated. Each feature has:</p>
      <ul>
        <li>node_id_column: The column that identifies each node (e.g. tweet_id).</li>
        <li>column_name: The text or numeric column for embedding.</li>
        <li>embedding_method: BERT, GloVe, Word2Vec, etc.</li>
      </ul>
      <p>These features are NOT shown as separate nodes; they attach to existing nodes in the final graph.</p>
    `
  },
  fileUploader: {
    title: "CSV File Uploader",
    description: `
      <p>Drag & drop a CSV file, or click to browse and upload.</p>
      <p>Ensure your CSV has a header row, as the parser expects column names.</p>
    `
  },
  graphFlow: {
    title: "React Flow Configuration",
    description: `
      <p>This interactive area lets you visualize and arrange your nodes. 
      You can also draw edges between nodes to define relationships.</p>
      <p>Right-click or drag edges to connect nodes, or click a node to edit its properties.</p>
    `
  },
  processGraph: {
    title: "Process Graph",
    description: `
      <p>When ready, click "Process Graph" to send data to the backend. 
      The server will build your network (and embeddings, if configured) 
      and return the final graph for visualization.</p>
    `
  },
  graphVisualization: {
    title: "Graph Visualization",
    description: `
      <p>This 2D force-directed graph shows the final network. Hover over nodes to see embeddings/features.</p>
      <p>Directed edges have arrows, while undirected edges do not.</p>
    `
  }
};

export default sectionsInfo;

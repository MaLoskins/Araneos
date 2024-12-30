// src/components/GraphNet-Tab/ConfigurationPanel.js
import React, { useState } from 'react';
import InfoButton from '../InfoButton';  // <-- import
import sectionsInfo from '../../sectionsInfo';         // <-- import

const ConfigurationPanel = ({
  columns,
  onSelectNode,
  onSubmit,
  loading,
  selectedNodes,
  useFeatureSpace,
  onToggleFeatureSpace,
  featureConfigs,
  setFeatureConfigs
}) => {
  // We add a small snippet so the user can pick "which node ID column" the feature should attach to.
  // For example, if the user wants "text" embeddings to attach to the "tweet_id" node.
  const addFeature = () => {
    setFeatureConfigs([
      ...featureConfigs,
      {
        node_id_column: '',      // <--- new
        column_name: '',         // e.g. "text"
        type: 'text',            // text or numeric
        embedding_method: 'bert',
        embedding_dim: 768,
        additional_params: {},
        data_type: 'float',
        processing: 'none',
        projection: {}
      }
    ]);
  };

  const removeFeature = (index) => {
    const updated = featureConfigs.filter((_, i) => i !== index);
    setFeatureConfigs(updated);
  };

  const updateFeature = (index, key, value) => {
    const updated = featureConfigs.map((feature, i) => {
      if (i === index) {
        return { ...feature, [key]: value };
      }
      return feature;
    });
    setFeatureConfigs(updated);
  };

  return (
    <div className="config-section">
      <h2>
        Node Selection
        <InfoButton
          title={sectionsInfo.configurationPanel.title}
          description={sectionsInfo.configurationPanel.description}
        />
      </h2>
      <div className="node-selection">
        {columns.map((col) => (
          <div key={col} className="node-selector">
            <input
              type="checkbox"
              id={`node-${col}`}
              value={col}
              checked={selectedNodes.includes(col)}
              onChange={() => onSelectNode(col)}
            />
            <label htmlFor={`node-${col}`}>{col}</label>
          </div>
        ))}
      </div>


      <hr style={{ margin: '20px 0' }} />

      <div style={{ textAlign: 'center', marginBottom: '15px' }}>
        <label style={{ marginRight: '10px' }}>
          <input
            type="checkbox"
            checked={useFeatureSpace}
            onChange={onToggleFeatureSpace}
          />
          Use advanced feature creation (BERT/GloVe/Word2Vec)?
        </label>
      </div>

      {useFeatureSpace && (
        <div className="feature-config">
          <h3>
            Feature Columns
            <InfoButton
              title={sectionsInfo.featureColumns.title}
              description={sectionsInfo.featureColumns.description}
            />
          </h3>
          {featureConfigs.map((feature, index) => (
            <div key={index} className="feature-config-item">
              <button
                type="button"
                className="remove-feature-btn"
                onClick={() => removeFeature(index)}
              >
                Remove
              </button>
              <label>
                Node ID Column:
                <select
                  value={feature.node_id_column}
                  onChange={(e) =>
                    updateFeature(index, 'node_id_column', e.target.value)
                  }
                  required
                >
                  <option value="">--Select--</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </label>

              <label>
                Feature Column Name:
                <select
                  value={feature.column_name}
                  onChange={(e) =>
                    updateFeature(index, 'column_name', e.target.value)
                  }
                  required
                >
                  <option value="">--Select--</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </label>

              <label>
                Feature Type:
                <select
                  value={feature.type}
                  onChange={(e) => updateFeature(index, 'type', e.target.value)}
                >
                  <option value="text">Text</option>
                  <option value="numeric">Numeric</option>
                </select>
              </label>

              {/* If text, choose embedding method */}
              {feature.type === 'text' && (
                <>
                  <label>
                    Embedding Method:
                    <select
                      value={feature.embedding_method}
                      onChange={(e) =>
                        updateFeature(index, 'embedding_method', e.target.value)
                      }
                    >
                      <option value="bert">BERT</option>
                      <option value="glove">GloVe</option>
                      <option value="word2vec">Word2Vec</option>
                    </select>
                  </label>
                  <label>
                    Embedding Dimension:
                    <input
                      type="number"
                      value={feature.embedding_dim}
                      onChange={(e) =>
                        updateFeature(index, 'embedding_dim', parseInt(e.target.value))
                      }
                      placeholder="e.g., 768"
                      required
                      min="50"
                      max="2048"
                    />
                  </label>
                  {feature.embedding_method === 'glove' && (
                    <label>
                      GloVe Cache Path:
                      <input
                        type="text"
                        value={feature.additional_params.glove_cache_path || ''}
                        onChange={(e) =>
                          updateFeature(index, 'additional_params', {
                            ...feature.additional_params,
                            glove_cache_path: e.target.value
                          })
                        }
                        placeholder="Path to GloVe cache"
                        required
                      />
                    </label>
                  )}
                  {feature.embedding_method === 'word2vec' && (
                    <label>
                      Word2Vec Model Path:
                      <input
                        type="text"
                        value={feature.additional_params.word2vec_model_path || ''}
                        onChange={(e) =>
                          updateFeature(index, 'additional_params', {
                            ...feature.additional_params,
                            word2vec_model_path: e.target.value
                          })
                        }
                        placeholder="Path to Word2Vec model"
                      />
                    </label>
                  )}
                  {feature.embedding_method === 'bert' && (
                    <label>
                      BERT Model Name:
                      <input
                        type="text"
                        value={
                          feature.additional_params.bert_model_name ||
                          'bert-base-uncased'
                        }
                        onChange={(e) =>
                          updateFeature(index, 'additional_params', {
                            ...feature.additional_params,
                            bert_model_name: e.target.value
                          })
                        }
                        placeholder="e.g., bert-base-uncased"
                        required
                      />
                    </label>
                  )}
                </>
              )}

              {/* If numeric, pick transformations */}
              {feature.type === 'numeric' && (
                <>
                  <label>
                    Data Type:
                    <select
                      value={feature.data_type}
                      onChange={(e) =>
                        updateFeature(index, 'data_type', e.target.value)
                      }
                    >
                      <option value="float">Float</option>
                      <option value="int">Integer</option>
                    </select>
                  </label>
                  <label>
                    Processing:
                    <select
                      value={feature.processing}
                      onChange={(e) =>
                        updateFeature(index, 'processing', e.target.value)
                      }
                    >
                      <option value="none">None</option>
                      <option value="standardize">Standardize</option>
                      <option value="normalize">Normalize</option>
                    </select>
                  </label>
                  <label>
                    Projection Method:
                    <select
                      value={feature.projection.method || 'none'}
                      onChange={(e) =>
                        updateFeature(index, 'projection', {
                          ...feature.projection,
                          method: e.target.value
                        })
                      }
                    >
                      <option value="none">None</option>
                      <option value="linear">Linear</option>
                    </select>
                  </label>
                  {feature.projection.method === 'linear' && (
                    <label>
                      Target Dimension:
                      <input
                        type="number"
                        value={feature.projection.target_dim || 1}
                        onChange={(e) =>
                          updateFeature(index, 'projection', {
                            ...feature.projection,
                            target_dim: parseInt(e.target.value)
                          })
                        }
                        placeholder="e.g., 10"
                        required
                        min="1"
                        max="2048"
                      />
                    </label>
                  )}
                </>
              )}
            </div>
          ))}
          <button
            type="button"
            className="add-feature-btn"
            onClick={addFeature}
            style={{ marginRight: 'auto' }} // add some space on the left
          >
            Add Feature
          </button>
        </div>
      )}

      {/* Add some spacing between "Add Feature" & "Process Graph" */}
      <div style={{ marginTop: '10px' }}>
        <button
          onClick={onSubmit}
          disabled={loading || selectedNodes.length === 0}
          style={{ marginTop: '10px' }}
        >
          {loading ? 'Processing...' : 'Process Graph'}
        </button>
        <InfoButton
          title={sectionsInfo.processGraph.title}
          description={sectionsInfo.processGraph.description}
        />
      </div>
    </div>
  );
};

export default ConfigurationPanel;
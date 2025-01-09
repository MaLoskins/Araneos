// src/components/GraphNet-Tab/ConfigurationPanel.js

import React, { useState } from 'react';
import { Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { FiList, FiSettings, FiPlay } from 'react-icons/fi';
import InfoButton from '../InfoButton';
import sectionsInfo from '../../sectionsInfo';

/**
 * Minimal criteria to treat a feature as "complete":
 *  - node_id_column, column_name, and type must be set.
 *  - if type === 'text', check embedding_dim (or any other fields you consider mandatory).
 *  - if type === 'numeric', check data_type or others if you like.
 */
function isFeatureComplete(feature) {
  const { node_id_column, column_name, type } = feature;
  if (!node_id_column || !column_name || !type) return false;
  if (type === 'text' && !feature.embedding_dim) return false;
  return true;
}

const ConfigurationPanel = ({
  columns,
  onSelectNode,
  onSubmit, // function(labelColumn)
  loading,
  selectedNodes,
  useFeatureSpace,
  onToggleFeatureSpace,
  featureConfigs,
  setFeatureConfigs
}) => {
  // Local state for “expanded” feature panels
  const [expandedIndices, setExpandedIndices] = useState(
    featureConfigs.map(() => true)
  );

  // Local state for chosen label column
  const [labelColumn, setLabelColumn] = useState('');

  React.useEffect(() => {
    if (expandedIndices.length < featureConfigs.length) {
      setExpandedIndices((prev) => [
        ...prev,
        ...Array(featureConfigs.length - prev.length).fill(true),
      ]);
    }
  }, [featureConfigs, expandedIndices]);

  const toggleExpand = (index) => {
    setExpandedIndices((prev) => {
      const updated = [...prev];
      updated[index] = !prev[index];
      return updated;
    });
  };

  const addFeature = () => {
    setFeatureConfigs((prev) => [
      ...prev,
      {
        node_id_column: '',
        column_name: '',
        type: 'text',
        embedding_method: 'bert',
        embedding_dim: 768,
        additional_params: {},
        data_type: 'float',
        processing: 'none',
        projection: {},
      },
    ]);
    setExpandedIndices((prev) => [...prev, true]);
  };

  const removeFeature = (index) => {
    const updated = featureConfigs.filter((_, i) => i !== index);
    setFeatureConfigs(updated);
    const updatedExpanded = expandedIndices.filter((_, i) => i !== index);
    setExpandedIndices(updatedExpanded);
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

  function renderFeatureSummary(feature, index) {
    const {
      node_id_column,
      column_name,
      type,
      embedding_method,
      embedding_dim,
      data_type,
      processing,
      projection
    } = feature;

    return (
      <div
        style={{
          background: 'var(--input-background)',
          padding: '10px',
          borderRadius: '4px',
          marginBottom: '10px',
          fontSize: '0.85rem',
        }}
      >
        <strong>Feature {index + 1}</strong>
        <div style={{ marginTop: '6px' }}>
          <p><strong>Node ID:</strong> {node_id_column}</p>
          <p><strong>Column:</strong> {column_name}</p>
          <p><strong>Type:</strong> {type}</p>
          {type === 'text' && (
            <p>
              <strong>Embedding:</strong> {embedding_method} ({embedding_dim}D)
            </p>
          )}
          {type === 'numeric' && (
            <>
              <p><strong>Data Type:</strong> {data_type}</p>
              <p><strong>Processing:</strong> {processing}</p>
              <p>
                <strong>Projection:</strong> {projection.method || 'none'}
              </p>
            </>
          )}
        </div>
      </div>
    );
  }

  function renderEditForm(feature, index) {
    return (
      <div style={{ fontSize: '0.85rem' }}>
        <label>
          Node ID Column:
          <select
            value={feature.node_id_column}
            onChange={(e) => updateFeature(index, 'node_id_column', e.target.value)}
            required
          >
            <option value="">--Select--</option>
            {columns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </label>

        <label>
          Feature Column Name:
          <select
            value={feature.column_name}
            onChange={(e) => updateFeature(index, 'column_name', e.target.value)}
            required
          >
            <option value="">--Select--</option>
            {columns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
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

        {/* text-based feature config */}
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
                      glove_cache_path: e.target.value,
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
                      word2vec_model_path: e.target.value,
                    })
                  }
                  placeholder="Path to Word2Vec model"
                />
              </label>
            )}
            {feature.embedding_method === 'bert' && (
              <>
                <label>
                  BERT Model Name:
                  <input
                    type="text"
                    value={feature.additional_params.bert_model_name || 'bert-base-uncased'}
                    onChange={(e) =>
                      updateFeature(index, 'additional_params', {
                        ...feature.additional_params,
                        bert_model_name: e.target.value,
                      })
                    }
                    placeholder="e.g., bert-base-uncased"
                    required
                  />
                </label>
                <label>
                  BERT Batch Size (optional):
                  <input
                    type="number"
                    min="1"
                    max="512"
                    value={feature.additional_params.bert_batch_size || ''}
                    onChange={(e) =>
                      updateFeature(index, 'additional_params', {
                        ...feature.additional_params,
                        bert_batch_size: parseInt(e.target.value) || 1,
                      })
                    }
                    placeholder="e.g., 16"
                  />
                </label>
              </>
            )}
          </>
        )}

        {/* numeric-based feature config */}
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
                    method: e.target.value,
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
                      target_dim: parseInt(e.target.value),
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
    );
  }

  return (
    <div className="config-section">
      <Accordion
        sx={{
          backgroundColor: 'var(--primary-color)',
          color: 'var(--text-color)',
          border: '1px solid var(--border-color)',
          marginBottom: '10px',
        }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ color: 'var(--text-color)' }} />}
          sx={{
            backgroundColor: 'var(--primary-color)',
            '& .MuiAccordionSummary-content': { alignItems: 'center' },
          }}
        >
          <FiList style={{ marginRight: 8 }} />
          <strong>Node Selection</strong>
          <InfoButton
            title={sectionsInfo.configurationPanel.title}
            description={sectionsInfo.configurationPanel.description}
          />
        </AccordionSummary>
        <AccordionDetails
          sx={{
            backgroundColor: 'var(--secondary-color)',
          }}
        >
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

          {/* Label Column Picker */}
          <div style={{ marginBottom: '15px', textAlign: 'center' }}>
            <label style={{ marginRight: '10px' }}>
              Choose Label Column:&nbsp;
              <select
                value={labelColumn}
                onChange={(e) => setLabelColumn(e.target.value)}
                style={{ minWidth: '120px' }}
              >
                <option value="">--None--</option>
                {columns.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>
          </div>

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
        </AccordionDetails>
      </Accordion>

      {useFeatureSpace && (
        <Accordion
          sx={{
            backgroundColor: 'var(--primary-color)',
            color: 'var(--text-color)',
            border: '1px solid var(--border-color)',
            marginBottom: '10px',
          }}
        >
          <AccordionSummary
            expandIcon={<ExpandMoreIcon sx={{ color: 'var(--text-color)' }} />}
            sx={{
              backgroundColor: 'var(--primary-color)',
            }}
          >
            <FiSettings style={{ marginRight: 8 }} />
            <strong>Advanced Feature Creation</strong>
            <InfoButton
              title={sectionsInfo.featureColumns.title}
              description={sectionsInfo.featureColumns.description}
            />
          </AccordionSummary>
          <AccordionDetails
            sx={{
              backgroundColor: 'var(--secondary-color)',
            }}
          >
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
                gap: '16px',
              }}
            >
              {featureConfigs.map((feature, index) => {
                const complete = isFeatureComplete(feature);
                const expanded = expandedIndices[index];

                return (
                  <div
                    key={index}
                    className="feature-config-item"
                    style={{
                      background: 'var(--primary-color)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '5px',
                      padding: '12px',
                      boxSizing: 'border-box',
                      position: 'relative',
                    }}
                  >
                    <button
                      type="button"
                      className="remove-feature-btn"
                      onClick={() => removeFeature(index)}
                      style={{
                        background: '#ff4d4d',
                        color: '#fff',
                        border: 'none',
                        padding: '4px 6px',
                        cursor: 'pointer',
                        fontSize: '0.8rem',
                        borderRadius: '4px',
                        position: 'absolute',
                        top: '8px',
                        right: '8px',
                      }}
                    >
                      Remove
                    </button>

                    {complete && !expanded && (
                      <>
                        {renderFeatureSummary(feature, index)}
                        <button
                          onClick={() => toggleExpand(index)}
                          style={{
                            marginTop: '4px',
                            fontSize: '0.85rem',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            background: 'var(--button-background)',
                            color: 'var(--button-text-color)',
                            border: 'none',
                          }}
                        >
                          Edit
                        </button>
                      </>
                    )}
                    {(!complete || expanded) && (
                      <>
                        {complete && (
                          <div style={{ marginBottom: '8px' }}>
                            {renderFeatureSummary(feature, index)}
                          </div>
                        )}
                        {renderEditForm(feature, index)}
                        <button
                          onClick={() => toggleExpand(index)}
                          style={{
                            marginTop: '6px',
                            fontSize: '0.85rem',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            background: 'var(--button-background)',
                            color: 'var(--button-text-color)',
                            border: 'none',
                          }}
                        >
                          {complete ? 'Close' : 'Done'}
                        </button>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
            <button
              type="button"
              className="add-feature-btn"
              onClick={addFeature}
              style={{
                marginTop: '16px',
                fontSize: '0.9rem',
                padding: '6px 12px',
                borderRadius: '4px',
                cursor: 'pointer',
                background: 'var(--button-background)',
                color: 'var(--button-text-color)',
                border: 'none',
              }}
            >
              Add Feature
            </button>
          </AccordionDetails>
        </Accordion>
      )}

      <Accordion
        sx={{
          backgroundColor: 'var(--primary-color)',
          color: 'var(--text-color)',
          border: '1px solid var(--border-color)',
        }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ color: 'var(--text-color)' }} />}
          sx={{
            backgroundColor: 'var(--primary-color)',
          }}
        >
          <FiPlay style={{ marginRight: 8 }} />
          <strong>Process Graph</strong>
          <InfoButton
            title={sectionsInfo.processGraph.title}
            description={sectionsInfo.processGraph.description}
          />
        </AccordionSummary>
        <AccordionDetails
          sx={{
            backgroundColor: 'var(--secondary-color)',
          }}
        >
          <div style={{ marginTop: '10px', textAlign: 'center' }}>
            <button
              onClick={() => onSubmit(labelColumn)}
              disabled={loading || selectedNodes.length === 0}
              style={{ marginTop: '10px' }}
            >
              {loading ? 'Processing...' : 'Process Graph'}
            </button>
          </div>
        </AccordionDetails>
      </Accordion>
    </div>
  );
};

export default ConfigurationPanel;

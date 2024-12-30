import React, { useState, useEffect } from 'react';
import Modal from 'react-modal';

Modal.setAppElement('#root');

const NodeEditModal = ({ isOpen, onRequestClose, node, onSaveNodeEdit }) => {
  const [nodeType, setNodeType] = useState(node.type || '');
  const [nodeFeatures, setNodeFeatures] = useState(node.features || []);

  useEffect(() => {
    if (isOpen) {
      setNodeType(node.type || '');
      setNodeFeatures(node.features || []);
    }
  }, [isOpen, node]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSaveNodeEdit({ nodeType, nodeFeatures });
  };

  return (
    <Modal
      isOpen={isOpen}
      onRequestClose={onRequestClose}
      contentLabel="Edit Node"
      className="node-edit-modal"
      overlayClassName="overlay"
    >
      <h2>Edit Node: {node.id}</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="node-type">Type:</label>
          <input
            type="text"
            id="node-type"
            value={nodeType}
            onChange={(e) => setNodeType(e.target.value)}
            placeholder="e.g., User, Post"
          />
        </div>
        <div className="modal-buttons">
          <button type="submit">Save</button>
          <button type="button" onClick={onRequestClose}>
            Cancel
          </button>
        </div>
      </form>
    </Modal>
  );
};

export default NodeEditModal;

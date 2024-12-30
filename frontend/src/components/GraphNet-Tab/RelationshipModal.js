import React, { useState } from 'react';
import Modal from 'react-modal';

Modal.setAppElement('#root');

const RelationshipModal = ({ isOpen, onRequestClose, onSaveRelationship }) => {
  const [relationshipType, setRelationshipType] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSaveRelationship({ relationshipType });
  };

  return (
    <Modal
      isOpen={isOpen}
      onRequestClose={onRequestClose}
      contentLabel="Define Relationship"
      className="node-edit-modal relationship-modal"
      overlayClassName="overlay"
    >
      <h2>Define Relationship</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="rel-type">Relationship Type:</label>
          <input
            type="text"
            id="rel-type"
            value={relationshipType}
            onChange={(e) => setRelationshipType(e.target.value)}
            placeholder="e.g., connects, influences"
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

export default RelationshipModal;

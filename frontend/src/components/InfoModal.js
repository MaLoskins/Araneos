// src/components/InfoModal.js
import React from 'react';
import Modal from 'react-modal';

Modal.setAppElement('#root');

const InfoModal = ({ isOpen, onRequestClose, title, description }) => {
  return (
    <Modal
      isOpen={isOpen}
      onRequestClose={onRequestClose}
      contentLabel="Information Modal"
      className="node-edit-modal info-modal-content"
      overlayClassName="overlay"
    >
      <h2 className="info-modal-title">{title}</h2>
      <div 
        className="info-modal-description" 
        dangerouslySetInnerHTML={{ __html: description }}
      />
      <div className="modal-buttons">
        <button onClick={onRequestClose}>Close</button>
      </div>
    </Modal>
  );
};

export default InfoModal;

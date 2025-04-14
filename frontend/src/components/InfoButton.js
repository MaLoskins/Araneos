// src/components/InfoButton.js
import React, { useState } from 'react';
import { FiInfo } from 'react-icons/fi';
import InfoModal from './InfoModal';

/**
 * @param {Object} props
 * @param {string} props.title - The title for the info modal.
 * @param {string} props.description - The description text for the info modal (HTML allowed).
 */
const InfoButton = ({ title, description }) => {
  const [modalOpen, setModalOpen] = useState(false);

  const handleOpen = () => setModalOpen(true);
  const handleClose = () => setModalOpen(false);

  return (
    <span className="info-button-container">
      <FiInfo onClick={handleOpen} className="info-icon" />
      <InfoModal 
        isOpen={modalOpen} 
        onRequestClose={handleClose} 
        title={title} 
        description={description}
      />
    </span>
  );
};

export default InfoButton;

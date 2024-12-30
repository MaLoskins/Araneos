import React from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import InfoButton from '../InfoButton';
import sectionsInfo from '../../sectionsInfo';

const FileUploader = ({ onFileDrop }) => {
  const onDrop = (acceptedFiles) => {
    if (!acceptedFiles.length) return;
    const file = acceptedFiles[0];

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const data = results.data;
        const fields = results.meta.fields;
        onFileDrop(data, fields);
      },
      error: (error) => {
        console.error('Error parsing CSV:', error);
        alert('Error parsing CSV file.');
      },
    });
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: '.csv',
    multiple: false,
  });

  return (
    <div className="file-uploader">
      <h3 style={{ textAlign: 'center' }}>
        Upload Your CSV
        <InfoButton
          title={sectionsInfo.fileUploader.title}
          description={sectionsInfo.fileUploader.description}
        />
      </h3>
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''}`}
        style={{ marginTop: '10px' }}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the CSV file here...</p>
        ) : (
          <p>Drag & drop a CSV file here, or click to select file</p>
        )}
      </div>
    </div>
  );
};

export default FileUploader;
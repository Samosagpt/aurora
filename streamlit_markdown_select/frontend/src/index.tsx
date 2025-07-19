import React from 'react';
import ReactDOM from 'react-dom/client';
import StreamlitMarkdownSelect from './StreamlitMarkdownSelect.tsx';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <StreamlitMarkdownSelect />
  </React.StrictMode>
);